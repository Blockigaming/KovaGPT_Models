"""Read-only live watchdog admission and narrowly scoped cleanup for a screen.

The external authorized operator creates the two disposable groups/watchdog
first, then calls verify_before_vm BEFORE allocating the VM. Nothing here
creates resources. This reuses the independently checked ARM watchdog shape;
the guest's process timeout is never treated as deallocation.
"""

from datetime import datetime, timezone
import re
import time

from training.a35_nova_screen import need


def scopes(subscription, run_id):
    need(re.fullmatch(r"[a-f0-9-]{36}", subscription)
         and re.fullmatch(r"[a-f0-9]{32}", run_id), "invalid exclusive scope")
    prefix = "/subscriptions/" + subscription + "/resourceGroups/"
    suffix = run_id[:10]
    pilot = prefix + "kova-a35-nova-" + suffix
    control = prefix + "kova-a35-watch-" + suffix
    return {"pilot_resource_group_id": pilot, "watchdog_resource_group_id": control,
            "vm_id": pilot + "/providers/Microsoft.Compute/virtualMachines/kova-t4-" + suffix,
            "watchdog_id": control + "/providers/Microsoft.Logic/workflows/kova-pilot-watchdog-" + suffix}


def verify_before_vm(subscription, run_id, allocation_started_epoch, read, *, now=None):
    """Fresh ARM reads; booleans or an operator-written observation are insufficient."""
    from training.cosmo_controller_azure import AzureRequestVerifier, watchdog_definition, CONTRIBUTOR, ARM
    now = time.time() if now is None else now
    need(type(allocation_started_epoch) is int
         and allocation_started_epoch <= now < allocation_started_epoch + 600,
         "watchdog setup exceeded reserved startup time")
    context = scopes(subscription, run_id)
    pilot, control = context["pilot_resource_group_id"], context["watchdog_resource_group_id"]
    deadline = datetime.fromtimestamp(allocation_started_epoch + 4500, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    watchdog = read(ARM + context["watchdog_id"] + "?api-version=2019-05-01")
    principal = watchdog["identity"]["principalId"]
    need(watchdog["id"].lower() == context["watchdog_id"].lower()
         and watchdog["identity"]["type"] == "SystemAssigned"
         and re.fullmatch(r"[a-f0-9-]{36}", principal)
         and watchdog["location"].lower() == "eastus", "watchdog identity/scope drift")
    props = watchdog["properties"]
    need(props["state"] == "Enabled" and props["provisioningState"] == "Succeeded"
         and props["parameters"] == {"deadlineUtc": {"value": deadline}}
         and props["definition"] == watchdog_definition(context["vm_id"], pilot, control),
         "watchdog deadline, cleanup actions or enabled state drift")
    trigger = read(ARM + context["watchdog_id"] + "/triggers/every_minute?api-version=2019-05-01")
    need(trigger["properties"]["state"] == "Enabled", "watchdog trigger is not enabled")
    for group in (pilot, control):
        roles = read(ARM + group + "/providers/Microsoft.Authorization/roleAssignments?api-version=2022-04-01")
        need(not roles.get("nextLink") and type(roles.get("value")) is list, "incomplete role inventory")
        matches = [x for x in roles["value"] if x["properties"]["principalId"].lower() == principal.lower()]
        need(len(matches) == 1 and matches[0]["properties"]["scope"].lower() == group.lower()
             and matches[0]["properties"]["roleDefinitionId"].lower().endswith("/" + CONTRIBUTOR),
             "watchdog lacks exact cleanup grant")
    # Use the existing exhaustive inherited/child lock and deny-assignment
    # checks. No identity token or guest-supplied observation is needed here.
    verifier = object.__new__(AzureRequestVerifier)
    verifier.lifecycle = context
    verifier.read = read
    verifier.unlocked_cleanup_scopes()
    verifier.cleanup_scopes_without_denials()
    inventory = read(ARM + pilot + "/resources?api-version=2021-04-01")
    need(inventory.get("value") == [] and not inventory.get("nextLink"),
         "VM group is not empty: allocation/retry rejected")
    control_inventory = read(ARM + control + "/resources?api-version=2021-04-01")
    need(not control_inventory.get("nextLink") and
         [item["id"].lower() for item in control_inventory["value"]] == [context["watchdog_id"].lower()],
         "unrelated control resources rejected")
    return {"live_watchdog_verified": True, "observed_at_epoch": int(now),
            "watchdog_deadline_epoch": allocation_started_epoch + 4500,
            "allocation_deadline_epoch": allocation_started_epoch + 5400,
            "scope": context}


def cleanup(subscription, run_id, call, *, clock=time.monotonic, sleep=time.sleep):
    """Called in the external operator's finally block on success/failure.

    `call` is the authorized ARM transport returning (HTTP status, JSON body).
    Cleanup retries are permitted; training/allocation retries are not.
    The independent watchdog remains responsible if this controller disappears.
    """
    context = scopes(subscription, run_id)
    base = "https://management.azure.com"
    vm = base + context["vm_id"]
    pilot = base + context["pilot_resource_group_id"] + "?api-version=2022-09-01"
    control = base + context["watchdog_resource_group_id"] + "?api-version=2022-09-01"
    end = clock() + 900
    while clock() < end:
        # A failed deallocation must never suppress group deletion. An accepted
        # DELETE is not evidence that disks/NAT/IPs have stopped billing.
        try:
            call("POST", vm + "/deallocate?api-version=2024-07-01")
        except (OSError, TimeoutError):
            pass
        try:
            call("DELETE", pilot)
            status, _ = call("GET", pilot)
        except (OSError, TimeoutError):
            status = None
        if status == 404:
            call("DELETE", control)
            state, _ = call("GET", control)
            if state == 404:
                return {"vm_group_absent": True, "control_group_absent": True}
        sleep(10)
    raise TimeoutError("cleanup unconfirmed; independent watchdog remains armed; no retry")
