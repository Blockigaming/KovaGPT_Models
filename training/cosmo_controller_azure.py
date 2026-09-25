"""Read-only Azure verification for the existing Cosmo grant issuer.

Controller configuration is trusted, never supplied by the guest. The transport
uses controller credentials only for ARM GETs; guest tokens go only to PyJWT.
No resource creation, registration, role writes or training is implemented here.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import os
import re
import subprocess
import urllib.request
from urllib.parse import urlencode, urlsplit

import jwt

from training import cosmo_lifecycle_authority as authority
from training import cosmo_qlora_grant as client
from training import cosmo_qlora_launch as launch
from training.cosmo_controller_ledger import AzureBlobIO, LedgerRejected, digest, need, parse_json, utc_now

ARM = "https://management.azure.com"
CONTRIBUTOR = "b24988ac-6180-42a0-ab88-20f7382dd24c"


def cli_token(resource):
    """Explicit controller-host Azure CLI session; no guest credentials accepted."""
    need(resource in (ARM + "/", "https://storage.azure.com/"), "invalid credential audience")
    try:
        result = subprocess.run(["az", "account", "get-access-token", "--resource", resource,
                                 "--output", "json", "--only-show-errors"],
                                capture_output=True, check=True, timeout=10)
        need(len(result.stdout) <= 65536, "credential response too large")
        return parse_json(result.stdout)["accessToken"]
    except (OSError, subprocess.SubprocessError, KeyError, ValueError):
        raise LedgerRejected("controller Azure credential unavailable") from None


def managed_identity_token(resource, *, client_id, environ=None):
    """Container Apps' local token endpoint, bound to one controller identity.

    The host supplies the rotating endpoint/header. Never use the guest token,
    an arbitrary URL, a proxy, redirects or a default identity selection.
    """
    need(resource in (ARM + "/", "https://storage.azure.com/"), "invalid credential audience")
    need(type(client_id) is str and launch.UUID.fullmatch(client_id), "pinned controller client ID required")
    env = os.environ if environ is None else environ
    endpoint, secret = env.get("IDENTITY_ENDPOINT"), env.get("IDENTITY_HEADER")
    try:
        parts = urlsplit(endpoint)
        need(parts.scheme == "http" and parts.hostname in ("localhost", "127.0.0.1", "::1") and
             parts.port is not None and parts.path.startswith("/") and
             not parts.query and not parts.fragment and not parts.username and not parts.password and
             type(secret) is str and re.fullmatch(r"[a-zA-Z0-9-]{16,256}", secret),
             "trusted local managed identity endpoint required")
        query = urlencode({"resource": resource, "api-version": "2019-08-01", "client_id": client_id})
        request = urllib.request.Request(endpoint + "?" + query, headers={"X-IDENTITY-HEADER": secret})

        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args):
                return None

        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        with opener.open(request, timeout=5) as response:
            raw = response.read(65537)
            need(response.status == 200 and len(raw) <= 65536, "identity response invalid")
        value = parse_json(raw)
        token, expires = value["access_token"], value["expires_on"]
        need(value["token_type"] == "Bearer" and value["resource"] == resource and
             value["client_id"].casefold() == client_id.casefold() and
             type(expires) is str and re.fullmatch(r"[0-9]{10,12}", expires) and
             int(expires) > int(utc_now().timestamp()) + 60 and
             type(token) is str and re.fullmatch(r"[A-Za-z0-9._~+/=-]{32,16384}", token),
             "managed identity token response mismatch")
        return token
    except (OSError, KeyError, TypeError, AttributeError, ValueError, OverflowError):
        raise LedgerRejected("controller managed identity unavailable") from None


class AzureReadIO:
    def __init__(self, *, account, tenant_id, token_for=cli_token):
        need(bool(launch.UUID.fullmatch(tenant_id)), "trusted tenant required")
        self.tenant = tenant_id
        self.arm = AzureBlobIO(account=account, token_for=token_for)

    def __call__(self, url):
        if url.startswith(ARM + "/"):
            result = self.arm("GET", url, {})
            need(result.status == 200, "Azure resource read failed")
            return parse_json(result.body)
        # Tenant-specific keys only: token jku/x5u fields never select a URL.
        need(url in (f"https://login.microsoftonline.com/{self.tenant}/discovery/keys",
                     f"https://login.microsoftonline.com/{self.tenant}/discovery/v2.0/keys"),
             "untrusted signing-key endpoint")
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args):
                return None
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
            with opener.open(url, timeout=5) as response:
                raw = response.read(262145)
                need(response.status == 200 and len(raw) <= 262144, "invalid signing-key response")
                return parse_json(raw)
        except (OSError, ValueError):
            raise LedgerRejected("Entra signing keys unavailable") from None


def watchdog_definition(vm_id, pilot_group_id, watchdog_group_id):
    """Expected deployed shape of the existing watchdog template."""
    def action(method, target):
        return {"type": "Http", "inputs": {"method": method, "uri": ARM + target,
            "authentication": {"type": "ManagedServiceIdentity", "audience": ARM + "/"}}}
    deallocate = action("POST", vm_id + "/deallocate?api-version=2024-07-01")
    deallocate["runAfter"] = {}
    delete = action("DELETE", pilot_group_id + "?api-version=2022-09-01")
    delete["runAfter"] = {"deallocate_after_deadline": ["Succeeded", "Failed", "TimedOut"]}
    self_delete = action("DELETE", watchdog_group_id + "?api-version=2022-09-01")
    self_delete["runAfter"] = {"delete_pilot_group": ["Succeeded"]}
    return {"$schema": "https://schema.management.azure.com/schemas/2016-06-01/Microsoft.Logic.json#",
        "contentVersion": "1.0.0.0", "parameters": {"deadlineUtc": {"type": "String"}},
        "triggers": {"every_minute": {"type": "Recurrence",
            "recurrence": {"frequency": "Minute", "interval": 1},
            "conditions": ["@greaterOrEquals(ticks(utcNow()), ticks(parameters('deadlineUtc')))" ]}},
        "actions": {"deallocate_after_deadline": deallocate, "delete_pilot_group": delete,
                    "delete_watchdog_group": self_delete},
        "outputs": {}}


def expected_rules():
    result = []
    for name, priority, direction, access, protocol, port in (
        ("allow-https-egress", 100, "Outbound", "Allow", "Tcp", "443"),
        ("allow-http-package-egress", 110, "Outbound", "Allow", "Tcp", "80"),
        ("deny-other-egress", 120, "Outbound", "Deny", "*", "*"),
        ("deny-all-inbound", 100, "Inbound", "Deny", "*", "*")):
        result.append({"name": name, "properties": {"priority": priority, "direction": direction,
            "access": access, "protocol": protocol, "sourcePortRange": "*",
            "destinationPortRange": port, "sourceAddressPrefix": "*", "destinationAddressPrefix": "*"}})
    return result


class AzureRequestVerifier:
    def __init__(self, *, tenant_id, token_version, lifecycle, watchdog_id, read_json, clock=utc_now):
        need(type(tenant_id) is str and launch.UUID.fullmatch(tenant_id), "trusted tenant required")
        need(token_version in ("1.0", "2.0"), "explicit token version required")
        self.tenant, self.version = tenant_id, token_version
        self.lifecycle, self.watchdog = deepcopy(lifecycle), watchdog_id
        authority_group = lifecycle["watchdog_resource_group_id"]
        need(type(watchdog_id) is str and re.fullmatch(
            re.escape(authority_group) + r"/providers/Microsoft.Logic/workflows/kova-pilot-watchdog-[a-z0-9]{3,10}",
            watchdog_id, re.IGNORECASE), "watchdog outside trusted cleanup group")
        self.read, self.clock = read_json, clock

    def resource(self, resource_id, api):
        group = self.lifecycle["pilot_resource_group_id"].split("/resourceGroups/")[0]
        need(type(resource_id) is str and resource_id.casefold().startswith(group.casefold() + "/") and
             not any(x in resource_id for x in ("?", "#", "..", "%", "\\")), "untrusted ARM resource")
        value = self.read(ARM + resource_id + "?api-version=" + api)
        need(type(value) is dict and value.get("id", "").casefold() == resource_id.casefold(),
             "Azure response resource mismatch")
        return value

    def identity(self, token, audience, instance):
        try:
            need(type(token) is str and len(token) <= 16384 and authority.JWT.fullmatch(token),
                 "invalid identity token")
            header = jwt.get_unverified_header(token)
            need(header.get("alg") == "RS256" and header.get("typ") == "JWT" and
                 not any(k in header for k in ("crit", "jku", "jwk", "x5u")) and
                 type(header.get("kid")) is str, "unsupported identity token")
            path = "discovery/keys" if self.version == "1.0" else "discovery/v2.0/keys"
            issuer = (f"https://sts.windows.net/{self.tenant}/" if self.version == "1.0"
                      else f"https://login.microsoftonline.com/{self.tenant}/v2.0")
            document = self.read(f"https://login.microsoftonline.com/{self.tenant}/{path}")
            keys = [k for k in document["keys"] if k.get("kid") == header["kid"]]
            need(len(keys) == 1 and keys[0].get("kty") == "RSA" and
                 keys[0].get("use") == "sig" and keys[0].get("alg", "RS256") == "RS256",
                 "Entra signing key missing")
            key = keys[0]
            # Entra may publish a tenant placeholder on the signing key itself.
            need(key.get("issuer", issuer).replace("{tenantid}", self.tenant) == issuer,
                 "signing-key issuer mismatch")
            claims = jwt.decode(token, jwt.PyJWK(key, algorithm="RS256").key,
                algorithms=["RS256"], audience=audience, issuer=issuer,
                options={"require": ["exp", "nbf", "iat", "aud", "iss", "tid", "oid", "xms_mirid", "ver"],
                         "strict_aud": True})
            now = self.clock().timestamp()
            need(all(type(claims[k]) is int for k in ("exp", "nbf", "iat")) and
                 claims["nbf"] <= now < claims["exp"] and claims["iat"] <= now and
                 claims["tid"] == self.tenant and claims["ver"] == self.version and
                 claims["oid"].casefold() == instance["system_assigned_identity_principal_id"].casefold() and
                 claims["xms_mirid"].casefold() == instance["resource_id"].casefold(),
                 "token does not identify the pinned VM")
            return datetime.fromtimestamp(claims["exp"], timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except (jwt.PyJWTError, KeyError, TypeError, AttributeError, ValueError):
            raise LedgerRejected("Entra identity rejected") from None

    def unlocked_cleanup_scopes(self):
        groups = [self.lifecycle[k].casefold() for k in
                  ("pilot_resource_group_id", "watchdog_resource_group_id")]
        subscription = groups[0].split("/resourcegroups/")[0]
        # The unfiltered subscription list includes child resource locks as well
        # as locks inherited from the subscription. Incomplete evidence rejects.
        locks = self.read(ARM + subscription +
            "/providers/Microsoft.Authorization/locks?api-version=2016-09-01")
        need(type(locks) is dict and type(locks.get("value")) is list and
             not locks.get("nextLink"), "cleanup lock inventory incomplete")
        for lock in locks["value"]:
            rid = lock["id"].casefold()
            parts = rid.rsplit("/providers/microsoft.authorization/locks/", 1)
            need(len(parts) == 2 and parts[1] and "/" not in parts[1] and
                 (parts[0] == subscription or parts[0].startswith(subscription + "/")),
                 "invalid management lock scope")
            need(not any(parts[0] == group or group.startswith(parts[0] + "/") or
                         parts[0].startswith(group + "/") for group in groups),
                 "management lock prevents bounded cleanup")

    def exact_group_resources(self, group, required, optional=()):
        inventory = self.read(ARM + group + "/resources?api-version=2021-04-01")
        need(type(inventory) is dict and type(inventory.get("value")) is list and
             not inventory.get("nextLink"), "cleanup group inventory incomplete")
        ids = [item.get("id") for item in inventory["value"] if type(item) is dict]
        need(len(ids) == len(inventory["value"]) and
             all(type(rid) is str and rid.casefold().startswith(group.casefold() + "/providers/")
                 for rid in ids), "invalid cleanup group inventory")
        seen = [rid.casefold() for rid in ids]
        needed = {rid.casefold() for rid in required}
        allowed = needed | {rid.casefold() for rid in optional}
        need(len(seen) == len(set(seen)) and needed <= set(seen) <= allowed,
             "unexpected or missing cleanup group resource")

    def __call__(self, *, token, audience, instance, network, cleanup_trigger_utc):
        try:
            return self.verify(token=token, audience=audience, instance=instance,
                               network=network, cleanup_trigger_utc=cleanup_trigger_utc)
        except (KeyError, TypeError, AttributeError, ValueError, OSError):
            raise LedgerRejected("independent Azure verification rejected") from None

    def verify(self, *, token, audience, instance, network, cleanup_trigger_utc):
        before = self.clock()
        instance = authority.validate_azure_instance(instance)
        client.verified_network(network, instance, current=before)
        expiry = self.identity(token, audience, instance)
        pilot = self.lifecycle["pilot_resource_group_id"]
        need(instance["resource_id"].casefold().startswith(pilot.casefold() + "/providers/microsoft.compute/virtualmachines/"),
             "VM outside pilot group")
        vm = self.resource(instance["resource_id"], "2024-07-01")
        props, identity = vm["properties"], vm["identity"]
        image = props["storageProfile"]["imageReference"]
        need(vm["location"].lower() == "eastus" and props["provisioningState"] == "Succeeded" and
             props["vmId"].casefold() == instance["vm_id"].casefold() and
             props["hardwareProfile"]["vmSize"] == launch.SKU and
             identity["type"] == "SystemAssigned" and identity["tenantId"] == self.tenant and
             identity["principalId"].casefold() == instance["system_assigned_identity_principal_id"].casefold() and
             image.get("publisher") == "Canonical" and image.get("offer") == "ubuntu-24_04-lts" and
             image.get("sku") == "server" and image.get("version") == "24.04.202609040" and
             image.get("exactVersion", "24.04.202609040") == "24.04.202609040",
             "live VM identity, SKU or image mismatch")
        # The VM's IMDS token is used only to prove its identity to the issuer.
        # No ARM role is needed by the guest. Check direct assignments at,
        # above and below the subscription, then effective group assignments.
        subscription = pilot.split("/resourceGroups/")[0]
        principal = instance["system_assigned_identity_principal_id"]
        guest_roles = self.read(ARM + subscription +
            "/providers/Microsoft.Authorization/roleAssignments?api-version=2022-04-01&" +
            urlencode({"$filter": "principalId eq " + principal}))
        need(type(guest_roles) is dict and type(guest_roles.get("value")) is list and
             not guest_roles.get("nextLink") and guest_roles["value"] == [],
             "pilot VM identity has ARM privileges or incomplete role evidence")
        effective_roles = self.read(ARM + subscription +
            "/providers/Microsoft.Authorization/roleAssignments?api-version=2022-04-01&" +
            urlencode({"$filter": "assignedTo('" + principal + "')"}))
        need(type(effective_roles) is dict and type(effective_roles.get("value")) is list and
             not effective_roles.get("nextLink") and effective_roles["value"] == [],
             "pilot VM identity inherits ARM privileges or incomplete role evidence")
        storage = props["storageProfile"]
        os_disk = storage["osDisk"]
        managed = os_disk["managedDisk"]
        need(os_disk["createOption"] == "FromImage" and os_disk["deleteOption"] == "Delete" and
             type(os_disk["diskSizeGB"]) is int and os_disk["diskSizeGB"] == 64 and
             managed["storageAccountType"] == "StandardSSD_LRS" and
             not os_disk.get("diffDiskSettings") and not storage.get("dataDisks") and
             managed["id"].casefold().startswith(pilot.casefold() + "/providers/microsoft.compute/disks/"),
             "VM disk configuration exceeds reviewed cost or cleanup scope")
        disk = self.resource(managed["id"], "2024-03-02")
        dp = disk["properties"]
        need(disk["sku"]["name"] == "StandardSSD_LRS" and
             type(dp["diskSizeGB"]) is int and dp["diskSizeGB"] == 64 and
             dp["creationData"]["createOption"] == "FromImage" and
             dp["provisioningState"] == "Succeeded" and dp["diskState"] == "Attached" and
             disk["managedBy"].casefold() == instance["resource_id"].casefold(),
             "live managed disk differs from reviewed configuration")
        need([x["id"] for x in props["networkProfile"]["networkInterfaces"]] == network["vm_nic_ids"],
             "VM network attachments changed")
        nic = self.resource(network["vm_nic_id"], "2024-05-01")["properties"]
        ips = nic["ipConfigurations"]
        need(nic.get("enableIPForwarding", False) is False and len(ips) == 1 and
             not nic.get("networkSecurityGroup") and not nic.get("privateEndpoint") and
             ips[0]["properties"].get("publicIPAddress") is None and
             ips[0]["properties"]["subnet"]["id"] == network["subnet_id"], "NIC networking changed")
        subnet = self.resource(network["subnet_id"], "2024-05-01")["properties"]
        need(subnet.get("defaultOutboundAccess") is False and not subnet.get("routeTable") and
             subnet["natGateway"]["id"] == network["nat_gateway_id"] and
             subnet["networkSecurityGroup"]["id"] == network["network_security_group_id"],
             "subnet networking changed")
        nat = self.resource(network["nat_gateway_id"], "2024-05-01")
        ip = self.resource(network["nat_gateway_public_ip_id"], "2024-05-01")
        need(nat["sku"]["name"] == ip["sku"]["name"] == "Standard" and
             nat["properties"]["publicIpAddresses"] == [{"id": network["nat_gateway_public_ip_id"]}] and
             not nat["properties"].get("publicIpPrefixes") and
             ip["properties"]["publicIPAllocationMethod"] == "Static" and
             ip["properties"].get("publicIPAddressVersion", "IPv4") == "IPv4", "NAT path changed")
        rules = self.resource(network["network_security_group_id"], "2024-05-01")["properties"]["securityRules"]
        normalized = []
        for rule in rules:
            rp = {k: v for k, v in rule["properties"].items()
                  if k not in ("provisioningState", "description") and v not in (None, [])}
            if isinstance(rp.get("protocol"), str) and rp["protocol"].lower() == "tcp":
                rp["protocol"] = "Tcp"
            normalized.append({"name": rule["name"], "properties": rp})
        need(sorted(normalized, key=lambda r: r["name"]) == sorted(expected_rules(), key=lambda r: r["name"]),
             "live inbound/outbound rules differ from approved template")
        workflow = self.resource(self.watchdog, "2019-05-01")
        wp, wi = workflow["properties"], workflow["identity"]
        need(wp["state"] == "Enabled" and wp["provisioningState"] == "Succeeded" and
             wi["type"] == "SystemAssigned" and wi["tenantId"] == self.tenant and
             wi["principalId"] != instance["system_assigned_identity_principal_id"] and
             wp["parameters"] == {"deadlineUtc": {"value": cleanup_trigger_utc}} and
             wp["definition"] == watchdog_definition(instance["resource_id"], pilot, self.lifecycle["watchdog_resource_group_id"]),
             "watchdog disabled, changed or bound to a different deadline")
        trigger = self.resource(self.watchdog + "/triggers/every_minute", "2019-05-01")["properties"]
        need(trigger.get("state") == "Enabled" and trigger.get("provisioningState") == "Succeeded",
             "watchdog recurrence trigger is not enabled")
        direct_role_ids = {}
        for scope in (pilot, self.lifecycle["watchdog_resource_group_id"]):
            assignments = self.read(ARM + scope + "/providers/Microsoft.Authorization/roleAssignments?api-version=2022-04-01")
            need(type(assignments) is dict and type(assignments.get("value")) is list and
                 not assignments.get("nextLink"), "cleanup role list incomplete")
            role_id = pilot.split("/resourceGroups/")[0] + "/providers/Microsoft.Authorization/roleDefinitions/" + CONTRIBUTOR
            matches = [a for a in assignments["value"] if
                 a.get("properties", {}).get("principalId") == wi["principalId"] and
                     a["properties"].get("scope", "").casefold() == scope.casefold() and
                     a["properties"].get("roleDefinitionId", "").casefold() == role_id.casefold() and
                     a["properties"].get("condition") in (None, "")]
            need(len(matches) == 1 and type(matches[0].get("id")) is str and
                 matches[0]["id"].casefold().startswith(
                     scope.casefold() + "/providers/microsoft.authorization/roleassignments/"),
                 "watchdog lacks a unique direct cleanup role")
            direct_role_ids[scope] = matches[0]["id"]
        optional_pilot = [instance["resource_id"] + "/extensions/NvidiaGpuDriverLinux",
                          network["subnet_id"], direct_role_ids[pilot]]
        optional_pilot.extend(network["network_security_group_id"] + "/securityRules/" +
                              rule["name"] for rule in expected_rules())
        self.exact_group_resources(pilot, (
            instance["resource_id"], managed["id"], network["vm_nic_id"],
            network["nat_gateway_id"], network["nat_gateway_public_ip_id"],
            network["network_security_group_id"],
            network["subnet_id"].rsplit("/subnets/", 1)[0]), optional_pilot)
        watchdog_group = self.lifecycle["watchdog_resource_group_id"]
        self.exact_group_resources(watchdog_group, (self.watchdog,), (
            self.watchdog + "/triggers/every_minute", direct_role_ids[watchdog_group]))
        self.unlocked_cleanup_scopes()
        after = self.clock()
        need(0 <= (after - before).total_seconds() <= 30 and
             after < authority.timestamp(cleanup_trigger_utc) < authority.timestamp(expiry),
             "live observation timed out or expired")
        return {"token_sha256": hashlib.sha256(token.encode("ascii")).hexdigest(), "audience": audience,
            "azure_instance": instance, "token_expires_at_utc": expiry,
            "network_evidence_sha256": digest(network), "watchdog_cleanup_trigger_utc": cleanup_trigger_utc,
            "observed_at_utc": after.strftime("%Y-%m-%dT%H:%M:%SZ"), "entra_signature_verified": True,
            "azure_control_plane_verified": True, "watchdog_healthy": True, "cleanup_scope_verified": True}
