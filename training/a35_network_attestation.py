"""Narrow ARM attachment comparison for the A35 pre-GPU network check.

Azure's 2024-05-01 subnet response can uppercase the resource group, NIC and
IP-configuration names. ARM names are case-insensitive; the full identity and
single-reference shape must still match. This does not admit a network by
itself: the caller retains the independent NIC, subnet, NAT, NSG and DDoS checks.
https://learn.microsoft.com/azure/azure-resource-manager/management/resource-name-rules
"""
import re

NIC_ID = re.compile(
    r"/subscriptions/[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}"
    r"/resourceGroups/[a-z0-9_.()-]{1,90}"
    r"/providers/Microsoft\.Network/networkInterfaces/[a-z0-9_.-]{1,80}",
    re.IGNORECASE | re.ASCII,
)


def private_nic_attachment_matches(attachments, nic_id):
    """Accept one exact private-IP reference, differing only in ASCII case."""
    if not (type(nic_id) is str and nic_id.isascii() and NIC_ID.fullmatch(nic_id)
            and type(attachments) is list and len(attachments) == 1
            and type(attachments[0]) is dict and set(attachments[0]) == {"id"}):
        return False
    actual = attachments[0]["id"]
    return (type(actual) is str and actual.isascii()
            and actual.casefold() == (nic_id + "/ipConfigurations/private").casefold())
