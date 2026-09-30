#!/usr/bin/env bash
# Only on the one owner-authorized VM AFTER independent watchdog verification.
set -euo pipefail
: "${SOURCE_COMMIT:?exact published source required}"
: "${OWNER_GRANT_FILE:?external owner grant path required}"
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || exit 2
test -f "$OWNER_GRANT_FILE"
# Bound the ENTIRE bootstrap process group, including installation/downloads.
# This guest timer supplements the separate ARM deallocation/deletion watchdog.
if [[ "${KOVA_A35_WRAPPED:-0}" != 1 ]]; then
  remaining=$(python3 - "$OWNER_GRANT_FILE" <<'PY'
import json,sys,time
g=json.load(open(sys.argv[1])); s=g['allocation_started_epoch']; n=int(time.time())
assert g['owner_authorized'] is True and type(s) is int and s<=n<s+4200
print(s+4200-n)
PY
)
  exec timeout --signal=TERM --kill-after=30 "$remaining" \
    env KOVA_A35_WRAPPED=1 bash "$0"
fi
work=/opt/kova-a35-nova-screen
mkdir -m 700 "$work" # Existing directory rejects automatic retries.
cp "$OWNER_GRANT_FILE" "$work/grant.json"
chmod 600 "$work/grant.json"
read -r work_deadline remaining < <(python3 - "$work/grant.json" "$SOURCE_COMMIT" <<'PY'
import json,sys,time
g=json.load(open(sys.argv[1]))
assert g['owner_authorized'] is True and g['source_commit']==sys.argv[2]
assert g['family']=='kova-nova' and g['training_runs']==g['evaluation_sweeps']==1
assert g['spend_ceiling_usd']=='5.00' and g['live_watchdog_verified'] is True
s=g['allocation_started_epoch']; n=int(time.time())
assert s<=n<s+4200 and g['watchdog_deadline_epoch']==s+4500
assert g['allocation_deadline_epoch']==s+5400
print(s+4200,s+4200-n)
PY
)
test "$remaining" -gt 1200
# Count all non-loopback traffic since boot, including driver/bootstrap traffic.
# Independent ingress/egress quotas bound combined network processing to 32 GB.
# No proxy, model endpoint or other worker can continue after these quotas.
command -v iptables >/dev/null
read -r receive_remaining send_remaining < <(python3 - <<'PY'
from pathlib import Path
rx=tx=0
for line in Path('/proc/net/dev').read_text().splitlines()[2:]:
 name,values=line.split(':'); v=values.split()
 if name.strip()!='lo': rx+=int(v[0]); tx+=int(v[8])
assert 0<=rx<31000000000 and 0<=tx<1000000000
print(31000000000-rx,1000000000-tx)
PY
)
test "$receive_remaining" -gt 0 && test "$send_remaining" -gt 0
iptables -N KOVA_A35_INPUT
iptables -A KOVA_A35_INPUT -m quota --quota "$receive_remaining" -j RETURN
iptables -A KOVA_A35_INPUT -j DROP
iptables -I INPUT 1 '!' -i lo -j KOVA_A35_INPUT
iptables -N KOVA_A35_OUTPUT
iptables -A KOVA_A35_OUTPUT -m quota --quota "$send_remaining" -j RETURN
iptables -A KOVA_A35_OUTPUT -j DROP
iptables -I OUTPUT 1 '!' -o lo -j KOVA_A35_OUTPUT
export PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL= PIP_NO_CACHE_DIR=1
export HF_HUB_DISABLE_TELEMETRY=1 WANDB_DISABLED=true
if ! python3.12 -m venv "$work/venv"; then
  # Use the image's signed package index; do not refresh the unrelated CUDA feed.
  DEBIAN_FRONTEND=noninteractive timeout 180 apt-get install -y --no-install-recommends python3.12-venv
  python3.12 -m venv "$work/venv"
fi
git init -q "$work/source"
git -C "$work/source" remote add origin https://github.com/Blockigaming/KovaGPT_Models.git
timeout 90 git -C "$work/source" fetch --quiet --depth 1 origin "$SOURCE_COMMIT"
git -C "$work/source" checkout --quiet --detach FETCH_HEAD
test "$(git -C "$work/source" rev-parse HEAD)" = "$SOURCE_COMMIT"
cd "$work/source"
timeout 900 "$work/venv/bin/python" -m pip install --disable-pip-version-check \
  --retries 0 --require-hashes -r requirements/kova-cosmo-sft-py312-linux.lock \
  -r requirements/kova-three-family-bitsandbytes-py312-linux.lock
"$work/venv/bin/python" -m pip check
"$work/venv/bin/python" -m training.a35_nova_screen # Pins/scope before weights.
timeout 900 "$work/venv/bin/python" -m training.pinned_snapshot_download \
  --family kova-nova --destination "$work/snapshot" --execute
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
remaining=$((work_deadline - $(date +%s)))
test "$remaining" -gt 600
timeout --signal=TERM --kill-after=30 "$remaining" "$work/venv/bin/python" \
  -m training.a35_nova_screen --execute --owner-grant "$work/grant.json" \
  --snapshot "$work/snapshot" --output "$work/candidate"
