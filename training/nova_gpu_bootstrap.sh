#!/usr/bin/env bash
# Managed Azure Run Command on the private, independently watched T4 VM.
# Experimental Nova adapter only; this does not enable selected pilots or routes.
set -euo pipefail

: "${SOURCE_COMMIT:?40-character published source commit required}"
: "${DEADLINE_UTC:?independent Azure watchdog deadline required}"
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || exit 2
end_epoch="$(date -u -d "$DEADLINE_UTC" +%s)"
now_epoch="$(date -u +%s)"
(( end_epoch > now_epoch + 25 * 60 && end_epoch <= now_epoch + 3 * 3600 )) || exit 2

work=/opt/kova-nova-experiment
mkdir -m 700 -p "$work"
log="$work/run.log"
on_exit() {
    status=$?
    echo "NOVA_EXPERIMENT_EXIT=$status TIME=$(date -u +%FT%TZ)"
    if [[ -f "$log" ]]; then tail -n 30 "$log"; fi
}
trap on_exit EXIT

{
    echo "NOVA_EXPERIMENT_START=$(date -u +%FT%TZ) DEADLINE=$DEADLINE_UTC SOURCE=$SOURCE_COMMIT"
    # The pinned NVIDIA extension can add an unrelated CUDA apt feed. Do not
    # refresh that feed here; the image already has Git and Python 3.12.
    command -v git >/dev/null
    export DEBIAN_FRONTEND=noninteractive
    if ! python3.12 -m venv "$work/venv"; then
        apt-get install -y -qq python3.12-venv
        python3.12 -m venv "$work/venv"
    fi
    git clone --quiet --depth 1 --branch feature/phase-a-cumulative-review-20260927 \
        https://github.com/Blockigaming/KovaGPT_Models.git "$work/source"
    [[ "$(git -C "$work/source" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || {
        echo 'Source head changed: refusing unreviewed code' >&2
        exit 3
    }
    "$work/venv/bin/python" -m pip install --disable-pip-version-check \
        --require-hashes -r "$work/source/requirements/kova-cosmo-sft-py312-linux.lock"
    "$work/venv/bin/python" -m pip install --disable-pip-version-check \
        --require-hashes -r "$work/source/requirements/kova-three-family-bitsandbytes-py312-linux.lock"
    cd "$work/source"
    "$work/venv/bin/python" -m training.pinned_snapshot_download \
        --family kova-nova --destination "$work/snapshot" --execute
    export HF_HUB_OFFLINE=1
    "$work/venv/bin/python" -m training.nova_gpu_candidate --execute \
        --snapshot "$work/snapshot" --output "$work/candidate" \
        --deadline-utc "$DEADLINE_UTC" --source-commit "$SOURCE_COMMIT"
    echo "NOVA_EXPERIMENT_FINISHED=$(date -u +%FT%TZ)"
} >>"$log" 2>&1
