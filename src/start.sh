#!/usr/bin/env bash
# Shared boot script for every HearmemanAI ComfyUI pod template.
# Invoked by each template's baked start_script.sh as:
#
#     exec bash /comfyui-runtime/src/start.sh /comfyui-<template>
#
# $1 is TEMPLATE_DIR. Everything else arrives via env (CONTRACTS.md section 9).
# Shape is comfyui-minimax/src/start.sh (the family's most current), minus every
# boot-time SageAttention source-build remnant (CONTRACTS.md section 12.2).
#
# Deliberately NO `set -e`: a failed download or install must surface as a
# notice, never as a dead pod (CONTRACTS.md section 7).

TEMPLATE_DIR="${1:-}"
RUNTIME_DIR="/comfyui-runtime"
TEMPLATE_JSON="$TEMPLATE_DIR/template.json"
export TEMPLATE_DIR RUNTIME_DIR

# ---------------------------------------------------------------------------
# Boot report state (EXECUTION.md item N1). Failures warn and continue
# (decision #19), so the deployment report printed at the END of boot is the
# only place a customer learns something degraded. Every phase below records
# what happened into this state file; boot_report.py renders it once, to the
# log and into the read-me note workflow. Hooks are sourced, so they can call
# report_warn / report_off too (e.g. ltx2's licence preflight, decision #18).
# ---------------------------------------------------------------------------
BOOT_STATE="/tmp/boot_report_state.tsv"
: > "$BOOT_STATE"
report_kv()   { printf 'set\t%s\t%s\n' "$1" "$2" >> "$BOOT_STATE"; }
report_warn() { printf 'warn\t%s\n' "$1" >> "$BOOT_STATE"; }
report_off()  { printf 'off\t%s\t%s\n' "$1" "$2" >> "$BOOT_STATE"; }
export BOOT_STATE
report_kv template_name "$(basename "${TEMPLATE_DIR:-unknown}")"
report_kv pod_id "${RUNPOD_POD_ID:-}"

if [ -z "$TEMPLATE_DIR" ] || [ ! -f "$TEMPLATE_JSON" ]; then
    echo "❌ template.json not found at '$TEMPLATE_JSON' (arg 1 must be the template repo dir)."
    echo "   Booting degraded: no models will be provisioned and no custom nodes cloned."
    report_warn "template.json not found; no models provisioned, no custom nodes cloned"
fi

# Read one value out of template.json. Dotted path; booleans print true/false,
# lists print one item per line, anything unreadable prints nothing.
template_json_get() {
    python3 - "$TEMPLATE_JSON" "$1" <<'PY'
import json
import sys

try:
    node = json.load(open(sys.argv[1]))
    for key in sys.argv[2].split("."):
        node = node[key]
except Exception:
    sys.exit(0)
if isinstance(node, bool):
    print("true" if node else "false")
elif isinstance(node, list):
    for item in node:
        print(item)
else:
    print(node)
PY
}

# Print both pins so every support log names the runtime SHA and base tag
# (CONTRACTS.md section 6, plan D2).
if [ -f "$TEMPLATE_DIR/pins.json" ]; then
    # Prints the pins line AND records both values for the boot report.
    python3 - "$TEMPLATE_DIR/pins.json" <<'PY'
import json
import os
import sys

try:
    pins = json.load(open(sys.argv[1]))
    ref = pins.get("runtime_ref", "?")
    base = pins.get("base_image", "?")
    print("📌 pins.json: runtime_ref=%s base_image=%s" % (ref, base))
    with open(os.environ["BOOT_STATE"], "a") as f:
        f.write("set\truntime_ref\t%s\nset\tbase_image\t%s\n" % (ref, base))
except Exception as exc:
    print("⚠️  Could not read pins.json: %r" % (exc,))
PY
else
    echo "⚠️  pins.json not found at $TEMPLATE_DIR/pins.json"
fi
RUNTIME_SHA="$(git -C "$RUNTIME_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "📌 comfyui-runtime checkout: $RUNTIME_SHA"
report_kv runtime_sha "$RUNTIME_SHA"

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

# Special installs or overrides that need to occur before starting ComfyUI
if [ -f "/workspace/additional_params.sh" ]; then
    chmod +x /workspace/additional_params.sh
    echo "Executing additional_params.sh..."
    /workspace/additional_params.sh
else
    echo "additional_params.sh not found in /workspace. Skipping..."
fi

# Set the network volume path
NETWORK_VOLUME="/workspace"
URL="http://127.0.0.1:8188"
if [ ! -d "$NETWORK_VOLUME" ]; then
    echo "NETWORK_VOLUME directory '$NETWORK_VOLUME' does not exist. You are NOT using a network volume. Setting NETWORK_VOLUME to '/' (root directory)."
    NETWORK_VOLUME="/"
fi
export NETWORK_VOLUME

# Keep a durable copy of the whole boot log on the volume so support never
# depends on RunPod's console scrollback (CLAUDE.md section 6).
exec > >(tee -a "$NETWORK_VOLUME/comfyui.log") 2>&1

# ---------------------------------------------------------------------------
# DNS preflight, ABOVE everything that touches the network. A pod with
# RunPod's "Global Networking" setting enabled has no public DNS and every
# clone/download dies with "Could not resolve host" (the single largest
# support cluster). Warn loudly and keep booting: the failure mode is missing
# models, never a dead pod.
# ---------------------------------------------------------------------------
DNS_OK=""
for dns_attempt in 1 2 3; do
    if getent hosts github.com >/dev/null 2>&1; then
        DNS_OK=1
        break
    fi
    echo "🌐 DNS preflight attempt $dns_attempt failed, retrying..."
    sleep 3
done
if [ -n "$DNS_OK" ]; then
    echo "🌐 DNS preflight passed."
else
    echo "❌ DNS resolution is BROKEN on this pod (cannot resolve github.com)."
    echo "   Most common cause: RunPod's \"Global Networking\" setting is ENABLED on this pod."
    echo "   Global Networking pods have no public DNS. Deploy a new pod with Global Networking"
    echo "   DISABLED. This is a RunPod pod setting, not a GitHub or HuggingFace outage."
    echo "   Model downloads, custom node installs and version resolution will all fail until DNS works."
    report_warn "DNS is broken on this pod (RunPod Global Networking enabled?); downloads likely failed"
fi

if [ "$NETWORK_VOLUME" = "/" ]; then
    echo "NETWORK_VOLUME directory doesn't exist. Starting JupyterLab on root directory..."
    jupyter-lab --ip=0.0.0.0 --allow-root --no-browser --NotebookApp.token='' --NotebookApp.password='' --ServerApp.allow_origin='*' --ServerApp.allow_credentials=True --notebook-dir=/ &
else
    echo "NETWORK_VOLUME directory exists. Starting JupyterLab..."
    jupyter-lab --ip=0.0.0.0 --allow-root --no-browser --NotebookApp.token='' --NotebookApp.password='' --ServerApp.allow_origin='*' --ServerApp.allow_credentials=True --notebook-dir=/workspace &
fi

# ComfyUI source stays in the image (ephemeral, fast local disk). Models,
# workflows, outputs, inputs and user-added custom_nodes live on the network
# volume via extra_model_paths.yaml + symlinks. This avoids the multi-minute
# mv of /ComfyUI to MooseFS on first boot.
COMFYUI_DIR="/ComfyUI"
PERSIST_ROOT="$NETWORK_VOLUME/ComfyUI"
WORKFLOW_DIR="$PERSIST_ROOT/user/default/workflows"
export COMFYUI_DIR PERSIST_ROOT WORKFLOW_DIR

MODELS_SYMLINK="$(template_json_get models_symlink)"

mkdir -p "$PERSIST_ROOT/models" "$PERSIST_ROOT/user" \
         "$PERSIST_ROOT/output" "$PERSIST_ROOT/input" \
         "$PERSIST_ROOT/custom_nodes"

# Symlink user/output/input (plus models when template.json says so, qwen's
# shape) into /ComfyUI so ComfyUI uses its default code paths (passing
# --user-directory triggers a None-user_dir bug in user_manager.get_users).
# A function because COMFYUI_VERSION below may `git reset --hard` /ComfyUI,
# which restores tracked dirs (input/) over the symlinks and must re-link.
link_volume_dirs() {
    [ "$NETWORK_VOLUME" = "/" ] && return 0
    # First boot only: migrate baked user/ content (default templates, schema)
    # to the volume before swapping in the symlink. cp -an is no-clobber, so
    # re-runs on existing volumes are safe.
    if [ -d "$COMFYUI_DIR/user" ] && [ ! -L "$COMFYUI_DIR/user" ]; then
        cp -an "$COMFYUI_DIR/user/." "$PERSIST_ROOT/user/" 2>/dev/null || true
        rm -rf "$COMFYUI_DIR/user"
    fi
    local link_subdirs=(user output input)
    [ "$MODELS_SYMLINK" = "true" ] && link_subdirs+=(models)
    local sub
    for sub in "${link_subdirs[@]}"; do
        [ -L "$COMFYUI_DIR/$sub" ] || rm -rf "${COMFYUI_DIR:?}/$sub" 2>/dev/null || true
        ln -sfn "$PERSIST_ROOT/$sub" "$COMFYUI_DIR/$sub"
    done
}
link_volume_dirs

# Generate extra_model_paths.yaml from the live $PERSIST_ROOT so paths always
# match the actual network volume. The yaml is generated from the SAME list
# that is mkdir'd: every declared path must exist before launch or ComfyUI
# dies in prestartup (CLAUDE.md section 9). Base list is wan's, frozen
# (CONTRACTS.md section 5); template.json extra_model_paths entries are added.
MODEL_PATH_CATEGORIES=(checkpoints clip clip_vision controlnet diffusion_models
    embeddings loras style_models text_encoders unet upscale_models
    latent_upscale_models detection vae)
while IFS= read -r extra_cat; do
    [ -z "$extra_cat" ] && continue
    already=""
    for cat in "${MODEL_PATH_CATEGORIES[@]}"; do
        [ "$cat" = "$extra_cat" ] && already=1 && break
    done
    [ -z "$already" ] && MODEL_PATH_CATEGORIES+=("$extra_cat")
done < <(template_json_get extra_model_paths)

for cat in "${MODEL_PATH_CATEGORIES[@]}"; do
    mkdir -p "$PERSIST_ROOT/models/$cat"
done

EXTRA_PATHS_FLAG=""
if [ "$NETWORK_VOLUME" != "/" ]; then
    {
        echo "network_volume:"
        echo "    base_path: $PERSIST_ROOT"
        for cat in "${MODEL_PATH_CATEGORIES[@]}"; do
            echo "    $cat: models/$cat"
        done
        echo "    custom_nodes: custom_nodes"
    } > "$COMFYUI_DIR/extra_model_paths.yaml"
    EXTRA_PATHS_FLAG="--extra-model-paths-config $COMFYUI_DIR/extra_model_paths.yaml"
else
    rm -f "$COMFYUI_DIR/extra_model_paths.yaml"
fi

# ---------------------------------------------------------------------------
# COMFYUI_VERSION (plan section 5b). approved (default) ACTIVELY RESTORES to
# the SHA baked at /comfyui-approved-ref: the writable layer persists across a
# restart, so a pod that once ran `latest` can otherwise never come back.
# latest resolves the newest upstream RELEASE tag. An explicit ref is used
# as-is. On any resolve failure: NO MOVEMENT, print the current SHA, one line.
# ---------------------------------------------------------------------------
COMFYUI_VERSION="${COMFYUI_VERSION:-approved}"
CURRENT_COMFY_SHA="$(git -C "$COMFYUI_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
TARGET_COMFY_SHA=""
case "$COMFYUI_VERSION" in
    approved)
        if [ -r /comfyui-approved-ref ]; then
            TARGET_COMFY_SHA="$(tr -d '[:space:]' < /comfyui-approved-ref)"
        fi
        ;;
    latest)
        LATEST_TAG="$(curl -s --max-time 30 https://api.github.com/repos/comfyanonymous/ComfyUI/releases/latest \
            | python3 -c 'import json,sys; print(json.load(sys.stdin).get("tag_name",""))' 2>/dev/null)"
        if [ -n "$LATEST_TAG" ] && git -C "$COMFYUI_DIR" fetch --quiet origin "refs/tags/$LATEST_TAG" 2>/dev/null; then
            TARGET_COMFY_SHA="$(git -C "$COMFYUI_DIR" rev-parse 'FETCH_HEAD^{commit}' 2>/dev/null)"
        fi
        ;;
    *)
        # Explicit ref: a SHA, tag or branch. Resolve locally first (the base
        # clone carries full refs), fall back to a fetch.
        if git -C "$COMFYUI_DIR" rev-parse --verify --quiet "$COMFYUI_VERSION^{commit}" >/dev/null 2>&1; then
            TARGET_COMFY_SHA="$(git -C "$COMFYUI_DIR" rev-parse "$COMFYUI_VERSION^{commit}")"
        elif git -C "$COMFYUI_DIR" fetch --quiet origin "$COMFYUI_VERSION" 2>/dev/null; then
            TARGET_COMFY_SHA="$(git -C "$COMFYUI_DIR" rev-parse 'FETCH_HEAD^{commit}' 2>/dev/null)"
        fi
        ;;
esac

if [ -z "$TARGET_COMFY_SHA" ]; then
    echo "⚠️  COMFYUI_VERSION=$COMFYUI_VERSION could not be resolved. ComfyUI stays at $CURRENT_COMFY_SHA."
    report_warn "COMFYUI_VERSION=$COMFYUI_VERSION could not be resolved; ComfyUI stayed where it was"
elif [ "$TARGET_COMFY_SHA" != "$CURRENT_COMFY_SHA" ]; then
    # Make sure the target's objects are local (approved's always are, from
    # the base clone), then move. Requirements reinstall runs under the
    # base-owned PIP_CONSTRAINT, so torch cannot move (plan section 5b).
    git -C "$COMFYUI_DIR" cat-file -e "$TARGET_COMFY_SHA^{commit}" 2>/dev/null \
        || git -C "$COMFYUI_DIR" fetch --quiet origin "$TARGET_COMFY_SHA" 2>/dev/null || true
    if git -C "$COMFYUI_DIR" reset --hard "$TARGET_COMFY_SHA" >/dev/null 2>&1; then
        echo "🔁 ComfyUI moved $CURRENT_COMFY_SHA -> $TARGET_COMFY_SHA (COMFYUI_VERSION=$COMFYUI_VERSION). Reinstalling requirements..."
        pip install -r "$COMFYUI_DIR/requirements.txt" > /tmp/pip_comfyui_version.log 2>&1 \
            || echo "⚠️  ComfyUI requirements reinstall failed (see /tmp/pip_comfyui_version.log)."
        # reset --hard restores tracked dirs (input/) over the volume symlinks.
        link_volume_dirs
    else
        echo "⚠️  Could not move ComfyUI to $TARGET_COMFY_SHA. ComfyUI stays at $CURRENT_COMFY_SHA."
        report_warn "Could not move ComfyUI to the requested version; it stayed at ${CURRENT_COMFY_SHA:0:7}"
    fi
else
    echo "✅ ComfyUI already at $CURRENT_COMFY_SHA (COMFYUI_VERSION=$COMFYUI_VERSION)."
fi
if [ "$COMFYUI_VERSION" != "approved" ]; then
    echo "⚠️  COMFYUI_VERSION=$COMFYUI_VERSION: the bundled workflows were validated against the approved"
    echo "    ComfyUI ref only. Upstream changes can break subgraph-based workflows with no warning."
    echo "    Set COMFYUI_VERSION=approved and restart the pod to return to the validated version."
    report_warn "COMFYUI_VERSION=$COMFYUI_VERSION: bundled workflows were only validated on the approved ref"
fi

# Record what ComfyUI we actually ended up on, for the deployment report.
report_kv comfy_sha "$(git -C "$COMFYUI_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
report_kv comfy_version "$(sed -n 's/^__version__ = "\(.*\)"/\1/p' "$COMFYUI_DIR/comfyui_version.py" 2>/dev/null)"
report_kv comfy_mode "$COMFYUI_VERSION"

# ---------------------------------------------------------------------------
# SageAttention: three synchronous steps (CONTRACTS.md sections 8/9, plan D9).
# No source build, no background subshell, no marker files, no wheel cache.
# 1. read torch.version.cuda major  2. install the matching baked wheel
# 3. run the kernel probe. The probe prints the verdict line; nothing extra.
# ---------------------------------------------------------------------------
TORCH_CUDA_MAJOR="$(python3 -c 'import torch; print((torch.version.cuda or "").split(".")[0])' 2>/dev/null)"
report_kv gpu_name "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)"
SAGE_FLAG=""
if [ "$(template_json_get sage)" = "true" ]; then
    case "$TORCH_CUDA_MAJOR" in
        12) SAGE_WHEEL_DIR="/opt/sage/cu128" ;;
        13) SAGE_WHEEL_DIR="/opt/sage/cu130" ;;
        *)  SAGE_WHEEL_DIR=""
            echo "⚠️  Unrecognized torch CUDA major '${TORCH_CUDA_MAJOR:-unknown}'. No SageAttention wheel installed."
            report_warn "Unrecognized torch CUDA major '${TORCH_CUDA_MAJOR:-unknown}'; no SageAttention wheel installed"
            ;;
    esac
    if [ -n "$SAGE_WHEEL_DIR" ]; then
        SAGE_WHEEL=""
        for whl in "$SAGE_WHEEL_DIR"/sageattention-*.whl; do
            [ -e "$whl" ] && SAGE_WHEEL="$whl" && break
        done
        if [ -n "$SAGE_WHEEL" ]; then
            echo "⚡ Installing baked SageAttention wheel: $SAGE_WHEEL"
            pip install --no-deps --force-reinstall "$SAGE_WHEEL" > /tmp/sage_wheel.log 2>&1 \
                || { echo "⚠️  SageAttention wheel install failed (see /tmp/sage_wheel.log)."
                     report_warn "SageAttention wheel install failed (see /tmp/sage_wheel.log)"; }
        else
            echo "⚠️  No SageAttention wheel found under $SAGE_WHEEL_DIR."
            report_warn "No SageAttention wheel found under $SAGE_WHEEL_DIR"
        fi
    fi
    # The probe owns the messaging for pass / unsupported arch / failure
    # (its arch check runs before the sageattention import, so it gives the
    # right verdict even when the wheel is absent). Its one line is captured
    # for the deployment report and relayed to the log unchanged.
    SAGE_PROBE_MSG="$(python3 "$RUNTIME_DIR/src/sage_probe.py")"
    sage_probe_rc=$?
    [ -n "$SAGE_PROBE_MSG" ] && echo "$SAGE_PROBE_MSG"
    report_kv sage_msg "$SAGE_PROBE_MSG"
    case "$sage_probe_rc" in
        0) SAGE_FLAG="--use-sage-attention"
           report_kv sage enabled ;;
        2) report_kv sage unsupported ;;
        *) report_kv sage probe_failed ;;
    esac
else
    echo "⏭️  SageAttention disabled for this template (template.json sage != true)."
    report_kv sage off_template
fi

# CivitAI downloader: baked at /usr/local/bin by the base image; the clone is
# ONLY an if-missing fallback for images built before the base exists
# (CONTRACTS.md section 9 step 7). Failure is not fatal: only the CivitAI ID
# downloads depend on it.
if [ ! -x /usr/local/bin/download_with_aria.py ]; then
    echo "CivitAI downloader missing from the image. Fetching to /usr/local/bin..."
    rm -rf /tmp/CivitAI_Downloader
    if git clone "https://github.com/Hearmeman24/CivitAI_Downloader.git" /tmp/CivitAI_Downloader; then
        mv /tmp/CivitAI_Downloader/download_with_aria.py /usr/local/bin/ \
            && chmod +x /usr/local/bin/download_with_aria.py
    else
        echo "⚠️  CivitAI downloader clone failed. CHECKPOINT/LORAS ID downloads will not work this boot."
    fi
    rm -rf /tmp/CivitAI_Downloader
fi

# ---------------------------------------------------------------------------
# Custom-node clone loop from template.json custom_nodes.repos.
# Entry syntax (CONTRACTS.md section 5): "<url>", "<url>|<sha>", "<url>|force".
# Requirements installs are backgrounded; PIDs collected and waited before
# launch (wan start.sh:199-217,413-429). PIP_CONSTRAINT (base-owned) applies.
# ---------------------------------------------------------------------------
if [ "$(template_json_get custom_nodes.target)" = "volume" ]; then
    CUSTOM_NODES_DIR="$PERSIST_ROOT/custom_nodes"
else
    CUSTOM_NODES_DIR="$COMFYUI_DIR/custom_nodes"
fi
export CUSTOM_NODES_DIR
mkdir -p "$CUSTOM_NODES_DIR"

declare -A PIP_INSTALL_PIDS=()
CUSTOM_NODE_REPOS=()
while IFS= read -r repo_entry; do
    [ -n "$repo_entry" ] && CUSTOM_NODE_REPOS+=("$repo_entry")
done < <(template_json_get custom_nodes.repos)

for entry in "${CUSTOM_NODE_REPOS[@]}"; do
    url="${entry%%|*}"
    pin=""
    [[ "$entry" == *"|"* ]] && pin="${entry#*|}"
    name="$(basename "$url" .git)"
    dir="$CUSTOM_NODES_DIR/$name"

    if [ "$pin" = "force" ]; then
        pin=""
        if [ -d "$dir" ]; then
            echo "🗑️  Removing existing $name (force-refresh)..."
            rm -rf "$dir"
        fi
    fi

    if [ ! -d "$dir/.git" ]; then
        echo "📥 Cloning $name..."
        git clone "$url" "$dir" || { echo "❌ Failed to clone $name. Its nodes will be missing."
                                     report_warn "Failed to clone custom node $name; its nodes are missing"
                                     continue; }
    elif [ -n "$pin" ]; then
        git -C "$dir" fetch --quiet origin 2>/dev/null || true
    else
        echo "🔄 Updating $name..."
        git -C "$dir" pull --ff-only 2>/dev/null || echo "⚠️  $name git pull failed. Keeping the existing checkout."
    fi

    if [ -n "$pin" ]; then
        git -C "$dir" reset --hard "$pin" \
            || echo "⚠️  Could not pin $name to $pin. Keeping the current checkout."
    fi

    if [ -f "$dir/requirements.txt" ]; then
        echo "🔧 Installing $name requirements (background)..."
        pip install -r "$dir/requirements.txt" > "/tmp/pip_${name}.log" 2>&1 &
        PIP_INSTALL_PIDS[$name]=$!
    fi
done

# ---------------------------------------------------------------------------
# pre_download hook (CONTRACTS.md section 7): sourced, not exec'd, so it may
# export env vars (flag flips, preflights) that the provisioner then reads.
# A hook handles its own errors; its failure never gates the boot.
# ---------------------------------------------------------------------------
HF_QUEUE_FILE="/tmp/hf_download_queue.tsv"
export HF_QUEUE_FILE
if [ -f "$TEMPLATE_DIR/src/hooks/pre_download.sh" ]; then
    echo "🪝 Sourcing pre_download hook..."
    # shellcheck disable=SC1091
    source "$TEMPLATE_DIR/src/hooks/pre_download.sh" \
        || { echo "⚠️  pre_download hook returned nonzero (continuing)."
             report_warn "pre_download hook returned nonzero"; }
fi

# Provisioner: flag state, quant/precision and variant choice are read from
# the process environment, mapped through template.json (CONTRACTS.md
# section 3). Exit 2 or 1 prints one loud line; boot continues.
# PROVISION_STATUS_FILE feeds the deployment report (workflow sets enabled,
# models already on disk); without it the report renders "unknown" rows.
echo "🧩 Provisioning workflows + download manifest..."
mkdir -p "$WORKFLOW_DIR"
PROVISION_STATUS_FILE="/tmp/provision_status.json"
export PROVISION_STATUS_FILE
rm -f "$PROVISION_STATUS_FILE"
python3 "$RUNTIME_DIR/src/provisioner.py" \
    --template "$TEMPLATE_DIR/template.json" \
    --registry "$TEMPLATE_DIR/src/models_registry.json" \
    --workflows-src "$TEMPLATE_DIR/workflows" \
    --workflows-dst "$WORKFLOW_DIR" \
    --models-root "$PERSIST_ROOT/models" \
    --manifest "$HF_QUEUE_FILE"
provisioner_rc=$?
if [ "$provisioner_rc" -ne 0 ]; then
    echo "❌ Provisioner exited $provisioner_rc. Model provisioning is incomplete; booting anyway (missing models surface as red nodes)."
    report_warn "Provisioner exited $provisioner_rc; model provisioning is incomplete"
fi

# Downloader, EXIT CODE CHECKED (wan does not check it today, start.sh:257;
# that bug dies here, CONTRACTS.md section 12.6). Nonzero: one line, boot
# continues. The manager's own final snapshot names each failed entry, and
# HF_STATUS_FILE hands the deployment report each failure's reason.
echo "🔽 Starting HF download manager..."
HF_STATUS_FILE="/tmp/hf_download_status.json"
export HF_STATUS_FILE
rm -f "$HF_STATUS_FILE"
python3 "$RUNTIME_DIR/src/hf_download_manager.py" "$HF_QUEUE_FILE"
downloader_rc=$?
if [ "$downloader_rc" -ne 0 ]; then
    echo "❌ Download manager exited $downloader_rc: one or more model downloads FAILED (the failed entries are named in the snapshot above). Booting anyway."
fi

# CivitAI ID downloads (minimax start.sh:227-263, with LORAS_DIR actually
# defined, which neither donor does).
CHECKPOINTS_DIR="$PERSIST_ROOT/models/checkpoints"
LORAS_DIR="$PERSIST_ROOT/models/loras"
declare -A MODEL_CATEGORIES=(
    ["$CHECKPOINTS_DIR"]="${CHECKPOINT_IDS_TO_DOWNLOAD:-replace_with_ids}"
    ["$LORAS_DIR"]="${LORAS_IDS_TO_DOWNLOAD:-replace_with_ids}"
)

download_count=0
for TARGET_DIR in "${!MODEL_CATEGORIES[@]}"; do
    mkdir -p "$TARGET_DIR"
    MODEL_IDS_STRING="${MODEL_CATEGORIES[$TARGET_DIR]}"

    # Skip if unset or still the RunPod template placeholder.
    if [[ -z "$MODEL_IDS_STRING" || "$MODEL_IDS_STRING" == "replace_with_ids" ]]; then
        echo "⏭️  Skipping CivitAI downloads for $TARGET_DIR (no IDs set)"
        continue
    fi

    IFS=',' read -ra MODEL_IDS <<< "$MODEL_IDS_STRING"
    for MODEL_ID in "${MODEL_IDS[@]}"; do
        MODEL_ID="${MODEL_ID// /}"
        [ -z "$MODEL_ID" ] && continue
        sleep 1
        echo "🚀 Scheduling CivitAI download: $MODEL_ID to $TARGET_DIR"
        (cd "$TARGET_DIR" && download_with_aria.py -m "$MODEL_ID") &
        ((download_count++))
    done
done

if [ "$download_count" -gt 0 ]; then
    echo "📋 Scheduled $download_count CivitAI downloads in background"
    echo "⏳ Waiting for CivitAI downloads to complete..."
    while pgrep -x "aria2c" > /dev/null; do
        echo "🔽 CivitAI downloads still in progress..."
        sleep 5
    done
    echo "✅ CivitAI downloads finished"
fi

# CivitAI serves some LoRAs as zips (minimax start.sh:303-307).
for zip_file in "$LORAS_DIR"/*.zip; do
    [ -e "$zip_file" ] || continue
    echo "Renaming $(basename "$zip_file") to .safetensors"
    mv "$zip_file" "${zip_file%.zip}.safetensors"
done

# Workspace as main working directory for the Jupyter terminal.
grep -qxF "cd $NETWORK_VOLUME" ~/.bashrc 2>/dev/null || echo "cd $NETWORK_VOLUME" >> ~/.bashrc

# Wait for the backgrounded node-requirements installs before launch.
for name in "${!PIP_INSTALL_PIDS[@]}"; do
    if wait "${PIP_INSTALL_PIDS[$name]}"; then
        echo "✅ $name requirements installed"
    else
        echo "❌ $name requirements install failed (see /tmp/pip_${name}.log). Its nodes may not load."
        report_warn "$name requirements install failed; its nodes may not load"
    fi
done

# Defensive: a custom node's requirements may have just clobbered
# onnxruntime-gpu with the CPU build (wan start.sh:431-440). Re-assert GPU.
# cu128 needs the CUDA-12 build from the Azure index; PyPI links CUDA 13
# (CLAUDE.md section 8).
if ! /opt/venv/bin/python -c \
    'import onnxruntime as o, sys; sys.exit(0 if "CUDAExecutionProvider" in o.get_available_providers() else 1)' \
    2>/dev/null; then
    echo "⚙️  onnxruntime CUDA provider missing. Reinstalling onnxruntime-gpu..."
    report_warn "A node install clobbered onnxruntime-gpu; it was reinstalled at boot"
    pip uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true
    if [ "$TORCH_CUDA_MAJOR" = "13" ]; then
        pip install onnxruntime-gpu
    else
        pip install onnxruntime-gpu \
            --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
    fi
fi

# pre_launch hook (CONTRACTS.md section 7): last-mile pip pins and file
# fixups, immediately before the launch.
if [ -f "$TEMPLATE_DIR/src/hooks/pre_launch.sh" ]; then
    echo "🪝 Sourcing pre_launch hook..."
    # shellcheck disable=SC1091
    source "$TEMPLATE_DIR/src/hooks/pre_launch.sh" \
        || { echo "⚠️  pre_launch hook returned nonzero (continuing)."
             report_warn "pre_launch hook returned nonzero"; }
fi

# Launch ComfyUI ONCE, nohup'ed, never restarted to add a flag. Never pipe
# the launch to tee while capturing $! (that names tee, not python;
# CLAUDE.md section 6). No PID is captured here at all.
echo "▶️  Starting ComfyUI"
nohup python3 "$COMFYUI_DIR/main.py" --listen --enable-cors-header '*' \
    $SAGE_FLAG $EXTRA_PATHS_FLAG \
    > "$NETWORK_VOLUME/comfyui_${RUNPOD_POD_ID}_nohup.log" 2>&1 &

# Liveness check, then the numbered self-serve triage block.
counter=0
max_wait=70
until curl --silent --fail "$URL" --output /dev/null; do
    if [ $counter -ge $max_wait ]; then
        echo "⚠️  ComfyUI should be up by now. If it's not running, there's probably an error."
        echo ""
        echo "🛠️  Troubleshooting Tips:"
        if [ "$TORCH_CUDA_MAJOR" = "13" ]; then
            echo "1. This is a CUDA 13 image. Make sure your CUDA Version is set to 13.0+ in the additional filters tab before deploying."
        else
            echo "1. Make sure that your CUDA Version is set to 12.8/12.9 by selecting that in the additional filters tab before deploying the template."
        fi
        echo "2. If you are deploying using network storage, try deploying without it."
        echo "3. If the log above shows \"Could not resolve host\", RunPod's Global Networking setting is enabled on this pod. Deploy a new pod with Global Networking DISABLED."
        echo "4. If all else fails, open the web terminal by clicking \"connect\", \"enable web terminal\" and running:"
        echo "   cat comfyui_${RUNPOD_POD_ID}_nohup.log"
        echo "   This should show a ComfyUI error. Please paste the error in HearmemanAI Discord Server for assistance."
        if [ "$COMFYUI_VERSION" != "approved" ]; then
            echo "Note: this pod is running COMFYUI_VERSION=$COMFYUI_VERSION. Set COMFYUI_VERSION=approved and restart to return to the validated ComfyUI."
        fi
        echo ""
        echo "📋 Startup logs location: $NETWORK_VOLUME/comfyui_${RUNPOD_POD_ID}_nohup.log"
        break
    fi

    echo "🔄  ComfyUI Starting Up... You can view the startup logs here: $NETWORK_VOLUME/comfyui_${RUNPOD_POD_ID}_nohup.log"
    sleep 2
    counter=$((counter + 2))
done

# ---------------------------------------------------------------------------
# Deployment report (EXECUTION.md item N1), replacing the old "ComfyUI is
# Ready" line. The liveness verdict above decides the header: the report
# never claims ready when 8188 is dead. The body is generated ONCE by
# boot_report.py and rendered twice from the same string: here (the pod log)
# and into the read-me MarkdownNote workflow (item N2), which is re-rendered
# on EVERY boot regardless of flags so its report always describes THIS boot.
# The note bypasses the provisioner's flag-gated copy on purpose.
# ---------------------------------------------------------------------------
if curl --silent --fail "$URL" --output /dev/null; then
    report_kv ready true
else
    report_kv ready false
    report_warn "ComfyUI did not answer on port 8188 within ${max_wait}s"
fi
python3 "$RUNTIME_DIR/src/boot_report.py" \
    --state "$BOOT_STATE" \
    --template "$TEMPLATE_JSON" \
    --manifest "$HF_QUEUE_FILE" \
    --provision-status "$PROVISION_STATUS_FILE" \
    --hf-status "$HF_STATUS_FILE" \
    --note-skeleton "$RUNTIME_DIR/src/readme_note.md" \
    --note-sections "$TEMPLATE_DIR/src/note_sections.md" \
    --note-out "$WORKFLOW_DIR/!! Read This First/Read This First.json" \
    || echo "⚠️  Deployment report renderer failed; the full boot log is at $NETWORK_VOLUME/comfyui.log"

# Never let the container exit when ComfyUI dies.
sleep infinity
