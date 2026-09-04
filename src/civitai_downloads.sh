#!/usr/bin/env bash
# Runtime-owned CivitAI downloader preparation and scheduling.
# Sourced by start.sh before any background dependency install can start.

CIVITAI_PYTHON="${CIVITAI_PYTHON:-python3}"
CIVITAI_VENDOR_DIR="$RUNTIME_DIR/vendor/civitai_downloader"
CIVITAI_DOWNLOADER="$CIVITAI_VENDOR_DIR/download_with_aria.py"
CIVITAI_REQUIREMENTS="$CIVITAI_VENDOR_DIR/requirements.txt"
CIVITAI_CHECKSUMS="$CIVITAI_VENDOR_DIR/SHA256SUMS"
CIVITAI_PREPARE_STATE=""

_civitai_warn() {
    echo "⚠️  $1"
    report_warn "$1"
}

_civitai_source_is_valid() {
    "$CIVITAI_PYTHON" -c '
import hashlib
import pathlib
import sys

source = pathlib.Path(sys.argv[1])
checksums = pathlib.Path(sys.argv[2])
requirements = pathlib.Path(sys.argv[3])
files = {source.name: source, requirements.name: requirements}
if not checksums.is_file() or any(not path.is_file() for path in files.values()):
    raise SystemExit(1)
entries = {}
for line in checksums.read_text(encoding="utf-8").splitlines():
    parts = line.split()
    if len(parts) != 2 or parts[1] in entries:
        raise SystemExit(1)
    entries[parts[1]] = parts[0]
if set(entries) != set(files):
    raise SystemExit(1)
for name, path in files.items():
    if hashlib.sha256(path.read_bytes()).hexdigest() != entries[name]:
        raise SystemExit(1)
payload = source.read_bytes()
compile(payload, str(source), "exec")
requirement = requirements.read_text(encoding="utf-8").strip()
if not __import__("re").fullmatch(r"[A-Za-z0-9_.-]+==[A-Za-z0-9_.+-]+", requirement):
    raise SystemExit(1)
' "$CIVITAI_DOWNLOADER" "$CIVITAI_CHECKSUMS" "$CIVITAI_REQUIREMENTS" \
        > /tmp/civitai_downloader_validation.log 2>&1
}

_civitai_dependencies_ready() {
    "$CIVITAI_PYTHON" -c '
import importlib.metadata
import pathlib
import re
import sys

import requests
from urllib3.util.retry import Retry  # noqa: F401

requirement = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").strip()
match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+-]+)", requirement)
if match is None:
    raise SystemExit(1)
try:
    installed = importlib.metadata.version(match.group(1))
except importlib.metadata.PackageNotFoundError:
    raise SystemExit(1)
raise SystemExit(0 if installed == match.group(2) == requests.__version__ else 1)
' "$CIVITAI_REQUIREMENTS"
}

prepare_civitai_downloader() {
    if ! _civitai_source_is_valid; then
        _civitai_warn "Runtime CivitAI downloader is missing or invalid at $CIVITAI_DOWNLOADER; requested CivitAI downloads are skipped"
        return 1
    fi

    if _civitai_dependencies_ready; then
        return 0
    fi

    echo "🔧 Installing runtime CivitAI downloader dependencies..."
    if ! "$CIVITAI_PYTHON" -m pip install --disable-pip-version-check \
        --requirement "$CIVITAI_REQUIREMENTS" \
        > /tmp/civitai_downloader_dependencies.log 2>&1; then
        _civitai_warn "CivitAI downloader dependency install failed (see /tmp/civitai_downloader_dependencies.log); requested CivitAI downloads are skipped"
        return 1
    fi
    if ! _civitai_dependencies_ready; then
        _civitai_warn "CivitAI downloader dependencies do not match $CIVITAI_REQUIREMENTS after install; requested CivitAI downloads are skipped"
        return 1
    fi
}

_civitai_value_is_set() {
    [ -n "$1" ] && [ "$1" != "replace_with_ids" ]
}

_civitai_trim() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "$value"
}

prepare_civitai_downloads_if_requested() {
    # Resolve aliases before any background pip install starts. If this runtime
    # needs to repair the downloader's dependency, it must own pip exclusively.
    # The actual model transfers still run later, after the HF manager.
    # shellcheck disable=SC1091
    source "$RUNTIME_DIR/src/civitai_env.sh"
    resolve_civitai_env

    if ! _civitai_value_is_set "${CIVITAI_CHECKPOINTS:-}" \
        && ! _civitai_value_is_set "${CIVITAI_LORAS:-}"; then
        CIVITAI_PREPARE_STATE="not_requested"
        return 0
    fi
    if prepare_civitai_downloader; then
        CIVITAI_PREPARE_STATE="ready"
        return 0
    fi
    CIVITAI_PREPARE_STATE="failed"
    return 1
}

run_civitai_downloads() {
    local persist_root="$1"
    local checkpoints_dir="$persist_root/models/checkpoints"
    local loras_dir="$persist_root/models/loras"
    local requested=""
    local category target_dir values identifier pid index
    local download_count=0
    local failure_count=0
    local -a categories=("checkpoints" "loras")
    local -a download_pids=()
    local -a download_categories=()

    if [ -z "$CIVITAI_PREPARE_STATE" ]; then
        prepare_civitai_downloads_if_requested
    fi

    if _civitai_value_is_set "${CIVITAI_CHECKPOINTS:-}" \
        || _civitai_value_is_set "${CIVITAI_LORAS:-}"; then
        requested=1
    fi
    if [ -z "$requested" ]; then
        echo "⏭️  Skipping CivitAI checkpoint downloads (no IDs set)"
        echo "⏭️  Skipping CivitAI LoRA downloads (no IDs set)"
        return 0
    fi

    if [ "$CIVITAI_PREPARE_STATE" != "ready" ]; then
        return 1
    fi

    for category in "${categories[@]}"; do
        case "$category" in
            checkpoints)
                target_dir="$checkpoints_dir"
                values="${CIVITAI_CHECKPOINTS:-}"
                ;;
            loras)
                target_dir="$loras_dir"
                values="${CIVITAI_LORAS:-}"
                ;;
        esac
        mkdir -p "$target_dir"
        if ! _civitai_value_is_set "$values"; then
            echo "⏭️  Skipping CivitAI $category downloads (no IDs set)"
            continue
        fi

        local -a identifiers=()
        IFS=',' read -r -a identifiers <<< "$values"
        for identifier in "${identifiers[@]}"; do
            identifier="$(_civitai_trim "$identifier")"
            [ -n "$identifier" ] || continue
            echo "🚀 Scheduling CivitAI $category download to $target_dir"
            "$CIVITAI_PYTHON" "$CIVITAI_DOWNLOADER" \
                --identifier "$identifier" --output "$target_dir" &
            pid=$!
            download_pids+=("$pid")
            download_categories+=("$category")
            download_count=$((download_count + 1))
        done
    done

    if [ "$download_count" -eq 0 ]; then
        echo "⏭️  No valid CivitAI identifiers were provided"
        return 0
    fi

    echo "📋 Scheduled $download_count CivitAI downloads in background"
    echo "⏳ Waiting for CivitAI downloads to complete..."
    for index in "${!download_pids[@]}"; do
        if ! wait "${download_pids[$index]}"; then
            failure_count=$((failure_count + 1))
            _civitai_warn "CivitAI ${download_categories[$index]} download failed"
        fi
    done

    if [ "$failure_count" -gt 0 ]; then
        echo "❌ $failure_count of $download_count CivitAI downloads failed; booting anyway"
        return 1
    fi
    echo "✅ All $download_count CivitAI downloads finished"
}
