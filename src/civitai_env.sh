# shellcheck shell=bash
# Canonical CivitAI env names, sourced by start.sh before the CivitAI ID
# downloads. Standardized on qwen's spelling (2026-08-13):
#
#     CIVITAI_LORAS        comma-separated LoRA version IDs
#     CIVITAI_CHECKPOINTS  comma-separated checkpoint version IDs
#
# The legacy names stay accepted forever: customers have them saved in RunPod
# template configs and will redeploy with them for months (CLAUDE.md: never
# drop an env value customers have saved). Canonical wins when both are set;
# a legacy name keeps working and lands a plain renamed-variable notice in
# the boot report, never an error.
#
# Callers provide report_warn (start.sh defines it; tests stub it).

# RunPod template fields ship this placeholder; treat it as unset.
_civitai_ids_set() { [ -n "$1" ] && [ "$1" != "replace_with_ids" ]; }

resolve_civitai_env() {
    local legacy val
    if ! _civitai_ids_set "${CIVITAI_LORAS:-}"; then
        CIVITAI_LORAS=""
        if _civitai_ids_set "${LORAS_IDS_TO_DOWNLOAD:-}"; then
            CIVITAI_LORAS="$LORAS_IDS_TO_DOWNLOAD"
            report_warn "LORAS_IDS_TO_DOWNLOAD still works, but its new name is CIVITAI_LORAS; please rename it in your template"
        fi
    fi
    if ! _civitai_ids_set "${CIVITAI_CHECKPOINTS:-}"; then
        CIVITAI_CHECKPOINTS=""
        for legacy in CHECKPOINT_IDS_TO_DOWNLOAD SDXL_MODEL_IDS_TO_DOWNLOAD; do
            val="${!legacy:-}"
            if _civitai_ids_set "$val"; then
                CIVITAI_CHECKPOINTS="$val"
                report_warn "$legacy still works, but its new name is CIVITAI_CHECKPOINTS; please rename it in your template"
                break
            fi
        done
    fi
    # download_with_aria.py reads CIVITAI_TOKEN or civitai_token. Accept
    # CIVITAI_API_KEY as well (CLAUDE.md section 3) by mapping it onto
    # CIVITAI_TOKEN when neither name the downloader reads is set.
    if [ -z "${CIVITAI_TOKEN:-}" ] && [ -z "${civitai_token:-}" ] \
        && [ -n "${CIVITAI_API_KEY:-}" ]; then
        CIVITAI_TOKEN="$CIVITAI_API_KEY"
    fi
    export CIVITAI_LORAS CIVITAI_CHECKPOINTS CIVITAI_TOKEN civitai_token
}
