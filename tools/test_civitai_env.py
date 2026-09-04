#!/usr/bin/env python3
"""Self-check for the CivitAI env alias mapping (src/civitai_env.sh).

The canonical customer-facing names are CIVITAI_LORAS and CIVITAI_CHECKPOINTS
(qwen's spelling, standardized family-wide 2026-08-13). Customers have
the legacy names saved in RunPod template configs and will redeploy with them
for months, so the runtime must keep accepting them forever (CLAUDE.md:
never drop an env value customers have saved):

    CIVITAI_LORAS        <- LORAS_IDS_TO_DOWNLOAD
    CIVITAI_CHECKPOINTS  <- CHECKPOINT_IDS_TO_DOWNLOAD, SDXL_MODEL_IDS_TO_DOWNLOAD

Canonical wins when both are set. A legacy name keeps working and lands a
plain renamed-variable notice in the boot report, never an error. The token
reaches the downloader as CIVITAI_TOKEN or civitai_token (what
download_with_aria.py reads); CIVITAI_API_KEY is accepted and mapped.

Everything runs bash in a subprocess with a stubbed report_warn: no pod, no
network. Run: python3 tools/test_civitai_env.py
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SH = REPO / "src" / "civitai_env.sh"

CHECKS = 0


def ok(cond, msg):
    global CHECKS
    assert cond, msg
    CHECKS += 1


def resolve(env):
    """Source civitai_env.sh with the given env, run resolve_civitai_env,
    print the resolved values and any warnings. Returns (values, warns)."""
    script = (
        'report_warn() { printf "WARN:%s\\n" "$1"; }\n'
        f'source "{SH}"\n'
        "resolve_civitai_env\n"
        'printf "LORAS=%s\\n" "$CIVITAI_LORAS"\n'
        'printf "CHECKPOINTS=%s\\n" "$CIVITAI_CHECKPOINTS"\n'
        'printf "TOKEN=%s\\n" "$CIVITAI_TOKEN"\n'
        'printf "token_lc=%s\\n" "$civitai_token"\n'
        'printf "API_KEY_SET=%s\\n" "${CIVITAI_API_KEY+x}"\n'
    )
    r = subprocess.run(["bash", "-c", script], env={"PATH": "/usr/bin:/bin",
                                                    **env},
                       capture_output=True, text=True)
    ok(r.returncode == 0, f"bash exited {r.returncode}: {r.stderr}")
    values, warns = {}, []
    for line in r.stdout.splitlines():
        if line.startswith("WARN:"):
            warns.append(line[5:])
        else:
            k, _, v = line.partition("=")
            values[k] = v
    return values, warns


def test_canonical_only():
    v, w = resolve({"CIVITAI_LORAS": "111,222", "CIVITAI_CHECKPOINTS": "333"})
    ok(v["LORAS"] == "111,222", v)
    ok(v["CHECKPOINTS"] == "333", v)
    ok(w == [], f"canonical names must not warn: {w}")


def test_legacy_only_maps_and_warns():
    v, w = resolve({"LORAS_IDS_TO_DOWNLOAD": "444",
                    "CHECKPOINT_IDS_TO_DOWNLOAD": "555"})
    ok(v["LORAS"] == "444", v)
    ok(v["CHECKPOINTS"] == "555", v)
    ok(len(w) == 2, f"one notice per legacy name used: {w}")
    ok(any("LORAS_IDS_TO_DOWNLOAD" in x and "CIVITAI_LORAS" in x for x in w),
       f"the notice must name both the old and the new name: {w}")
    ok(any("CHECKPOINT_IDS_TO_DOWNLOAD" in x and "CIVITAI_CHECKPOINTS" in x
           for x in w), w)
    for x in w:
        ok("error" not in x.lower() and "fail" not in x.lower(),
           f"the notice must be plain, not scary: {x}")


def test_sdxl_legacy_maps_to_checkpoints():
    # ltx2's historical name for the checkpoints slot.
    v, w = resolve({"SDXL_MODEL_IDS_TO_DOWNLOAD": "666"})
    ok(v["CHECKPOINTS"] == "666", v)
    ok(any("SDXL_MODEL_IDS_TO_DOWNLOAD" in x and "CIVITAI_CHECKPOINTS" in x
           for x in w), w)


def test_both_set_canonical_wins():
    v, w = resolve({"CIVITAI_LORAS": "111",
                    "LORAS_IDS_TO_DOWNLOAD": "999",
                    "CIVITAI_CHECKPOINTS": "222",
                    "CHECKPOINT_IDS_TO_DOWNLOAD": "888",
                    "SDXL_MODEL_IDS_TO_DOWNLOAD": "777"})
    ok(v["LORAS"] == "111", v)
    ok(v["CHECKPOINTS"] == "222", v)
    ok(w == [], f"no warning when the canonical name is in use: {w}")


def test_neither_set():
    v, w = resolve({})
    ok(v["LORAS"] == "", v)
    ok(v["CHECKPOINTS"] == "", v)
    ok(w == [], w)


def test_placeholder_counts_as_unset():
    # RunPod templates ship replace_with_ids as the field placeholder; a
    # customer who filled in only the legacy field must still get their IDs.
    v, w = resolve({"CIVITAI_LORAS": "replace_with_ids",
                    "LORAS_IDS_TO_DOWNLOAD": "444"})
    ok(v["LORAS"] == "444", v)
    ok(len(w) == 1, w)


def test_api_key_maps_to_token():
    v, w = resolve({"CIVITAI_API_KEY": "sk-abc"})
    ok(v["TOKEN"] == "sk-abc", v)
    ok(v["API_KEY_SET"] == "", "the legacy alias must be removed before "
       f"the downloader spawns aria2c: {v}")


def test_existing_token_wins_over_api_key():
    v, w = resolve({"civitai_token": "sk-low", "CIVITAI_API_KEY": "sk-abc"})
    ok(v["token_lc"] == "sk-low", v)
    ok(v["TOKEN"] == "", f"CIVITAI_TOKEN must stay empty when the downloader "
                         f"already sees civitai_token: {v}")
    ok(v["API_KEY_SET"] == "", "unused CIVITAI_API_KEY must not survive into "
       f"the downloader environment: {v}")


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print(f"civitai env self-test: all good ({CHECKS} assertions)")


if __name__ == "__main__":
    sys.exit(main())
