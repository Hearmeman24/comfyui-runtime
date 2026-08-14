#!/usr/bin/env python3
"""Offline self-test for the comfy_extra_args block in src/start.sh.

Run: python3 tools/test_comfy_extra_args.py

Extracts the block between the "comfy extra args" markers out of the SHIPPED
start.sh, plus the real nohup launch line, and runs both in bash against a
stub `nohup` that records argv. Nothing is duplicated here: if the block or
the launch line changes shape, this test runs the new one.

Why this exists: a template.json default and a customer env var both feed the
same launch, and getting the precedence backwards is invisible in review but
decides whether an upstream-bug workaround survives a customer setting an
unrelated flag. Stdlib only, no pytest.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
START = REPO / "src" / "start.sh"

CHECKS: list[tuple[bool, str]] = []


def ok(cond, label):
    CHECKS.append((bool(cond), label))
    print(("ok   " if cond else "FAIL ") + label)


def find_bash4():
    """start.sh needs bash, not sh. macOS /bin/bash is 3.2; prefer a modern one."""
    for cand in ("/opt/homebrew/bin/bash", "/usr/local/bin/bash", "/bin/bash"):
        if Path(cand).exists():
            return cand
    return "bash"


BASH4 = find_bash4()

# Records the launch argv, one entry per line, then exits instead of running.
NOHUP_STUB = """#!/usr/bin/env bash
: > "$ARGV_FILE"
for a in "$@"; do printf '%s\\n' "$a" >> "$ARGV_FILE"; done
exit 0
"""


def extract_marked(marker):
    lines = START.read_text().splitlines()
    beg = [i for i, l in enumerate(lines) if f"{marker}: begin" in l]
    end = [i for i, l in enumerate(lines) if f"{marker}: end" in l]
    ok(len(beg) == 1 and len(end) == 1 and beg[0] < end[0],
       f"start.sh must carry exactly one '{marker}' begin/end marker pair "
       f"(found begin={beg} end={end})")
    return "\n".join(lines[beg[0]:end[0] + 1])


def extract_launch():
    lines = START.read_text().splitlines()
    idx = [i for i, l in enumerate(lines)
           if l.startswith('nohup python3 "$COMFYUI_DIR/main.py"')]
    ok(len(idx) == 1, f"expected exactly one ComfyUI launch line, found {idx}")
    i = idx[0]
    out = [lines[i]]
    while out[-1].rstrip().endswith("\\"):
        i += 1
        out.append(lines[i])
    return "\n".join(out)


# template_json_get is defined near the top of start.sh and the block calls it,
# so the harness needs the real one rather than a reimplementation.
def extract_template_json_get():
    lines = START.read_text().splitlines()
    beg = [i for i, l in enumerate(lines) if l.startswith("template_json_get()")]
    ok(len(beg) == 1, "expected exactly one template_json_get definition")
    i = beg[0]
    out = []
    while True:
        out.append(lines[i])
        if lines[i] == "}":
            break
        i += 1
    return "\n".join(out)


def run_boot(template_json: dict, env_extra: str | None):
    """Run the real block + launch line; return (argv list, stdout)."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        stub_bin = tmp / "bin"
        stub_bin.mkdir()
        p = stub_bin / "nohup"
        p.write_text(NOHUP_STUB)
        p.chmod(0o755)
        (tmp / "ComfyUI").mkdir()
        (tmp / "nv").mkdir()
        tjson = tmp / "template.json"
        tjson.write_text(json.dumps(template_json))
        argv_file = tmp / "argv"

        script = tmp / "boot.sh"
        script.write_text("\n".join([
            "#!/usr/bin/env bash",
            'TEMPLATE_JSON="$TEMPLATE_JSON"',
            extract_template_json_get(),
            extract_marked("comfy extra args"),
            "SAGE_FLAG=''",
            extract_launch(),
            "wait\n",
        ]))
        env = {
            "PATH": f"{stub_bin}:{os.environ['PATH']}",
            "HOME": os.environ.get("HOME", str(tmp)),
            "TEMPLATE_JSON": str(tjson),
            "COMFYUI_DIR": str(tmp / "ComfyUI"),
            "NETWORK_VOLUME": str(tmp / "nv"),
            "RUNPOD_POD_ID": "testpod",
            "EXTRA_PATHS_FLAG": "",
            "ARGV_FILE": str(argv_file),
        }
        if env_extra is not None:
            env["COMFY_EXTRA_ARGS"] = env_extra
        r = subprocess.run([BASH4, str(script)], env=env,
                           capture_output=True, text=True, timeout=60)
        ok(r.returncode == 0, f"harness exited {r.returncode}: {r.stderr}")
        argv = argv_file.read_text().splitlines() if argv_file.exists() else []
        return argv, r.stdout


def flags(argv):
    """Just the launch flags, dropping python3/main.py and the fixed args."""
    return [a for a in argv if a.startswith("--")]


# --- tests ------------------------------------------------------------------

def test_neither_set_is_unchanged():
    """The no-config case must be byte for byte what shipped before, or this
    change silently alters every template that does not opt in."""
    argv, out = run_boot({}, None)
    f = flags(argv)
    ok("--disable-dynamic-vram" not in f,
       "no template key and no env var: no extra flags reach the launch")
    ok(f == ["--listen", "--enable-cors-header"],
       f"the untouched launch keeps exactly its fixed flags (got {f})")
    ok("Template ComfyUI args" not in out,
       "and nothing is echoed about template args when the key is absent")


def test_template_default_reaches_the_launch():
    argv, out = run_boot({"comfy_extra_args": "--disable-dynamic-vram"}, None)
    ok("--disable-dynamic-vram" in argv,
       "a template.json comfy_extra_args value reaches the ComfyUI launch")
    ok("🧩 Template ComfyUI args: --disable-dynamic-vram" in out,
       "and is echoed at boot so a support log shows it was applied")


def test_customer_env_still_works_alone():
    argv, _ = run_boot({}, "--lowvram")
    ok("--lowvram" in argv,
       "COMFY_EXTRA_ARGS alone still reaches the launch (unchanged behaviour)")


def test_both_are_concatenated_customer_last():
    """The precedence that matters. A customer setting an unrelated flag must
    NOT silently drop the template's upstream-bug workaround."""
    argv, _ = run_boot({"comfy_extra_args": "--disable-dynamic-vram"}, "--lowvram")
    ok("--disable-dynamic-vram" in argv and "--lowvram" in argv,
       "both survive: a customer flag does not replace the template default")
    ok(argv.index("--disable-dynamic-vram") < argv.index("--lowvram"),
       "template flags come FIRST so the customer's win on any conflict")


def test_customer_can_override_the_template_default():
    """cli_args.enables_dynamic_vram() early-returns True on
    --enable-dynamic-vram, so this pair is the documented escape hatch."""
    argv, _ = run_boot({"comfy_extra_args": "--disable-dynamic-vram"},
                       "--enable-dynamic-vram")
    ok(argv.index("--disable-dynamic-vram") < argv.index("--enable-dynamic-vram"),
       "a customer can put a template default back: their flag lands later")


def test_multiple_flags_word_split():
    argv, _ = run_boot({"comfy_extra_args": "--disable-dynamic-vram --lowvram"}, None)
    ok("--disable-dynamic-vram" in argv and "--lowvram" in argv,
       "a template value carrying two flags word-splits into two argv entries")
    ok(not any(" " in a for a in argv),
       "no argv entry keeps an embedded space (the value is not passed whole)")


def test_no_empty_argv_entries():
    """Concatenation puts a space between the two sources. Unquoted expansion
    must collapse it, or ComfyUI receives an empty positional argument."""
    for tj, ev in (({}, None), ({}, ""),
                   ({"comfy_extra_args": "--disable-dynamic-vram"}, ""),
                   ({"comfy_extra_args": ""}, "--lowvram")):
        argv, _ = run_boot(tj, ev)
        ok("" not in argv,
           f"no empty argv entry reaches ComfyUI (template={tj}, env={ev!r})")


def main():
    for t in (test_neither_set_is_unchanged,
              test_template_default_reaches_the_launch,
              test_customer_env_still_works_alone,
              test_both_are_concatenated_customer_last,
              test_customer_can_override_the_template_default,
              test_multiple_flags_word_split,
              test_no_empty_argv_entries):
        t()
    failed = [label for good, label in CHECKS if not good]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        for label in failed:
            print(f"FAILED: {label}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
