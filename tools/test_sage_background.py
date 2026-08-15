#!/usr/bin/env python3
"""Self-check for the backgrounded SageAttention install + probe in
src/start.sh (EXECUTION.md item E10).

The wheel install and the kernel probe run in one background subshell so they
overlap provisioning and the model downloads. The ComfyUI launch line
interpolates SAGE_FLAG, so the subshell is NOT fire-and-forget: it writes its
verdict (probe exit code and message) to files, and a join immediately above
the launch waits on it, reads the verdict, sets SAGE_FLAG and records the
sage report keys. A naive `&` would launch ComfyUI with an empty SAGE_FLAG on
a fast boot and silently disable SageAttention; these tests pin that down:

  - probe passes        -> --use-sage-attention IS in the captured launch argv
  - probe unsupported/2 -> it is NOT, and the report says unsupported
  - probe fails/1       -> it is NOT, and the report says probe_failed
  - subshell verdict missing entirely -> probe_failed, launch still happens
  - template sage != true -> no install, no probe, no join wait, no hang
  - the join happens BEFORE the launch (static order in the file), and the
    probe really overlaps the download phase (the stub probe blocks until the
    stand-in download step has run, so a synchronous regression fails)

Both start.sh blocks are exercised for real: extracted between their
begin/end markers and run under bash with a stubbed PATH (python3, pip,
nohup) standing in for the pod. No network, no GPU, no pod: the probe is a
stub, so this proves the plumbing, not the silicon.
Run: python3 tools/test_sage_background.py
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
START = REPO / "src" / "start.sh"

CHECKS = 0
SKIPPED = []


def find_bash4():
    """start.sh needs bash 4+ (declare -A, already relied on elsewhere in the
    script). The pod (Ubuntu 24.04) and CI (cimg/python) both ship bash 5;
    macOS /bin/bash is 3.2, so dev machines probe for a newer one."""
    for cand in ("bash", "/opt/homebrew/bin/bash", "/usr/local/bin/bash"):
        try:
            r = subprocess.run([cand, "-c", "declare -A __x=()"],
                               capture_output=True)
        except OSError:
            continue
        if r.returncode == 0:
            return cand
    return None


BASH4 = find_bash4()


def ok(cond, msg):
    global CHECKS
    assert cond, msg
    CHECKS += 1


# --- extraction -------------------------------------------------------------

def extract_marked(marker):
    """Return (block text, begin line index) for a begin/end marker pair."""
    lines = START.read_text().splitlines()
    beg = [i for i, l in enumerate(lines) if f"{marker}: begin" in l]
    end = [i for i, l in enumerate(lines) if f"{marker}: end" in l]
    ok(len(beg) == 1 and len(end) == 1 and beg[0] < end[0],
       f"start.sh must carry exactly one '{marker}' begin/end marker pair "
       f"(found begin={beg} end={end})")
    return "\n".join(lines[beg[0]:end[0] + 1]), beg[0]


def extract_launch():
    """The nohup ComfyUI launch line with its backslash continuations."""
    lines = START.read_text().splitlines()
    idx = [i for i, l in enumerate(lines)
           if l.startswith('nohup python3 "$COMFYUI_DIR/main.py"')]
    ok(len(idx) == 1, f"expected exactly one ComfyUI launch line, found {idx}")
    i = idx[0]
    out = [lines[i]]
    while out[-1].rstrip().endswith("\\"):
        i += 1
        out.append(lines[i])
    return "\n".join(out), idx[0]


def line_index(needle):
    for i, l in enumerate(START.read_text().splitlines()):
        if needle in l:
            return i
    return None


# --- the harness ------------------------------------------------------------

PRELUDE = """\
report_kv()   { printf 'set\\t%s\\t%s\\n' "$1" "$2" >> "$BOOT_STATE"; }
report_warn() { printf 'warn\\t%s\\n' "$1" >> "$BOOT_STATE"; }
template_json_get() { if [ "$1" = sage ]; then printf '%s\\n' "$STUB_SAGE"; fi; }
"""

# Stands in for provisioning + the HF download manager: everything that runs
# between the sage spawn and the join on the real boot path.
DOWNLOADS_STANDIN = """\
echo "downloads stand-in running"
touch "$MARK_DIR/downloads_started"
"""

PYTHON3_STUB = """\
#!/bin/sh
case "$*" in
  *sage_probe.py*)
    echo "probe $*" >> "$CALL_LOG"
    if [ -n "${STUB_PROBE_WAIT_FOR:-}" ]; then
      n=0
      while [ ! -e "$STUB_PROBE_WAIT_FOR" ] && [ "$n" -lt 50 ]; do
        sleep 0.2; n=$((n+1))
      done
      if [ -e "$STUB_PROBE_WAIT_FOR" ]; then
        printf '%s' "overlap confirmed"; exit 0
      fi
      printf '%s' "probe never saw the download step: sage ran synchronously"
      exit 1
    fi
    printf '%s' "$STUB_PROBE_MSG"
    exit "$STUB_PROBE_RC"
    ;;
esac
exit 0
"""

PIP_STUB = '#!/bin/sh\necho "pip $*" >> "$CALL_LOG"\n'

NOHUP_STUB = '#!/bin/sh\nprintf \'%s\\n\' "$@" > "$ARGV_FILE"\n'


def run_boot(tmp, stub_sage, probe_rc="0", probe_msg="sage ok",
             probe_wait_for=None, drop_verdict=False):
    """One assembled boot: sage block, downloads stand-in, join, launch.
    Returns (stdout, argv list, boot state lines, call log lines)."""
    tmp = Path(tmp)
    stub_bin = tmp / "bin"
    stub_bin.mkdir(exist_ok=True)
    for name, body in (("python3", PYTHON3_STUB), ("pip", PIP_STUB),
                       ("nohup", NOHUP_STUB)):
        p = stub_bin / name
        p.write_text(body)
        p.chmod(0o755)
    for sub in ("ComfyUI", "nv", "marks"):
        (tmp / sub).mkdir(exist_ok=True)
    boot_state = tmp / "boot_state.tsv"
    call_log = tmp / "calls"
    argv_file = tmp / "argv"
    for f in (boot_state, call_log):
        f.write_text("")

    sage_block, _ = extract_marked("sage install + probe")
    join_block, _ = extract_marked("sage join")
    launch, _ = extract_launch()
    script = tmp / "boot.sh"
    script.write_text("\n".join([
        PRELUDE, sage_block, DOWNLOADS_STANDIN, join_block, launch,
        "wait\n",
    ]))
    # drop_verdict: the subshell dies before writing its files. Simulated by
    # pointing the verdict files at a directory that does not exist, so its
    # writes fail and the join finds nothing.
    verdict_dir = tmp / ("gone" if drop_verdict else ".")

    env = {
        "PATH": f"{stub_bin}:{os.environ['PATH']}",
        "HOME": os.environ.get("HOME", str(tmp)),
        "RUNTIME_DIR": str(REPO),
        "COMFYUI_DIR": str(tmp / "ComfyUI"),
        "NETWORK_VOLUME": str(tmp / "nv"),
        "RUNPOD_POD_ID": "testpod",
        "TORCH_CUDA_MAJOR": "13",
        "EXTRA_PATHS_FLAG": "",
        "COMFY_EXTRA_ARGS": "",
        "BOOT_STATE": str(boot_state),
        "CALL_LOG": str(call_log),
        "ARGV_FILE": str(argv_file),
        "MARK_DIR": str(tmp / "marks"),
        "STUB_SAGE": stub_sage,
        "STUB_PROBE_RC": probe_rc,
        "STUB_PROBE_MSG": probe_msg,
        # The verdict files default to /tmp; isolate each run.
        "SAGE_RC_FILE": str(verdict_dir / "sage_verdict.rc"),
        "SAGE_MSG_FILE": str(verdict_dir / "sage_verdict.msg"),
    }
    if probe_wait_for is not None:
        env["STUB_PROBE_WAIT_FOR"] = str(probe_wait_for)
    try:
        r = subprocess.run([BASH4, str(script)], env=env,
                           capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        ok(False, f"boot harness hung (sage={stub_sage}): the join must "
                  "never wait on a subshell that was not spawned")
    ok(r.returncode == 0, f"boot harness exited {r.returncode}: {r.stderr}")
    argv = argv_file.read_text().splitlines() if argv_file.exists() else []
    state = [l for l in boot_state.read_text().splitlines() if l]
    calls = [l for l in call_log.read_text().splitlines() if l]
    return r.stdout, argv, state, calls


def sage_state(state):
    return [l for l in state if l.split("\t")[1:2] == ["sage"]]


# --- tests ------------------------------------------------------------------

def test_static_order():
    """spawn < downloads < join < launch, all in the one file."""
    _, sage_beg = extract_marked("sage install + probe")
    _, join_beg = extract_marked("sage join")
    _, launch_at = extract_launch()
    # The INVOCATION, not any mention: a comment naming the file is not the
    # call, and matching one silently moved this boundary to the top of the
    # script and failed the ordering check for the wrong reason.
    downloads_at = line_index('python3 "$RUNTIME_DIR/src/hf_download_manager.py"')
    ok(downloads_at is not None, "download manager call not found in start.sh")
    ok(sage_beg < downloads_at,
       "the sage spawn must sit above the download manager")
    ok(downloads_at < join_beg,
       "the sage join must sit below the download manager")
    ok(join_beg < launch_at,
       "the sage join must sit ABOVE the ComfyUI launch: the launch line "
       "interpolates SAGE_FLAG")


def test_probe_pass_enables_flag():
    if BASH4 is None:
        SKIPPED.append("test_probe_pass_enables_flag (no bash 4+)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        out, argv, state, calls = run_boot(tmp, "true", probe_rc="0",
                                           probe_msg="sage kernel ok")
        ok("--use-sage-attention" in argv,
           f"probe rc 0 must put --use-sage-attention in the launch argv: {argv}")
        ok(sage_state(state) == ["set\tsage\tenabled"], sage_state(state))
        ok("set\tsage_msg\tsage kernel ok" in state, state)
        ok("sage kernel ok" in out, "the probe verdict line must be relayed")
        ok(len([c for c in calls if c.startswith("probe")]) == 1, calls)


def test_probe_unsupported_omits_flag():
    if BASH4 is None:
        SKIPPED.append("test_probe_unsupported_omits_flag (no bash 4+)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        _, argv, state, _ = run_boot(tmp, "true", probe_rc="2",
                                     probe_msg="unsupported arch")
        ok("--use-sage-attention" not in argv,
           f"probe rc 2 must NOT enable sage: {argv}")
        ok(sage_state(state) == ["set\tsage\tunsupported"], sage_state(state))


def test_probe_failure_omits_flag():
    if BASH4 is None:
        SKIPPED.append("test_probe_failure_omits_flag (no bash 4+)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        _, argv, state, _ = run_boot(tmp, "true", probe_rc="1",
                                     probe_msg="kernel crashed")
        ok("--use-sage-attention" not in argv,
           f"probe rc 1 must NOT enable sage: {argv}")
        ok(sage_state(state) == ["set\tsage\tprobe_failed"], sage_state(state))


def test_missing_verdict_is_probe_failed():
    """The subshell died without writing its verdict: fail safe, launch anyway."""
    if BASH4 is None:
        SKIPPED.append("test_missing_verdict_is_probe_failed (no bash 4+)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        _, argv, state, _ = run_boot(tmp, "true", drop_verdict=True)
        ok(argv, "ComfyUI must still launch when the verdict is missing")
        ok("--use-sage-attention" not in argv, argv)
        ok(sage_state(state) == ["set\tsage\tprobe_failed"], sage_state(state))


def test_template_sage_off_does_nothing():
    if BASH4 is None:
        SKIPPED.append("test_template_sage_off_does_nothing (no bash 4+)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        _, argv, state, calls = run_boot(tmp, "false")
        ok(calls == [], f"sage: false must run no install and no probe: {calls}")
        ok("--use-sage-attention" not in argv, argv)
        ok(sage_state(state) == ["set\tsage\toff_template"], sage_state(state))
        ok(argv, "ComfyUI must still launch on the sage-off path")


def test_install_and_probe_overlap_downloads():
    """The stub probe blocks until the downloads stand-in has run. A
    synchronous sage phase deadlocks against that and times out inside the
    stub, which then fails the probe: rc 1, no flag, and the message names
    the regression."""
    if BASH4 is None:
        SKIPPED.append("test_install_and_probe_overlap_downloads (no bash 4+)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        mark = Path(tmp) / "marks" / "downloads_started"
        _, argv, state, _ = run_boot(tmp, "true", probe_wait_for=mark)
        ok("set\tsage_msg\toverlap confirmed" in state,
           f"the probe must overlap the download phase, not precede it: {state}")
        ok("--use-sage-attention" in argv,
           f"the join must still gate the launch on the verdict: {argv}")


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    for note in SKIPPED:
        print(f"sage background self-test: SKIPPED {note}")
    print(f"sage background self-test: all good ({CHECKS} assertions, "
          f"{len(SKIPPED)} skipped)")


if __name__ == "__main__":
    sys.exit(main())
