#!/usr/bin/env python3
"""Exercise the real ComfyUI startup-liveness shell block.

The harness extracts the marked block from src/start.sh and runs it under bash
with stubbed curl and sleep commands. A real background child supplies the PID,
so the tests cover the user-visible state transitions without a GPU or pod:

  - HTTP becomes ready -> ready=true;
  - startup crosses 70 seconds while the PID lives -> still starting, not failed;
  - the PID exits before HTTP readiness -> ready=false with log diagnostics;
  - every readiness sleep is five seconds.

Run: python3 tools/test_comfyui_liveness.py
"""
import os
import subprocess
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
START = REPO / "src" / "start.sh"
CHECKS = 0


def ok(condition, message):
    global CHECKS
    assert condition, message
    CHECKS += 1


def extract_block():
    lines = START.read_text().splitlines()
    begin = [i for i, line in enumerate(lines)
             if "comfyui liveness: begin" in line]
    end = [i for i, line in enumerate(lines)
           if "comfyui liveness: end" in line]
    ok(len(begin) == 1 and len(end) == 1 and begin[0] < end[0],
       "src/start.sh must contain exactly one comfyui liveness marker pair; "
       f"begin={begin}, end={end}")
    return "\n".join(lines[begin[0]:end[0] + 1])


CURL_STUB = """#!/bin/sh
count=$(cat "$CURL_COUNT_FILE")
count=$((count + 1))
printf '%s\n' "$count" > "$CURL_COUNT_FILE"
if [ "${CURL_ALWAYS_FAIL:-0}" = 1 ]; then
    exit 22
fi
if [ "$count" -le "${CURL_FAILURES_BEFORE_READY:-0}" ]; then
    exit 22
fi
exit 0
"""


SLEEP_STUB = """#!/bin/sh
printf '%s\n' "$1" >> "$SLEEP_CALLS_FILE"
if [ "${KILL_COMFYUI_ON_FIRST_SLEEP:-0}" = 1 ] && [ ! -e "$KILL_MARKER" ]; then
    : > "$KILL_MARKER"
    kill "$COMFYUI_PID"
    /bin/sleep 0.05
fi
exit 0
"""


def run_case(*, failures_before_ready=0, always_fail=False,
             kill_on_first_sleep=False):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        stub_bin = root / "bin"
        stub_bin.mkdir()
        for name, body in (("curl", CURL_STUB), ("sleep", SLEEP_STUB)):
            path = stub_bin / name
            path.write_text(body)
            path.chmod(0o755)

        log = root / "comfyui_testpod_nohup.log"
        log.write_text("first startup line\n\nlatest useful startup line\n")
        state = root / "boot_state.tsv"
        state.write_text("")
        curl_count = root / "curl_count"
        curl_count.write_text("0\n")
        sleep_calls = root / "sleep_calls"
        sleep_calls.write_text("")

        script = root / "run.sh"
        script.write_text(
            "report_kv() { printf 'set\\t%s\\t%s\\n' \"$1\" \"$2\" >> \"$BOOT_STATE\"; }\n"
            "report_warn() { printf 'warn\\t%s\\n' \"$1\" >> \"$BOOT_STATE\"; }\n"
            "/bin/sleep 60 &\n"
            "COMFYUI_PID=$!\n"
            "export COMFYUI_PID\n"
            + extract_block()
            + "\nkill \"$COMFYUI_PID\" 2>/dev/null || true\n"
            "wait \"$COMFYUI_PID\" 2>/dev/null || true\n"
        )

        env = dict(os.environ)
        env.update({
            "PATH": f"{stub_bin}:{env['PATH']}",
            "BOOT_STATE": str(state),
            "COMFYUI_LOG": str(log),
            "COMFYUI_VERSION": "approved",
            "CURL_ALWAYS_FAIL": "1" if always_fail else "0",
            "CURL_COUNT_FILE": str(curl_count),
            "CURL_FAILURES_BEFORE_READY": str(failures_before_ready),
            "KILL_COMFYUI_ON_FIRST_SLEEP":
                "1" if kill_on_first_sleep else "0",
            "KILL_MARKER": str(root / "killed"),
            "NETWORK_VOLUME": str(root),
            "RUNPOD_POD_ID": "testpod",
            "SLEEP_CALLS_FILE": str(sleep_calls),
            "TORCH_CUDA_MAJOR": "13",
            "URL": "http://127.0.0.1:8188",
        })
        result = subprocess.run(
            ["bash", str(script)], capture_output=True, text=True,
            env=env, timeout=10,
        )
        ok(result.returncode == 0,
           f"liveness harness exited {result.returncode}:\n"
           f"{result.stdout}\n{result.stderr}")
        return {
            "output": result.stdout + result.stderr,
            "state": state.read_text(),
            "sleeps": sleep_calls.read_text().splitlines(),
            "curl_count": int(curl_count.read_text()),
        }


def test_launch_captures_the_python_pid():
    source = START.read_text()
    launch_at = source.index('nohup python3 "$COMFYUI_DIR/main.py"')
    pid_at = source.index("COMFYUI_PID=$!", launch_at)
    liveness_at = source.index("comfyui liveness: begin", launch_at)
    ok(launch_at < pid_at < liveness_at,
       "the launch must capture $! before the liveness block")


def test_ready_uses_five_second_poll():
    result = run_case(failures_before_ready=1)
    ok("set\tready\ttrue" in result["state"], result["state"])
    ok("set\tready\tfalse" not in result["state"], result["state"])
    ok(result["sleeps"] == ["5"], result["sleeps"])
    ok(result["curl_count"] == 2, result["curl_count"])
    ok("FAILED" not in result["output"], result["output"])


def test_slow_live_process_keeps_starting():
    # Fifteen failed checks advance the synthetic elapsed counter to 75s.
    result = run_case(failures_before_ready=15)
    ok("set\tready\ttrue" in result["state"], result["state"])
    ok("set\tready\tfalse" not in result["state"], result["state"])
    ok(len(result["sleeps"]) == 15, result["sleeps"])
    ok(set(result["sleeps"]) == {"5"}, result["sleeps"])
    ok("still starting after 70s" in result["output"], result["output"])
    ok("Latest log: latest useful startup line" in result["output"],
       result["output"])
    ok("FAILED" not in result["output"], result["output"])
    ok("Troubleshooting Tips" not in result["output"], result["output"])


def test_exited_process_fails_with_log_tail():
    result = run_case(always_fail=True, kill_on_first_sleep=True)
    ok("set\tready\tfalse" in result["state"], result["state"])
    ok("set\tready\ttrue" not in result["state"], result["state"])
    ok("ComfyUI process exited" in result["output"], result["output"])
    ok("latest useful startup line" in result["output"], result["output"])
    ok("Troubleshooting Tips" in result["output"], result["output"])
    ok("did not answer on port 8188 within 70s" not in result["state"],
       result["state"])
    ok(result["sleeps"] == ["5"], result["sleeps"])


def test_contract_documents_process_aware_wait():
    contract = (REPO / "CONTRACTS.md").read_text()
    architecture = (REPO / "ARCHITECTURE.md").read_text()
    for text, label in ((contract, "CONTRACTS.md"),
                        (architecture, "ARCHITECTURE.md")):
        ok("5 seconds" in text, f"{label} does not document the poll interval")
        ok("still starting" in text,
           f"{label} does not document the slow-start state")


def main():
    test_launch_captures_the_python_pid()
    test_ready_uses_five_second_poll()
    test_slow_live_process_keeps_starting()
    test_exited_process_fails_with_log_tail()
    test_contract_documents_process_aware_wait()
    print(f"ComfyUI liveness self-test: all good ({CHECKS} assertions)")


if __name__ == "__main__":
    main()
