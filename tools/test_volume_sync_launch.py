#!/usr/bin/env python3
"""Self-check for the stage-to-volume launch gate in src/start.sh.

The real shell block between the VOLUME-SYNC-LAUNCH markers is executed with a
temporary manifest and a harmless volume_sync.py stub. This pins the public
contract without reimplementing the launch decision in the test:

  - unset/default starts the detached copy for a live staged symlink;
  - only a trimmed, case-insensitive literal ``false`` disables it;
  - a typo keeps persistence on;
  - no pending symlink is a no-op;
  - no network volume wins over the env setting and never launches a copy.

No network, pod, third-party package, or model bytes are required.
Run: python3 tools/test_volume_sync_launch.py
"""
import os
import subprocess
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
START_SH = REPO / "src" / "start.sh"
MARK_START = "# >>> VOLUME-SYNC-LAUNCH"
MARK_END = "# <<< VOLUME-SYNC-LAUNCH"
NO_ENV = object()
CHECKS = 0


def ok(condition, message):
    global CHECKS
    assert condition, message
    CHECKS += 1


def extract_block() -> str:
    """Return the shipped launch block verbatim."""
    text = START_SH.read_text()
    start = text.find(MARK_START)
    end = text.find(MARK_END)
    ok(start != -1, f"{MARK_START} marker missing from src/start.sh")
    ok(end > start, f"{MARK_END} marker missing or misplaced in src/start.sh")
    return text[start:end]


def run_gate(value=NO_ENV, *, pending=True, network_volume=True):
    """Execute the real shell gate and return launch/report/stdout evidence."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        volume = root / "workspace"
        volume.mkdir()
        stage = root / "hf_stage"
        stage.mkdir()
        models = volume / "ComfyUI" / "models" / "checkpoints"
        models.mkdir(parents=True)

        dest = models / "model.safetensors"
        if pending:
            target = stage / dest.name
            target.write_bytes(b"model-bytes")
            dest.symlink_to(target)
        else:
            dest.write_bytes(b"already-persistent")

        manifest = root / "manifest.tsv"
        manifest.write_text(f"https://example.invalid/model\t{dest}\t0\n")

        runtime = root / "runtime"
        (runtime / "src").mkdir(parents=True)
        launched = root / "launched.txt"
        (runtime / "src" / "volume_sync.py").write_text(
            "import os, sys\n"
            "from pathlib import Path\n"
            "Path(os.environ['LAUNCHED_FILE']).write_text(sys.argv[1])\n"
        )

        report = root / "report.tsv"
        script = root / "gate.sh"
        script.write_text(
            "report_kv() { printf 'set\\t%s\\t%s\\n' \"$1\" \"$2\" >> \"$REPORT_FILE\"; }\n"
            + extract_block()
            + "\nwait\n"
        )

        env = dict(os.environ)
        env.update({
            "HOME": str(root),
            "HF_QUEUE_FILE": str(manifest),
            "LAUNCHED_FILE": str(launched),
            "NETWORK_VOLUME": str(volume) if network_volume else "/",
            "REPORT_FILE": str(report),
            "RUNTIME_DIR": str(runtime),
        })
        env.pop("PERSIST_MODELS_TO_VOLUME", None)
        if value is not NO_ENV:
            env["PERSIST_MODELS_TO_VOLUME"] = value

        result = subprocess.run(
            ["bash", str(script)], capture_output=True, text=True, env=env
        )
        ok(result.returncode == 0, result.stdout + result.stderr)
        launched_manifest = launched.read_text() if launched.is_file() else None
        report_text = report.read_text() if report.is_file() else ""
        log_text = (volume / "comfyui.log").read_text() \
            if (volume / "comfyui.log").is_file() else ""
        return launched_manifest, report_text, result.stdout + result.stderr + log_text


def test_default_and_explicit_on_launch():
    for value in (NO_ENV, "", "true", "TRUE", "yes", "1"):
        launched, report, output = run_gate(value)
        ok(launched is not None, f"{value!r} must start volume_sync.py: {output}")
        ok(launched.endswith("manifest.tsv"), launched)
        ok("set\tvolume_sync\trunning" in report, report)
        ok("copying them to your network volume" in output, output)


def test_trimmed_case_insensitive_false_disables():
    for value in ("false", "False", "FALSE", " FaLsE ", "\tfalse\n"):
        launched, report, output = run_gate(value)
        ok(launched is None, f"{value!r} unexpectedly launched the copy")
        ok("set\tvolume_sync\tdisabled_by_env" in report, report)
        ok("PERSIST_MODELS_TO_VOLUME=false" in output, output)
        ok("local disk is discarded" in output, output)


def test_only_literal_false_disables():
    for value in ("0", "no", "off", "f alse", "false-ish", "typo"):
        launched, report, output = run_gate(value)
        ok(launched is not None, f"{value!r} silently disabled persistence: {output}")
        ok("set\tvolume_sync\trunning" in report, report)


def test_no_pending_symlink_is_a_noop():
    launched, report, output = run_gate("false", pending=False)
    ok(launched is None, "a real destination file must not launch a copy")
    ok("volume_sync" not in report, report)
    ok("PERSIST_MODELS_TO_VOLUME" not in output, output)


def test_no_network_volume_never_launches():
    for value in (NO_ENV, "false", "true"):
        launched, report, output = run_gate(value, network_volume=False)
        ok(launched is None, f"no-volume state launched a copy for {value!r}")
        ok("set\tvolume_sync\tskipped_no_volume" in report, report)
        ok("no network volume" in output, output)
        ok("disabled_by_env" not in report, report)


def test_contract_documents_the_switch():
    contract = (REPO / "CONTRACTS.md").read_text()
    architecture = (REPO / "ARCHITECTURE.md").read_text()
    for text, label in ((contract, "CONTRACTS.md"),
                        (architecture, "ARCHITECTURE.md")):
        ok("PERSIST_MODELS_TO_VOLUME" in text,
           f"{label} does not document PERSIST_MODELS_TO_VOLUME")


def main():
    test_default_and_explicit_on_launch()
    test_trimmed_case_insensitive_false_disables()
    test_only_literal_false_disables()
    test_no_pending_symlink_is_a_noop()
    test_no_network_volume_never_launches()
    test_contract_documents_the_switch()
    print(f"volume sync launch self-test: all good ({CHECKS} assertions)")


if __name__ == "__main__":
    main()
