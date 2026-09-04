#!/usr/bin/env python3
"""Offline integration tests for the runtime-owned CivitAI boot stage.

The shipped src/civitai_downloads.sh runs under bash with a fake Python
executable. The harness exercises selection, dependency preparation, token
mapping, argument fidelity, concurrent child joins, and degraded failures. It
also reads every active template's real entrypoint/pin to close the family-wide
adapter boundary without building or running a container image.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
FAMILY = REPO.parent
HELPER = REPO / "src" / "civitai_downloads.sh"
ENV_HELPER = REPO / "src" / "civitai_env.sh"
START = REPO / "src" / "start.sh"
BASE_DOCKERFILE = REPO / "base" / "Dockerfile"
SECRET = "civitai-CANARY-never-print-or-arg-123456"
FAMILIES = {
    "wan": (FAMILY / "comfyui-wan", "/comfyui-wan"),
    "minimax": (FAMILY / "comfyui-minimax", "/comfyui-minimax"),
    "qwen-image": (FAMILY / "comfyui-qwen-image", "/comfyui-qwen-template"),
    "ltx2": (FAMILY / "comfyui-ltx2", "/comfyui-ltx2"),
}


def find_bash4() -> str:
    for candidate in ("/opt/homebrew/bin/bash", "/usr/local/bin/bash", "/bin/bash"):
        if Path(candidate).exists():
            return candidate
    return "bash"


BASH = find_bash4()
CHECKS = 0
SKIPPED: list[str] = []


def ok(condition, message):
    global CHECKS
    assert condition, message
    CHECKS += 1


FAKE_PYTHON = r"""#!/usr/bin/env bash
if [ "$1" = "-c" ]; then
    case "$2" in
        *importlib.metadata*)
            [ -f "$FAKE_DEP_READY" ]
            exit $?
            ;;
        *)
            [ "${FAKE_SOURCE_INVALID:-0}" != "1" ]
            exit $?
            ;;
    esac
fi

if [ "$1" = "-m" ] && [ "$2" = "pip" ]; then
    printf 'PIP_INSTALL=%s\n' "$*" >> "$CAPTURE_FILE"
    if [ "${FAKE_PIP_FAIL:-0}" = "1" ]; then
        exit 8
    fi
    : > "$FAKE_DEP_READY"
    exit 0
fi

downloader="$1"
shift
identifier=""
output=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --identifier) identifier="$2"; shift 2 ;;
        --output) output="$2"; shift 2 ;;
        *) shift ;;
    esac
done
token_ok=0
[ "${CIVITAI_TOKEN:-${civitai_token:-}}" = "$EXPECTED_TOKEN" ] && token_ok=1
api_key_set=0
[ "${CIVITAI_API_KEY+x}" = "x" ] && api_key_set=1
printf 'DOWNLOADER=%s\tID=%s\tOUTPUT=%s\tTOKEN_OK=%s\tAPI_KEY_SET=%s\n' \
    "$downloader" "$identifier" "$output" "$token_ok" "$api_key_set" \
    >> "$CAPTURE_FILE"
printf 'INFO resolve model=1 version=2 file=3 name=model.safetensors format=SafeTensor\n'
if [[ "$identifier" == *fail* ]]; then
    printf 'ERROR failure stage=download message="simulated failure"\n'
    exit 7
fi
printf 'OK ready status=downloaded bytes=1 sha256=A files=1 path=%s/model.safetensors\n' "$output"
"""


LEGACY_DOWNLOADER = r"""#!/usr/bin/env bash
printf 'LEGACY_USED=1\n' >> "$CAPTURE_FILE"
exit 99
"""


@dataclass
class Run:
    rc: int
    output: str
    capture: list[str]
    report: str

    @property
    def downloads(self):
        return [line for line in self.capture if line.startswith("DOWNLOADER=")]


def make_runtime(root: Path, *, valid_vendor=True) -> Path:
    runtime = root / "runtime"
    (runtime / "src").mkdir(parents=True)
    shutil.copy2(ENV_HELPER, runtime / "src" / "civitai_env.sh")
    vendor = runtime / "vendor" / "civitai_downloader"
    vendor.mkdir(parents=True)
    source = vendor / "download_with_aria.py"
    source.write_text("#!/usr/bin/env python3\nprint('runtime marker')\n")
    requirements = vendor / "requirements.txt"
    requirements.write_text("requests==2.34.2\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if not valid_vendor:
        digest = "0" * 64
    requirements_digest = hashlib.sha256(requirements.read_bytes()).hexdigest()
    (vendor / "SHA256SUMS").write_text(
        f"{digest}  download_with_aria.py\n"
        f"{requirements_digest}  requirements.txt\n"
    )
    return runtime


def run_stage(
    *,
    runtime: Path = REPO,
    loras="model:2834417",
    checkpoints="",
    token_name="CIVITAI_API_KEY",
    fake_python=True,
    dependency_ready=True,
    pip_fails=False,
) -> Run:
    with tempfile.TemporaryDirectory() as temp_name:
        root = Path(temp_name)
        capture = root / "capture.tsv"
        report = root / "report.tsv"
        persistent = root / "workspace" / "ComfyUI"
        persistent.mkdir(parents=True)
        dep_ready = root / "dependency-ready"
        if dependency_ready:
            dep_ready.touch()

        bin_dir = root / "bin"
        bin_dir.mkdir()
        legacy = bin_dir / "download_with_aria.py"
        legacy.write_text(LEGACY_DOWNLOADER)
        legacy.chmod(0o755)

        python_path = sys.executable
        if fake_python:
            fake = root / "fake-python"
            fake.write_text(FAKE_PYTHON)
            fake.chmod(0o755)
            python_path = str(fake)

        script = root / "run.sh"
        script.write_text(
            "#!/usr/bin/env bash\n"
            'report_warn() { printf "warn\\t%s\\n" "$1" >> "$REPORT_FILE"; }\n'
            f'source "{HELPER}"\n'
            'run_civitai_downloads "$PERSIST_ROOT"\n'
            'stage_rc=$?\n'
            'printf "STAGE_RC=%s\\n" "$stage_rc"\n'
        )
        env = dict(os.environ)
        for name in (
            "CIVITAI_TOKEN",
            "civitai_token",
            "CIVITAI_API_KEY",
            "CIVITAI_LORAS",
            "CIVITAI_CHECKPOINTS",
            "LORAS_IDS_TO_DOWNLOAD",
            "CHECKPOINT_IDS_TO_DOWNLOAD",
            "SDXL_MODEL_IDS_TO_DOWNLOAD",
        ):
            env.pop(name, None)
        env.update(
            {
                "PATH": f"{bin_dir}:{env.get('PATH', '')}",
                "RUNTIME_DIR": str(runtime),
                "PERSIST_ROOT": str(persistent),
                "CIVITAI_PYTHON": python_path,
                "CIVITAI_LORAS": loras,
                "CIVITAI_CHECKPOINTS": checkpoints,
                "EXPECTED_TOKEN": SECRET,
                "FAKE_DEP_READY": str(dep_ready),
                "CAPTURE_FILE": str(capture),
                "REPORT_FILE": str(report),
                "FAKE_PIP_FAIL": "1" if pip_fails else "0",
            }
        )
        if token_name:
            env[token_name] = SECRET
        result = subprocess.run(
            [BASH, str(script)],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = result.stdout + result.stderr
        stage_rc = -1
        for line in result.stdout.splitlines():
            if line.startswith("STAGE_RC="):
                stage_rc = int(line.split("=", 1)[1])
        return Run(
            stage_rc,
            output,
            capture.read_text().splitlines() if capture.exists() else [],
            report.read_text() if report.exists() else "",
        )


def parse_capture(line: str) -> dict[str, str]:
    return dict(field.split("=", 1) for field in line.split("\t"))


def test_runtime_snapshot_beats_a_legacy_path_copy_and_preserves_identifiers():
    identifiers = (
        "model:2834417",
        "version:3268303",
        "urn:air:minimaxh3:lora:civitai:2834417@3268303+3152083",
        "https://civitai.com/api/download/models/3268303?fileId=3152083",
    )
    run = run_stage(loras=",".join(identifiers))
    ok(run.rc == 0, run.output)
    ok(not any("LEGACY_USED" in line for line in run.capture), run.capture)
    parsed = [parse_capture(line) for line in run.downloads]
    ok({item["ID"] for item in parsed} == set(identifiers), parsed)
    expected_path = str(REPO / "vendor" / "civitai_downloader" / "download_with_aria.py")
    ok(all(item["DOWNLOADER"] == expected_path for item in parsed), parsed)
    ok(all(item["TOKEN_OK"] == "1" for item in parsed), parsed)
    ok(all(item["API_KEY_SET"] == "0" for item in parsed), parsed)
    ok(SECRET not in run.output and SECRET not in "\n".join(run.capture), run.output)


def test_parallel_children_are_joined_and_failures_are_reported():
    run = run_stage(loras="version:good", checkpoints="version:fail")
    ok(run.rc == 1, run.output)
    ok(len(run.downloads) == 2, run.capture)
    ok("1 of 2 CivitAI downloads failed" in run.output, run.output)
    ok("warn\tCivitAI checkpoints download failed" in run.report, run.report)


def test_missing_and_invalid_runtime_state_never_fall_back():
    with tempfile.TemporaryDirectory() as temp_name:
        root = Path(temp_name)
        missing = root / "missing-runtime"
        (missing / "src").mkdir(parents=True)
        shutil.copy2(ENV_HELPER, missing / "src" / "civitai_env.sh")
        run = run_stage(runtime=missing, fake_python=False)
        ok(run.rc == 1, run.output)
        ok("missing or invalid" in run.output, run.output)
        ok(not run.downloads and "LEGACY_USED" not in "\n".join(run.capture), run.capture)

        invalid = make_runtime(root / "invalid", valid_vendor=False)
        run = run_stage(runtime=invalid, fake_python=False)
        ok(run.rc == 1, run.output)
        ok("missing or invalid" in run.output, run.output)
        ok(not run.downloads and "LEGACY_USED" not in "\n".join(run.capture), run.capture)


def test_dependency_is_installed_from_the_runtime_manifest_before_download():
    run = run_stage(dependency_ready=False)
    ok(run.rc == 0, run.output)
    installs = [line for line in run.capture if line.startswith("PIP_INSTALL=")]
    ok(len(installs) == 1, run.capture)
    ok(str(REPO / "vendor" / "civitai_downloader" / "requirements.txt") in installs[0], installs)
    ok(len(run.downloads) == 1, run.capture)


def test_dependency_failure_skips_requested_downloads_safely():
    run = run_stage(dependency_ready=False, pip_fails=True)
    ok(run.rc == 1, run.output)
    ok(not run.downloads, run.capture)
    ok("dependency install failed" in run.output, run.output)
    ok("warn\tCivitAI downloader dependency install failed" in run.report, run.report)


def test_no_requested_ids_performs_no_dependency_or_download_work():
    run = run_stage(loras="", checkpoints="", dependency_ready=False)
    ok(run.rc == 0, run.output)
    ok(not run.capture, run.capture)
    ok(run.output.count("Skipping CivitAI") == 2, run.output)


def test_runtime_revision_path_changes_with_the_selected_checkout():
    with tempfile.TemporaryDirectory() as temp_name:
        root = Path(temp_name)
        first = make_runtime(root / "first")
        rolled_back = make_runtime(root / "rolled-back")
        first_run = run_stage(runtime=first)
        rollback_run = run_stage(runtime=rolled_back)
        first_path = parse_capture(first_run.downloads[0])["DOWNLOADER"]
        rollback_path = parse_capture(rollback_run.downloads[0])["DOWNLOADER"]
        ok(first_path.startswith(str(first)), first_path)
        ok(rollback_path.startswith(str(rolled_back)), rollback_path)
        ok(first_path != rollback_path, (first_path, rollback_path))


def test_all_four_template_entrypoints_select_the_shared_runtime():
    for name, (repo, container_dir) in FAMILIES.items():
        if not repo.exists():
            SKIPPED.append(f"{name}: sibling checkout absent")
            continue
        pins = (repo / "pins.json").read_text(encoding="utf-8")
        entrypoint = (repo / "src" / "start_script.sh").read_text(encoding="utf-8")
        ok('"runtime_ref": "stable"' in pins, f"{name} does not pin stable")
        ok(
            f"exec bash /comfyui-runtime/src/start.sh {container_dir}" in entrypoint,
            f"{name} does not exec the shared runtime",
        )
        ok("download_with_aria" not in entrypoint, f"{name} bypasses runtime ownership")


def test_start_and_future_base_builds_have_no_legacy_authority():
    start = START.read_text(encoding="utf-8")
    base = BASE_DOCKERFILE.read_text(encoding="utf-8")
    ok(start.count('source "$RUNTIME_DIR/src/civitai_downloads.sh"') == 1, "runtime helper not sourced exactly once")
    ok(start.count("prepare_civitai_downloads_if_requested") == 1, "runtime dependency preparation not called exactly once")
    ok(start.count('run_civitai_downloads "$PERSIST_ROOT"') == 1, "runtime stage not called exactly once")
    ok(
        start.index("prepare_civitai_downloads_if_requested")
        < start.index("# --- sage install + probe: begin"),
        "runtime dependency preparation can race a background pip install",
    )
    ok("git clone \"https://github.com/Hearmeman24/CivitAI_Downloader" not in start, "boot still clones the downloader")
    ok("CivitAI_Downloader.git" not in base, "future base builds still clone the downloader")
    ok("/usr/local/bin/download_with_aria.py" not in base, "future base builds still bake the downloader")
    ok("Renaming $(basename" not in start, "legacy ZIP-to-safetensors renaming still bypasses downloader validation")


def main():
    test_runtime_snapshot_beats_a_legacy_path_copy_and_preserves_identifiers()
    test_parallel_children_are_joined_and_failures_are_reported()
    test_missing_and_invalid_runtime_state_never_fall_back()
    test_dependency_is_installed_from_the_runtime_manifest_before_download()
    test_dependency_failure_skips_requested_downloads_safely()
    test_no_requested_ids_performs_no_dependency_or_download_work()
    test_runtime_revision_path_changes_with_the_selected_checkout()
    test_all_four_template_entrypoints_select_the_shared_runtime()
    test_start_and_future_base_builds_have_no_legacy_authority()
    print(f"civitai runtime self-test: all good ({CHECKS} assertions)")
    for skipped in SKIPPED:
        print(f"SKIP: {skipped}")


if __name__ == "__main__":
    main()
