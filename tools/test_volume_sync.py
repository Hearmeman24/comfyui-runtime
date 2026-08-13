#!/usr/bin/env python3
"""Self-check for the background stage->volume copier.

Runs fully offline: huggingface_hub is stubbed before volume_sync imports the
download manager for fmt_bytes/parse_manifest. No network, no GPU.

Run: python3 tools/test_volume_sync.py
"""
import contextlib
import io
import os
import sys
import tempfile
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

hf_stub = types.ModuleType("huggingface_hub")
hf_stub.hf_hub_download = lambda **kw: None
hf_stub.HfFileSystem = object
sys.modules["huggingface_hub"] = hf_stub

sys.path.insert(0, str(REPO / "src"))
import volume_sync as vs  # noqa: E402

URL = "https://huggingface.co/org/repo/resolve/main/m.safetensors"


def run(manifest) -> tuple[int, str]:
    argv_old = sys.argv
    sys.argv = ["volume_sync.py", str(manifest)]
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            rc = vs.main()
    finally:
        sys.argv = argv_old
    return rc, buf.getvalue()


def staged(tmp, name, size=4096):
    """A model as the download manager leaves it: staged file + symlink."""
    stage = tmp / "hf_stage"
    stage.mkdir(parents=True, exist_ok=True)
    models = tmp / "models" / "diffusion_models"
    models.mkdir(parents=True, exist_ok=True)
    target = stage / name
    target.write_bytes(b"\xab" * size)
    dest = models / name
    dest.symlink_to(target)
    return dest, target


def manifest_for(tmp, *dests):
    man = tmp / "m.tsv"
    man.write_text("".join(
        f"https://huggingface.co/org/repo/resolve/main/{d.name}\t{d}\t0.001\n"
        for d in dests))
    return man


def test_usage_and_missing_manifest(tmp):
    argv_old = sys.argv
    try:
        sys.argv = ["volume_sync.py"]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            assert vs.main() == 2, "wrong argc must exit 2"
    finally:
        sys.argv = argv_old
    rc, out = run(tmp / "nope.tsv")
    assert rc == 0 and "nothing to do" in out, out
    print("ok: exit 2 on bad argc, exit 0 on a missing manifest")


def test_symlink_becomes_real_file(tmp):
    dest, target = staged(tmp, "m.safetensors", 8192)
    man = manifest_for(tmp, dest)
    rc, out = run(man)

    assert rc == 0, f"expected 0, got {rc}\n{out}"
    assert not dest.is_symlink(), "the symlink must be replaced by a real file"
    assert dest.is_file() and dest.stat().st_size == 8192, "content must survive"
    assert dest.read_bytes() == b"\xab" * 8192, "bytes must match exactly"
    assert not target.exists(), "the staged copy must be freed"
    assert not dest.with_name(dest.name + ".partial").exists(), \
        ".partial must not be left behind"
    assert "landed m.safetensors" in out
    assert "Safe to restart or terminate." in out, \
        "the completion line is the user's signal that the pod is durable"
    print("ok: a staged symlink becomes a real file and the stage is freed")


def test_idempotent_and_noop_on_real_files(tmp):
    dest, _ = staged(tmp, "m.safetensors")
    man = manifest_for(tmp, dest)
    rc, _ = run(man)
    assert rc == 0
    before = dest.read_bytes()

    rc, out = run(man)  # second run, nothing staged any more
    assert rc == 0, out
    assert "nothing to copy" in out, out
    assert dest.is_file() and dest.read_bytes() == before, \
        "a second run must not disturb an already-landed model"
    print("ok: a second run is a no-op and does not touch landed models")


def test_dangling_symlink_is_left_alone(tmp):
    models = tmp / "models" / "diffusion_models"
    models.mkdir(parents=True)
    dead = models / "dead.safetensors"
    dead.symlink_to(tmp / "hf_stage" / "gone.safetensors")
    man = manifest_for(tmp, dead)

    rc, out = run(man)
    assert rc == 0, out
    assert "nothing to copy" in out, out
    assert dead.is_symlink(), \
        "reclaiming a dangling link is the download manager's job, not ours"
    print("ok: a dangling symlink is skipped, not touched")


def test_one_failure_does_not_abort_the_run(tmp):
    ok1, _ = staged(tmp, "a.safetensors")
    bad, bad_target = staged(tmp, "b.safetensors")
    ok2, _ = staged(tmp, "c.safetensors")
    man = manifest_for(tmp, ok1, bad, ok2)

    real_copy = vs.shutil.copy2

    def flaky(src, dst):
        if Path(src).name == "b.safetensors":
            raise OSError(28, "No space left on device")
        return real_copy(src, dst)

    vs.shutil.copy2 = flaky
    try:
        rc, out = run(man)
    finally:
        vs.shutil.copy2 = real_copy

    assert rc == 0, "a per-file failure must not become a nonzero exit"
    assert ok1.is_file() and not ok1.is_symlink(), "a.safetensors must land"
    assert ok2.is_file() and not ok2.is_symlink(), \
        "c.safetensors must land: a failure before it must not abort the run"
    assert bad.is_symlink() and bad.exists(), \
        "the failed entry must keep a working symlink so ComfyUI still loads it"
    assert bad_target.is_file(), "its staged copy must not be freed"
    assert not bad.with_name(bad.name + ".partial").exists(), \
        "a failed copy must not leave a .partial at the load path"
    assert "FAILED b.safetensors" in out and "1 failure" in out
    assert "Safe to restart" not in out, \
        "the safe-to-restart line must NOT print when something is still staged"
    print("ok: one failure is logged, skipped, and does not claim safe-to-restart")


def test_short_copy_is_caught(tmp):
    dest, target = staged(tmp, "m.safetensors", 8192)
    man = manifest_for(tmp, dest)

    def truncating(src, dst):
        Path(dst).write_bytes(Path(src).read_bytes()[:10])

    real_copy = vs.shutil.copy2
    vs.shutil.copy2 = truncating
    try:
        rc, out = run(man)
    finally:
        vs.shutil.copy2 = real_copy

    assert rc == 0, out
    assert dest.is_symlink() and dest.exists(), \
        "a short copy must never be published at the load path"
    assert target.is_file(), "the staged original must survive a short copy"
    assert "FAILED" in out and "expected 8192" in out, out
    print("ok: a short copy is detected and never reaches the load path")


def main() -> int:
    for test in (test_usage_and_missing_manifest, test_symlink_becomes_real_file,
                 test_idempotent_and_noop_on_real_files,
                 test_dangling_symlink_is_left_alone,
                 test_one_failure_does_not_abort_the_run,
                 test_short_copy_is_caught):
        with tempfile.TemporaryDirectory() as tmp:
            test(Path(tmp))
    print("all volume_sync self-tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
