#!/usr/bin/env python3
"""Self-check for the unified download manager (slice A).

Runs fully offline: huggingface_hub is replaced with a stub BEFORE the manager
imports it, and the subprocess downloaders (aria2c, gdown) are faked. No
network, no GPU, no third-party packages.

Run: python3 tools/test_download_manager.py
"""
import contextlib
import io
import os
import shutil
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Stub huggingface_hub before the manager imports it, so the test needs
# neither the package nor the network.
# ---------------------------------------------------------------------------
HF_CALLS: list[dict] = []
HF_BEHAVIOR = {"size_bytes": 32 * 1024, "block": None}


def fake_hf_hub_download(repo_id, filename, revision, local_dir, token=None):
    HF_CALLS.append({"repo_id": repo_id, "filename": filename,
                     "revision": revision, "local_dir": local_dir,
                     "token": token})
    if HF_BEHAVIOR["block"] is not None:
        HF_BEHAVIOR["block"].wait()
    out = Path(local_dir) / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"\0" * HF_BEHAVIOR["size_bytes"])
    return str(out)


class FakeHfFileSystem:
    def __init__(self, token=None):
        self.token = token

    def info(self, path, revision=None):
        return {"size": HF_BEHAVIOR["size_bytes"]}


hf_stub = types.ModuleType("huggingface_hub")
hf_stub.hf_hub_download = fake_hf_hub_download
hf_stub.HfFileSystem = FakeHfFileSystem
sys.modules["huggingface_hub"] = hf_stub

sys.path.insert(0, str(REPO / "src"))
import hf_download_manager as dm  # noqa: E402

dm.SNAPSHOT_INTERVAL = 0.05  # keep the pool loop fast under test

# ---------------------------------------------------------------------------
# Stub the subprocess downloaders (aria2c, gdown).
# ---------------------------------------------------------------------------
SUB_CALLS: list[list] = []
SUB_SIZE = [32 * 1024]


def fake_subprocess_run(cmd, check=True):
    SUB_CALLS.append(list(cmd))
    if cmd[0] == "aria2c":
        out = Path(cmd[cmd.index("-d") + 1]) / cmd[cmd.index("-o") + 1]
    elif cmd[0] == "gdown":
        out = Path(cmd[cmd.index("-O") + 1])
    else:
        raise AssertionError(f"unexpected subprocess: {cmd}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"\0" * SUB_SIZE[0])
    return types.SimpleNamespace(returncode=0)


dm.subprocess = types.SimpleNamespace(run=fake_subprocess_run)


def reset_calls():
    HF_CALLS.clear()
    SUB_CALLS.clear()


def run_main(manifest_path) -> tuple[int, str]:
    argv_old = sys.argv
    sys.argv = ["hf_download_manager.py", str(manifest_path)]
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            rc = dm.main()
    finally:
        sys.argv = argv_old
    return rc, buf.getvalue()


HF_URL = "https://huggingface.co/org/repo/resolve/main/hfmodel.safetensors"


def test_env_timeouts():
    """CONTRACTS.md §2: HF timeout defaults are set before the hub import."""
    assert os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] == "30"
    assert os.environ["HF_HUB_ETAG_TIMEOUT"] == "15"
    print("ok: HF timeout env defaults")


def test_parse_manifest(tmp):
    man = tmp / "m.tsv"
    man.write_text(
        "# comment line\n"
        "\n"
        "this line has no tab\n"
        f"{HF_URL}\t/x/a.safetensors\n"
        "https://huggingface.co/org/repo/resolve/main/sub/dir/model.bin?download=true\t/x/b.bin\t0.5\n"
        "https://example.com/files/thing.safetensors\t/x/c.safetensors\t2\n"
        "https://drive.google.com/file/d/ABC123/view\t/x/d.safetensors\n"
        "https://huggingface.co/org/repo/resolve/v1.0/e.pt\t/x/e.pt\t1\textra-field\n"
    )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        jobs = dm.parse_manifest(man)
    out = buf.getvalue()

    assert len(jobs) == 5, f"expected 5 jobs, got {len(jobs)}"
    a, b, c, d, e = jobs

    assert (a.kind, a.repo_id, a.revision, a.filename) == \
        ("hf", "org/repo", "main", "hfmodel.safetensors")
    assert a.min_size_mb == 10.0, "two-field line must default to the 10 MB floor"

    assert b.kind == "hf" and b.filename == "sub/dir/model.bin", \
        "query string must be stripped from the HF file path"
    assert b.min_size_mb == 0.5, "per-entry min_size_mb must be honoured (float)"

    assert c.kind == "direct", "a non-HF http URL must route to aria2c, not be dropped"
    assert c.min_size_mb == 2.0

    assert d.kind == "gdrive", "a Google Drive URL must route to gdown"

    assert e.min_size_mb == 1.0
    assert "extra" in out.lower(), "fields beyond the third must warn"
    assert "no tab" in out, "a tab-less line must be warned about"
    assert "skip non-HF" not in out, "wan's silent non-HF drop must be gone"
    print("ok: manifest parsing and URL routing")


def test_exit_codes(tmp):
    argv_old = sys.argv
    buf = io.StringIO()
    try:
        sys.argv = ["hf_download_manager.py"]
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            assert dm.main() == 2, "wrong argc must exit 2"
    finally:
        sys.argv = argv_old

    rc, _ = run_main(tmp / "does_not_exist.tsv")
    assert rc == 0, "missing manifest must exit 0"

    empty = tmp / "empty.tsv"
    empty.write_text("# only a comment\n\n")
    rc, _ = run_main(empty)
    assert rc == 0, "empty manifest must exit 0"
    print("ok: exit codes 2 / 0 / 0")


def test_happy_path(tmp):
    reset_calls()
    dm.LOCAL_STAGE = tmp / "hf_stage"
    os.environ["HF_TOKEN"] = "testtok"
    HF_BEHAVIOR["size_bytes"] = 32 * 1024
    SUB_SIZE[0] = 32 * 1024

    dests = {
        "hf": tmp / "models" / "diffusion_models" / "hfmodel.safetensors",
        "direct": tmp / "models" / "upscale_models" / "thing.bin",
        "gdrive": tmp / "models" / "loras" / "gd.safetensors",
    }
    man = tmp / "m.tsv"
    man.write_text(
        f"{HF_URL}\t{dests['hf']}\t0.01\n"
        f"https://example.com/thing.bin\t{dests['direct']}\t0.01\n"
        f"https://drive.google.com/uc?id=XYZ\t{dests['gdrive']}\t0.01\n"
    )
    rc, out = run_main(man)
    assert rc == 0, f"expected 0, got {rc}\n{out}"
    for name, dest in dests.items():
        assert dest.is_file() and dest.stat().st_size == 32 * 1024, \
            f"{name} dest missing or wrong size"
        assert not dest.with_name(dest.name + ".partial").exists(), \
            f"{name} left a .partial behind"
    assert "all 3 downloads complete" in out

    assert len(HF_CALLS) == 1 and HF_CALLS[0]["token"] == "testtok", \
        "HF_TOKEN must be passed explicitly to hf_hub_download"
    assert dm.LOCAL_STAGE.as_posix() in HF_CALLS[0]["local_dir"], \
        "HF download must stage on the local disk, not the volume"

    aria = [c for c in SUB_CALLS if c[0] == "aria2c"]
    assert len(aria) == 1
    assert aria[0][:9] == ["aria2c", "-x", "16", "-s", "16", "-k", "1M",
                           "--continue=true", "--summary-interval=0"], \
        f"aria2c flags diverged from qwen's frozen set: {aria[0]}"
    assert "--console-log-level=warn" in aria[0]
    gdown = [c for c in SUB_CALLS if c[0] == "gdown"]
    assert len(gdown) == 1 and "-O" in gdown[0]

    leftovers = [p for p in dm.LOCAL_STAGE.rglob("*") if p.is_file()] \
        if dm.LOCAL_STAGE.exists() else []
    assert not leftovers, f"staging not cleaned: {leftovers}"
    del os.environ["HF_TOKEN"]
    print("ok: happy path (hf + aria2c + gdown), atomic handoff, staging cleaned")


def test_skip_and_refetch(tmp):
    reset_calls()
    dm.LOCAL_STAGE = tmp / "hf_stage"
    dest = tmp / "models" / "vae" / "small.safetensors"
    dest.parent.mkdir(parents=True)
    man = tmp / "m.tsv"
    man.write_text(f"{HF_URL}\t{dest}\t0.5\n")

    # At/above its floor: counted done, never re-downloaded.
    dest.write_bytes(b"\0" * (1024 * 1024))
    stale = dest.with_name(dest.name + ".partial")
    stale.write_bytes(b"junk")
    rc, out = run_main(man)
    assert rc == 0
    assert not HF_CALLS, "an at-floor file must be skipped, not re-downloaded"
    assert "skip" in out.lower()
    assert not stale.exists(), "a stale .partial must be removed at startup"

    # Below its floor: deleted and refetched.
    reset_calls()
    dest.write_bytes(b"\0" * 1024)
    HF_BEHAVIOR["size_bytes"] = 1024 * 1024
    rc, out = run_main(man)
    assert rc == 0
    assert len(HF_CALLS) == 1, "a sub-floor file must be deleted and refetched"
    assert dest.stat().st_size == 1024 * 1024
    print("ok: skip-if-present honours the per-entry floor; sub-floor refetches")


def test_floor_failure(tmp):
    reset_calls()
    dm.LOCAL_STAGE = tmp / "hf_stage"
    dest = tmp / "models" / "checkpoints" / "big.safetensors"
    man = tmp / "m.tsv"
    man.write_text(f"{HF_URL}\t{dest}\t50\n")
    HF_BEHAVIOR["size_bytes"] = 32 * 1024  # far below the 50 MB floor
    rc, out = run_main(man)
    assert rc == 1, f"a sub-floor download must fail the run, got {rc}"
    assert not dest.exists(), "a sub-floor download must never land at dest"
    assert "1 failures" in out
    print("ok: post-download floor verify fails the entry and exits 1")


def test_stage_fallback(tmp):
    dm.LOCAL_STAGE = tmp / "hf_stage"
    job = dm.Job(url="u", dest=tmp / "models" / "big.safetensors", kind="hf")
    job.total_bytes = 10 * 1024 ** 3
    real_du = dm.shutil.disk_usage
    buf = io.StringIO()
    try:
        dm.shutil.disk_usage = lambda p: types.SimpleNamespace(free=1024)
        with contextlib.redirect_stdout(buf):
            stage = dm.pick_stage_dir(job)
        assert stage == job.dest.parent / ".hf_stage" / job.dest.name, \
            "no local headroom must fall back to volume-side staging"

        dm.shutil.disk_usage = lambda p: types.SimpleNamespace(free=10 ** 15)
        with contextlib.redirect_stdout(buf):
            stage = dm.pick_stage_dir(job)
        assert stage == dm.LOCAL_STAGE / job.dest.name, \
            "with 2.5x headroom the local disk must be used"
    finally:
        dm.shutil.disk_usage = real_du
    print("ok: 2.5x headroom check picks local disk vs volume fallback")


def test_status_file(tmp):
    """HF_STATUS_FILE (boot-report feed, EXECUTION.md N1): every entry's
    final status and error land in the JSON, success and failure alike."""
    import json
    reset_calls()
    dm.LOCAL_STAGE = tmp / "hf_stage"
    status_path = tmp / "hf_status.json"
    os.environ["HF_STATUS_FILE"] = str(status_path)
    ok_dest = tmp / "models" / "vae" / "ok.safetensors"
    bad_dest = tmp / "models" / "checkpoints" / "bad.safetensors"
    man = tmp / "m.tsv"
    man.write_text(
        f"https://huggingface.co/org/repo/resolve/main/ok.safetensors\t{ok_dest}\t0.01\n"
        f"https://huggingface.co/org/repo/resolve/main/bad.safetensors\t{bad_dest}\t50\n"
    )
    HF_BEHAVIOR["size_bytes"] = 32 * 1024  # below bad's 50 MB floor
    try:
        rc, _ = run_main(man)
    finally:
        del os.environ["HF_STATUS_FILE"]
    assert rc == 1
    status = json.loads(status_path.read_text())
    assert status["ok.safetensors"]["status"] == "done", status
    assert status["ok.safetensors"]["error"] is None, status
    assert status["bad.safetensors"]["status"] == "failed", status
    assert "floor" in status["bad.safetensors"]["error"], status
    print("ok: HF_STATUS_FILE records per-entry status and error")


class WatchdogExit(Exception):
    pass


def test_watchdog(tmp):
    reset_calls()
    dm.LOCAL_STAGE = tmp / "hf_stage"
    dm.STALL_SECS = 0.15
    man = tmp / "m.tsv"
    man.write_text(f"{HF_URL}\t{tmp / 'models' / 'stuck.safetensors'}\t0.01\n")

    HF_BEHAVIOR["block"] = threading.Event()

    def fake_exit(code):
        raise WatchdogExit(code)

    real_exit = os._exit
    os._exit = fake_exit
    argv_old = sys.argv
    sys.argv = ["hf_download_manager.py", str(man)]
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            try:
                dm.main()
                raise AssertionError("watchdog did not fire on a stalled download")
            except WatchdogExit as e:
                assert e.args[0] == 1, "watchdog must exit 1"
    finally:
        os._exit = real_exit
        sys.argv = argv_old
        HF_BEHAVIOR["block"].set()
        HF_BEHAVIOR["block"] = None
        dm.STALL_SECS = 300
        time.sleep(0.3)  # let the released worker thread drain
    out = buf.getvalue()
    assert "abandoning" in out, f"watchdog must print its one-line abandon: {out}"
    print("ok: stall watchdog abandons and exits 1")


def test_watchdog_deadline(tmp):
    """D14: DEADLINE_SECS must trip on total elapsed time even while byte
    progress is continuous, i.e. when the stall condition can never fire."""
    import json
    reset_calls()
    dm.LOCAL_STAGE = tmp / "hf_stage"
    dm.STALL_SECS = 300     # stall must stay unreachable in this test
    dm.DEADLINE_SECS = 0.3
    status_path = tmp / "hf_status.json"
    os.environ["HF_STATUS_FILE"] = str(status_path)
    dest = tmp / "models" / "slow.safetensors"
    man = tmp / "m.tsv"
    man.write_text(f"{HF_URL}\t{dest}\t0.01\n")

    block = threading.Event()
    HF_BEHAVIOR["block"] = block
    stop_feeding = threading.Event()

    def feed_progress():
        # Grow a file inside the worker's stage dir so global byte progress
        # never pauses and the stall condition cannot be what fires. Give up
        # after 3 s and release the blocked download, so a broken deadline
        # branch fails the test instead of spinning until STALL_SECS.
        stage_file = dm.LOCAL_STAGE / dest.name / "chunk"
        n = 0
        give_up_at = time.time() + 3
        while not stop_feeding.is_set() and time.time() < give_up_at:
            n += 4096
            try:
                stage_file.parent.mkdir(parents=True, exist_ok=True)
                stage_file.write_bytes(b"\0" * n)
            except OSError:
                pass
            time.sleep(0.02)
        block.set()

    feeder = threading.Thread(target=feed_progress, daemon=True)

    def fake_exit(code):
        raise WatchdogExit(code)

    real_exit = os._exit
    os._exit = fake_exit
    argv_old = sys.argv
    sys.argv = ["hf_download_manager.py", str(man)]
    buf = io.StringIO()
    feeder.start()
    try:
        with contextlib.redirect_stdout(buf):
            try:
                dm.main()
                raise AssertionError(
                    "deadline watchdog did not fire on a progressing download")
            except WatchdogExit as e:
                assert e.args[0] == 1, "watchdog must exit 1"
    finally:
        os._exit = real_exit
        sys.argv = argv_old
        stop_feeding.set()
        block.set()
        HF_BEHAVIOR["block"] = None
        feeder.join()
        dm.STALL_SECS = 300
        dm.DEADLINE_SECS = 3600
        del os.environ["HF_STATUS_FILE"]
        time.sleep(0.3)  # let the released worker thread drain
    out = buf.getvalue()
    assert "abandoning" in out, f"watchdog must print its one-line abandon: {out}"
    assert "deadline exceeded" in out, f"the trip must name the deadline: {out}"
    assert "no progress for" not in out, \
        f"the stall condition must not be what fired: {out}"
    status = json.loads(status_path.read_text())
    assert status[dest.name]["status"] == "failed", status
    assert status[dest.name]["error"] == "deadline exceeded", \
        f"failure reason must be the deadline, not the stall: {status}"
    print("ok: deadline watchdog abandons a progressing-but-overdue phase and exits 1")


def test_handoff_counts_as_progress(tmp):
    """A slow stage->volume handoff must not read as a stall.

    The real failure this reproduces (minimax pod, 2026-08-13): three 20 GB
    models finished downloading, then spent >5 min in shutil.move to the
    network volume. stage_bytes stops changing the moment the download
    returns, so the watchdog saw zero progress and os._exit(1)'d mid-copy,
    losing 64 GB and stranding the queued taeh3 behind the held pool slots.

    Here the move takes ~8x STALL_SECS while the .partial grows steadily.
    Before the fix the watchdog fires; after it, the copy is progress.
    """
    reset_calls()
    dm.LOCAL_STAGE = tmp / "hf_stage"
    dm.STALL_SECS = 0.3
    HF_BEHAVIOR["size_bytes"] = 64 * 1024
    dest = tmp / "models" / "diffusion_models" / "slow.safetensors"
    man = tmp / "m.tsv"
    man.write_text(f"{HF_URL}\t{dest}\t0.01\n")

    real_move = shutil.move

    def slow_move(src, dst):
        """Copy in 8 chunks over ~2.4s, growing dst the way a real cross-FS
        move does. Only the manifest's own file is slowed."""
        data = Path(src).read_bytes()
        step = max(len(data) // 8, 1)
        with open(dst, "wb") as fh:
            for off in range(0, len(data), step):
                fh.write(data[off:off + step])
                fh.flush()
                os.fsync(fh.fileno())
                time.sleep(0.3)
        Path(src).unlink()

    dm.shutil.move = slow_move

    def fake_exit(code):
        raise WatchdogExit(code)

    real_exit = os._exit
    os._exit = fake_exit
    try:
        rc, out = run_main(man)
    except WatchdogExit as e:
        raise AssertionError(
            f"watchdog fired (exit {e.args[0]}) during an active handoff; "
            "the copy was making progress on the volume") from None
    finally:
        os._exit = real_exit
        dm.shutil.move = real_move
        dm.STALL_SECS = 300

    assert rc == 0, f"expected 0, got {rc}\n{out}"
    assert dest.is_file() and dest.stat().st_size == 64 * 1024, \
        f"handoff did not land the file\n{out}"
    assert not dest.with_name(dest.name + ".partial").exists(), \
        "handoff left a .partial behind"
    assert "handoff" in out, \
        f"the snapshot must name the handoff phase, not show a frozen 100%:\n{out}"
    print("ok: a slow stage->volume handoff counts as progress and is labelled")


def test_orphan_partial_sweep(tmp):
    """A .partial whose model is no longer in the manifest must still go.

    Customer report 2026-08-13: leftovers on the volume. The old sweep only
    looked up <dest>.partial for entries in THIS manifest, so a model since
    flag-disabled or renamed in the registry could never be reclaimed.
    """
    reset_calls()
    dm.LOCAL_STAGE = tmp / "hf_stage"
    HF_BEHAVIOR["size_bytes"] = 32 * 1024
    models = tmp / "models" / "diffusion_models"
    models.mkdir(parents=True)
    dest = models / "hfmodel.safetensors"

    orphan = models / "some_disabled_model.safetensors.partial"
    orphan.write_bytes(b"\0" * 4096)
    own = dest.with_name(dest.name + ".partial")
    own.write_bytes(b"\0" * 4096)
    keeper = models / "unrelated.safetensors"
    keeper.write_bytes(b"\0" * 4096)

    man = tmp / "m.tsv"
    man.write_text(f"{HF_URL}\t{dest}\t0.01\n")
    rc, out = run_main(man)

    assert rc == 0, f"expected 0, got {rc}\n{out}"
    assert not orphan.exists(), \
        "a .partial outside the manifest survived the sweep"
    assert not own.exists(), "the manifest's own .partial survived"
    assert keeper.is_file(), \
        "the sweep deleted a real model; it must only ever remove *.partial"
    print("ok: orphan .partial files are swept, real files are not")


def main() -> int:
    test_env_timeouts()
    for test in (test_parse_manifest, test_exit_codes, test_happy_path,
                 test_skip_and_refetch, test_floor_failure,
                 test_stage_fallback, test_status_file, test_watchdog,
                 test_watchdog_deadline, test_handoff_counts_as_progress,
                 test_orphan_partial_sweep):
        with tempfile.TemporaryDirectory() as tmp:
            test(Path(tmp))
    print("all download manager self-tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
