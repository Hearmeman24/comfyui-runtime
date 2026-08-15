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
import subprocess
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
SUB_FAIL = {"exit_code": 0}  # non-zero makes every child downloader fail


def fake_subprocess_run(cmd, check=False):
    # check defaults to False, exactly like subprocess.run: a caller that opts
    # into check=True gets the real CalledProcessError, whose __str__ carries
    # the whole argv including the URL. That is the leak test_error_redaction
    # guards against.
    SUB_CALLS.append(list(cmd))
    if SUB_FAIL["exit_code"]:
        if check:
            raise subprocess.CalledProcessError(SUB_FAIL["exit_code"], cmd)
        return types.SimpleNamespace(returncode=SUB_FAIL["exit_code"])
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

# ---------------------------------------------------------------------------
# Stub the ranged-GET size probe (non-HF URLs). No network.
# ---------------------------------------------------------------------------
PROBE_CALLS: list[dict] = []
PROBE = {"size": 32 * 1024, "status": 206, "raise": None}


class FakeResponse:
    def __init__(self, headers, status):
        self.headers = headers
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def fake_urlopen(req, timeout=None):
    PROBE_CALLS.append({"url": req.full_url, "method": req.get_method(),
                        "range": req.headers.get("Range"), "timeout": timeout})
    if PROBE["raise"] is not None:
        raise PROBE["raise"]
    if PROBE["status"] == 206:
        return FakeResponse({"Content-Range": f"bytes 0-0/{PROBE['size']}"}, 206)
    return FakeResponse({"Content-Length": str(PROBE["size"])}, 200)


dm.urlopen = fake_urlopen


def reset_calls():
    HF_CALLS.clear()
    SUB_CALLS.clear()
    PROBE_CALLS.clear()
    SUB_FAIL["exit_code"] = 0
    PROBE.update({"size": 32 * 1024, "status": 206, "raise": None})
    dm.reset_reservations()


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

    # The aria2c entry is sized by a ranged GET, never a HEAD: a presigned R2
    # URL is signed for GET and 403s a HEAD.
    assert len(PROBE_CALLS) == 1, f"expected one size probe, got {PROBE_CALLS}"
    assert PROBE_CALLS[0]["url"] == "https://example.com/thing.bin"
    assert PROBE_CALLS[0]["method"] == "GET"
    assert PROBE_CALLS[0]["range"] == "bytes=0-0"
    assert PROBE_CALLS[0]["timeout"] == dm.SIZE_PROBE_TIMEOUT, \
        "the probe must be bounded so a dead host cannot hold the boot"

    # NVMe-first: the stage is NOT cleaned here. The staged file IS the model
    # until volume_sync.py copies it across and unlinks it. A sized entry is a
    # symlink into the stage at this point.
    for name in ("hf", "direct"):
        dest = dests[name]
        assert dest.is_symlink(), f"{name} dest must be a symlink into the stage"
        assert Path(os.readlink(dest)).is_file(), \
            f"{name} symlink target must still exist in the stage"
    # Drive is not probed (a plain GET returns an HTML confirm page, not the
    # file), so it stays size-unknown and lands the slow, safe way.
    gd = dests["gdrive"]
    assert not gd.is_symlink() and gd.is_file(), \
        "an unsized gdrive entry must stage on the volume, not the local disk"
    del os.environ["HF_TOKEN"]
    print("ok: happy path (hf + aria2c + gdown), atomic symlink publish")


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
    dm.reset_reservations()
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
        dm.reset_reservations()
    print("ok: 2.5x headroom check picks local disk vs volume fallback")


def test_no_volume_disables_local_staging(tmp):
    """A pod with no network volume must not stage locally.

    start.sh sets NETWORK_VOLUME=/ when /workspace is absent (start.sh:105-110),
    which makes PERSIST_ROOT="//ComfyUI" -> /ComfyUI: the SAME container disk
    the stage is on. Staging then copies every model local->local, so the pod
    transiently needs 2x the bytes and volume_sync moves 77 GB from one
    directory to another on one filesystem for no gain. Observed on a real
    minimax pod, 2026-08-15.
    """
    dm.LOCAL_STAGE = tmp / "hf_stage"
    dm.reset_reservations()
    job = dm.Job(url="u", dest=tmp / "models" / "big.safetensors", kind="hf")
    job.total_bytes = 10 * 1024 ** 3
    real_du = dm.shutil.disk_usage
    real_flag = dm.STAGE_LOCAL
    buf = io.StringIO()
    try:
        # Plenty of headroom: the ONLY reason not to stage locally is the flag.
        dm.shutil.disk_usage = lambda p: types.SimpleNamespace(free=10 ** 15)

        dm.STAGE_LOCAL = False
        with contextlib.redirect_stdout(buf):
            stage = dm.pick_stage_dir(job)
        assert stage == job.dest.parent / ".hf_stage" / job.dest.name, \
            f"no volume must stage beside the destination, got {stage}"
        assert job.reserved_bytes == 0, \
            "a skipped local stage must not reserve local budget"
        assert not dm.stage_is_local(stage), \
            "and the handoff must be a same-fs rename, not a symlink + sync"

        dm.STAGE_LOCAL = True
        with contextlib.redirect_stdout(buf):
            stage = dm.pick_stage_dir(job)
        assert stage == dm.LOCAL_STAGE / job.dest.name, \
            "with a volume attached the local disk is still used"
    finally:
        dm.shutil.disk_usage = real_du
        dm.STAGE_LOCAL = real_flag
        dm.reset_reservations()
    print("ok: no network volume disables local staging (and its volume_sync)")


def test_stage_local_flag_reads_the_env(tmp):
    """start.sh communicates 'no volume' by exporting HF_STAGE_LOCAL=0.

    STAGE_LOCAL is resolved at import time, so this needs a fresh interpreter
    per value rather than a monkeypatch. The child stubs huggingface_hub the
    same way this module does: hf_download_manager imports it at module scope
    (:45) and the CI executor has no third-party packages at all, so a bare
    import there is a ModuleNotFoundError, not a test result.
    """
    child = (
        "import sys, types\n"
        "hf = types.ModuleType('huggingface_hub')\n"
        "hf.hf_hub_download = lambda *a, **k: None\n"
        "hf.HfFileSystem = object\n"
        "sys.modules['huggingface_hub'] = hf\n"
        "sys.path.insert(0, %r)\n"
        "import hf_download_manager as m; print(m.STAGE_LOCAL)\n"
        % str(REPO / "src")
    )
    for value, expected in (("0", False), ("false", False), ("FALSE", False),
                            ("1", True), ("", True), (None, True)):
        env = dict(os.environ)
        env.pop("HF_STAGE_LOCAL", None)
        if value is not None:
            env["HF_STAGE_LOCAL"] = value
        r = subprocess.run([sys.executable, "-c", child],
                           env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        got = r.stdout.strip() == "True"
        assert got is expected, \
            f"HF_STAGE_LOCAL={value!r} should give STAGE_LOCAL={expected}, got {got}"
    print("ok: HF_STAGE_LOCAL is read from the environment")


def test_direct_headroom(tmp):
    """BUG 1: the 2.5x guard must apply to aria2c and gdown entries too.

    fetch_sizes only ever ran for kind == "hf", so every direct/gdrive job kept
    total_bytes == 0 and pick_stage_dir's `not job.total_bytes` branch sent it
    to the local disk unconditionally - the exact entries (presigned R2,
    CivitAI) that run to tens of GB, CONTRACTS.md:85-87.
    """
    reset_calls()
    dm.LOCAL_STAGE = tmp / "hf_stage"
    real_du = dm.shutil.disk_usage
    buf = io.StringIO()
    try:
        # 40 GB free, a 25 GB presigned object: 62.5 GB needed, so no.
        dm.shutil.disk_usage = lambda p: types.SimpleNamespace(free=40 * 1024 ** 3)
        direct = dm.Job(url="https://r2.example.com/x.safetensors?X-Amz-Signature=deadbeef",
                        dest=tmp / "models" / "x.safetensors", kind="direct")
        PROBE["size"] = 25 * 1024 ** 3
        with contextlib.redirect_stdout(buf):
            dm.fetch_direct_sizes([direct])
            stage = dm.pick_stage_dir(direct)
        assert direct.total_bytes == 25 * 1024 ** 3, \
            "a direct entry must be sized by the ranged GET"
        assert stage == direct.dest.parent / ".hf_stage" / direct.dest.name, \
            "a 25 GB direct entry must not stage on a 40 GB disk"
        assert "?X-Amz-Signature" not in buf.getvalue(), \
            "the staging decision must not echo a presigned query string"

        # Probe fails (403, DNS, anything): unknown size now means the volume,
        # not a free pass onto the local disk.
        unknown = dm.Job(url="https://r2.example.com/y.safetensors",
                         dest=tmp / "models" / "y.safetensors", kind="direct")
        PROBE["raise"] = OSError("403 forbidden")
        dm.shutil.disk_usage = lambda p: types.SimpleNamespace(free=10 ** 15)
        with contextlib.redirect_stdout(buf):
            dm.fetch_direct_sizes([unknown])
            stage = dm.pick_stage_dir(unknown)
        assert unknown.total_bytes == 0
        assert stage == unknown.dest.parent / ".hf_stage" / unknown.dest.name, \
            "an unsized entry must stage on the volume even with room to spare"

        # A server that ignores Range answers 200; Content-Length is then the
        # whole file, and a small one still takes the fast path.
        PROBE["raise"] = None
        PROBE["status"] = 200
        PROBE["size"] = 4 * 1024 ** 2
        small = dm.Job(url="https://example.com/small.pth",
                       dest=tmp / "models" / "small.pth", kind="direct")
        with contextlib.redirect_stdout(buf):
            dm.fetch_direct_sizes([small])
            stage = dm.pick_stage_dir(small)
        assert small.total_bytes == 4 * 1024 ** 2, "200 must fall back to Content-Length"
        assert stage == dm.LOCAL_STAGE / small.dest.name
    finally:
        dm.shutil.disk_usage = real_du
        dm.reset_reservations()
    print("ok: the 2.5x headroom guard covers direct entries; unsized ones use the volume")


def test_stage_reservation(tmp):
    """BUG 2: concurrent pick_stage_dir calls must not each spend the same disk.

    Deterministic by construction rather than by timing: with a FIXED stubbed
    free space, only one of three identical jobs can fit, so whatever order the
    threads win in, the outcome is exactly 1 local + 2 volume. Without the
    reservation all three read the same disk_usage() and all three go local.
    """
    dm.LOCAL_STAGE = tmp / "hf_stage"
    dm.reset_reservations()
    lock = threading.Lock()
    real_du = dm.shutil.disk_usage
    # 100 GB free, three 30 GB models: 75 GB needed each, so exactly one fits.
    dm.shutil.disk_usage = lambda p: types.SimpleNamespace(free=100 * 1024 ** 3)
    jobs = [dm.Job(url="u", dest=tmp / "models" / f"m{i}.safetensors", kind="hf")
            for i in range(dm.POOL_SIZE)]
    for j in jobs:
        j.total_bytes = 30 * 1024 ** 3
    picks: dict = {}
    ready = threading.Barrier(dm.POOL_SIZE)
    buf = io.StringIO()

    def worker(job):
        ready.wait()
        picks[job.dest.name] = dm.pick_stage_dir(job, lock)

    try:
        with contextlib.redirect_stdout(buf):
            threads = [threading.Thread(target=worker, args=(j,)) for j in jobs]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        local = [n for n, p in picks.items() if p == dm.LOCAL_STAGE / n]
        volume = [n for n, p in picks.items() if p != dm.LOCAL_STAGE / n]
        assert len(picks) == dm.POOL_SIZE
        assert len(local) == 1, \
            f"three 30 GB jobs on a 100 GB disk: exactly one may stage locally, got {picks}"
        assert len(volume) == 2, f"the other two must fall back to the volume, got {picks}"
        assert dm._local_reserved == 30 * 1024 ** 3, \
            "the winner's bytes must stay reserved while it is in flight"
        assert "already committed" in buf.getvalue(), \
            "a caller losing to a reservation must say so"

        # Releasing gives the budget back, so a later job can use the disk.
        for j in jobs:
            dm.release_reservation(j, lock)
        assert dm._local_reserved == 0
        later = dm.Job(url="u", dest=tmp / "models" / "late.safetensors", kind="hf")
        later.total_bytes = 30 * 1024 ** 3
        with contextlib.redirect_stdout(buf):
            assert dm.pick_stage_dir(later, lock) == dm.LOCAL_STAGE / later.dest.name, \
                "a released reservation must return the budget"
    finally:
        dm.shutil.disk_usage = real_du
        dm.reset_reservations()
    print("ok: local-stage reservations serialise the headroom decision")


SIGNED_URL = ("https://acct.r2.cloudflarestorage.com/bucket/client.safetensors"
              "?X-Amz-Credential=AKIAEXAMPLE&X-Amz-Signature=c0ffee0123456789")


def test_error_redaction(tmp):
    """BUG 3: a failed download must not print its URL's query string.

    All boot stdout is teed to $NETWORK_VOLUME/comfyui.log (src/start.sh:115),
    the file the triage block tells a customer to paste into Discord. A
    presigned R2 signature must not be in it.
    """
    import json
    reset_calls()
    dm.LOCAL_STAGE = tmp / "hf_stage"
    SUB_FAIL["exit_code"] = 22
    status_path = tmp / "hf_status.json"
    os.environ["HF_STATUS_FILE"] = str(status_path)
    dest = tmp / "models" / "loras" / "client.safetensors"
    man = tmp / "m.tsv"
    man.write_text(f"{SIGNED_URL}\t{dest}\t0.01\n")
    try:
        rc, out = run_main(man)
    finally:
        del os.environ["HF_STATUS_FILE"]
        SUB_FAIL["exit_code"] = 0

    assert rc == 1, f"a failed aria2c entry must fail the run, got {rc}"
    for needle in ("X-Amz-Signature", "c0ffee0123456789", "X-Amz-Credential",
                   "AKIAEXAMPLE"):
        assert needle not in out, f"{needle} leaked into the boot log:\n{out}"
    assert "exited 22" in out, f"the exit code must survive redaction:\n{out}"
    assert "acct.r2.cloudflarestorage.com/bucket/client.safetensors" in out, \
        f"host and path must survive redaction:\n{out}"

    status = json.loads(status_path.read_text())
    entry = status["client.safetensors"]
    blob = json.dumps(status)
    for needle in ("X-Amz-Signature", "c0ffee0123456789", "AKIAEXAMPLE"):
        assert needle not in blob, f"{needle} leaked into the status file:\n{blob}"
    assert entry["status"] == "failed"
    assert "exited 22" in entry["error"]
    assert entry["url"] == ("https://acct.r2.cloudflarestorage.com/bucket/"
                            "client.safetensors"), entry

    # The scrubber also covers messages we did not format: a child's stderr or
    # huggingface_hub echoing the URL it was handed.
    scrubbed = dm.scrub_urls(f"aria2c: download failed for {SIGNED_URL} (403)")
    assert "X-Amz-Signature" not in scrubbed and "403" in scrubbed, scrubbed
    assert dm.redact_url("https://user:pw@host.example.com:8443/a/b?x=1") == \
        "https://host.example.com:8443/a/b", "userinfo must go too"
    print("ok: a failed download logs exit code + host/path, never the signature")


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


def test_symlink_handoff(tmp):
    """The manager leaves a symlink into the stage, not a copy on the volume.

    This is the whole point of NVMe-first provisioning: the boot must not wait
    for bytes to cross the network volume. Verified the way ComfyUI resolves a
    model, i.e. through os.path.isfile, which follows the link.
    """
    reset_calls()
    dm.LOCAL_STAGE = tmp / "hf_stage"
    HF_BEHAVIOR["size_bytes"] = 64 * 1024
    dest = tmp / "models" / "diffusion_models" / "hfmodel.safetensors"
    man = tmp / "m.tsv"
    man.write_text(f"{HF_URL}\t{dest}\t0.01\n")

    rc, out = run_main(man)
    assert rc == 0, f"expected 0, got {rc}\n{out}"
    assert dest.is_symlink(), f"dest must be a symlink, not a copy\n{out}"
    target = Path(os.readlink(dest))
    assert dm.LOCAL_STAGE in target.parents, \
        f"symlink must point into the stage, points at {target}"
    assert os.path.isfile(dest), "os.path.isfile must follow it (folder_paths.py:453)"
    assert dest.stat().st_size == 64 * 1024, "stat must follow the link to the real size"
    assert target.is_file(), "the staged file must survive; volume_sync consumes it"
    assert not dest.with_name(dest.name + ".partial").exists()
    print("ok: the manager hands off a symlink into the stage, not a volume copy")


def test_existing_state_table(tmp):
    """Every row of spec section 2b, in one pass.

    Row 3 (live symlink) is the one the pre-2026-08-13 code could not express:
    is_file() follows a symlink, so it skipped the download and then never
    copied, leaving the pod depending on a stage dir the next restart wipes.
    """
    reset_calls()
    dm.LOCAL_STAGE = tmp / "hf_stage"
    dm.LOCAL_STAGE.mkdir(parents=True, exist_ok=True)
    HF_BEHAVIOR["size_bytes"] = 64 * 1024
    models = tmp / "models" / "diffusion_models"
    models.mkdir(parents=True)

    landed = models / "landed.safetensors"          # row 1: real file, big enough
    landed.write_bytes(b"\0" * 64 * 1024)
    small = models / "small.safetensors"            # row 2: real file, sub-floor
    small.write_bytes(b"\0" * 128)
    live = models / "live.safetensors"              # row 3: symlink, target alive
    live_target = dm.LOCAL_STAGE / "live.safetensors"
    live_target.write_bytes(b"\0" * 64 * 1024)
    live.symlink_to(live_target)
    dead = models / "dead.safetensors"              # row 4: symlink, target gone
    dead.symlink_to(dm.LOCAL_STAGE / "vanished.safetensors")
    absent = models / "absent.safetensors"          # row 5

    man = tmp / "m.tsv"
    man.write_text("".join(
        f"https://huggingface.co/org/repo/resolve/main/{p.name}\t{p}\t0.01\n"
        for p in (landed, small, live, dead, absent)))
    downloaded_before = len(HF_CALLS)
    rc, out = run_main(man)
    assert rc == 0, f"expected 0, got {rc}\n{out}"

    fetched = {c["filename"] for c in HF_CALLS[downloaded_before:]}
    assert "landed.safetensors" not in fetched, "row 1: a landed file must not refetch"
    assert "small.safetensors" in fetched, "row 2: a sub-floor file must refetch"
    assert "live.safetensors" not in fetched, \
        "row 3: a symlink with a live target must NOT refetch"
    assert "dead.safetensors" in fetched, "row 4: a dangling symlink must refetch"
    assert "absent.safetensors" in fetched, "row 5: a missing file must download"

    assert not landed.is_symlink() and landed.is_file(), "row 1 must stay a real file"
    assert live.is_symlink() and live.exists(), "row 3's symlink must survive intact"
    assert os.readlink(live) == str(live_target), "row 3 must still point at its target"
    assert dead.is_symlink() and dead.exists(), \
        "row 4 must end as a fresh, resolvable symlink"
    print("ok: all five rows of the existing-state table behave as specified")


def test_dangling_symlink_sweep(tmp):
    """A dangling symlink for a model NOT in this manifest is still removed.

    ComfyUI lists it via os.walk(followlinks=True) and then logs
    "doesn't link anywhere" on every load attempt. It is dropdown litter.
    """
    reset_calls()
    dm.LOCAL_STAGE = tmp / "hf_stage"
    HF_BEHAVIOR["size_bytes"] = 32 * 1024
    models = tmp / "models" / "diffusion_models"
    models.mkdir(parents=True)
    dest = models / "hfmodel.safetensors"

    orphan = models / "disabled_model.safetensors"
    orphan.symlink_to(tmp / "hf_stage" / "gone.safetensors")
    real = models / "real.safetensors"
    real.write_bytes(b"\0" * 4096)
    good_target = tmp / "elsewhere.safetensors"
    good_target.write_bytes(b"\0" * 4096)
    good_link = models / "good.safetensors"
    good_link.symlink_to(good_target)

    man = tmp / "m.tsv"
    man.write_text(f"{HF_URL}\t{dest}\t0.01\n")
    rc, out = run_main(man)

    assert rc == 0, f"expected 0, got {rc}\n{out}"
    assert not orphan.is_symlink() and not orphan.exists(), \
        "a dangling symlink outside the manifest must be swept"
    assert real.is_file(), "the sweep must not touch real files"
    assert good_link.is_symlink() and good_link.exists(), \
        "the sweep must not touch symlinks that resolve"
    print("ok: dangling symlinks are swept, live links and real files are not")


def main() -> int:
    test_env_timeouts()
    for test in (test_parse_manifest, test_exit_codes, test_happy_path,
                 test_skip_and_refetch, test_floor_failure,
                 test_stage_fallback,
                 test_no_volume_disables_local_staging,
                 test_stage_local_flag_reads_the_env,
                 test_direct_headroom,
                 test_stage_reservation, test_error_redaction,
                 test_status_file, test_watchdog,
                 test_watchdog_deadline, test_orphan_partial_sweep,
                 test_symlink_handoff, test_existing_state_table,
                 test_dangling_symlink_sweep):
        with tempfile.TemporaryDirectory() as tmp:
            test(Path(tmp))
    print("all download manager self-tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
