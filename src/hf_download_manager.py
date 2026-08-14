#!/usr/bin/env python3
"""Unified manifest-driven model download manager.

Reads a manifest of `<url>\\t<abs_dest>[\\t<min_size_mb>]` lines (CONTRACTS.md
section 1), downloads each entry with a thread pool of 3 and a single snapshot
logger that prints an append-only status block every 10 seconds (non-TTY safe,
no carriage-return progress bars).

Per-URL routing, so nothing is ever silently dropped:
- an HF resolve URL goes through hf_hub_download (with hf_xet acceleration),
- a Google Drive URL goes through gdown,
- anything else goes through aria2c.

Every download is staged off the destination volume (local disk when it has
room). NVMe-first: a locally staged model is published at its volume path as a
SYMLINK into the stage, so ComfyUI can load it immediately and the boot never
waits for bytes to cross the network volume. `volume_sync.py` runs detached,
after the launch, and turns each link into a real file. When there was no room
locally the file staged on the volume already, and the rename is instant.

Either way the publish goes through `<dest>.partial` + os.replace, so a partial
file is never visible at the path a workflow loads from.
"""
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, urlsplit, urlunsplit
from urllib.request import Request, urlopen

# Bound per-request HTTP waits so a stuck/gated transfer raises instead of
# blocking forever (set before huggingface_hub reads it at import time).
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "15")

from huggingface_hub import HfFileSystem, hf_hub_download  # noqa: E402

# 3, never more: RunPod's network volume caps around 150 MB/s aggregate, so
# extra streams only split the same bandwidth.
POOL_SIZE = 3
SNAPSHOT_INTERVAL = 10  # seconds
STALL_SECS = 300        # abandon if no global byte progress for this long
DEADLINE_SECS = 3600    # absolute backstop for the whole download phase

# "Looks complete" floor in MB when a manifest line carries no third field.
DEFAULT_MIN_SIZE_MB = 10.0

# Stage on the pod's own disk, not on the network volume.
#
# hf_xet writes thousands of small chunk files under .cache/huggingface before
# assembling the result. Doing that over NFS is slow to write and much worse to
# read back: stage_bytes() walks the staging tree on every snapshot, and an
# os.walk over thousands of NFS files can block the logger for minutes, which
# looks exactly like a hung download.
#
# Staging locally also takes the volume out of the download path entirely; the
# finished file crosses to it once, as a single sequential copy.
LOCAL_STAGE = Path(os.getenv("HF_LOCAL_STAGE", "/hf_stage"))
# xet keeps its chunk cache alongside the assembled file, so a file needs
# roughly 2x its own size on disk while in flight. Ask for 2.5x before
# committing to the local path; a container disk too small for the file falls
# back to the volume.
LOCAL_STAGE_HEADROOM = 2.5
# Bound the ranged-GET size probe used for non-HF URLs: one round trip per
# direct entry at boot, and a dead host must not add minutes to it.
SIZE_PROBE_TIMEOUT = 15

# https://huggingface.co/<repo_id>/resolve/<revision>/<file_in_repo>[?...]
HF_URL_RE = re.compile(
    r"^https?://huggingface\.co/(?P<repo>[^/]+/[^/]+)/resolve/(?P<rev>[^/]+)/(?P<file>.+?)(?:\?.*)?$"
)

GDRIVE_HOSTS = {"drive.google.com", "docs.google.com"}

# Any http(s) URL inside a free-text error message.
URL_IN_TEXT_RE = re.compile(r"https?://[^\s'\"<>\\]+")


def redact_url(url: str) -> str:
    """scheme://host/path, with the query string and any userinfo dropped.

    A presigned R2/S3 link carries a LIVE signature in its query string, and a
    failed download's message is printed by snapshot() into stdout, which
    start.sh:115 tees to $NETWORK_VOLUME/comfyui.log - the file the triage
    block tells a customer to paste into Discord. Host and path are what
    troubleshooting actually needs; the credential is not.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return "<url>"
    if not parts.scheme:
        return url
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def scrub_urls(text: str) -> str:
    """Redact every URL in a message we did not write.

    aria2c, gdown and huggingface_hub all echo the URL they were given, so
    redacting only our own strings would leave the signature in the child
    process's message.
    """
    return URL_IN_TEXT_RE.sub(lambda m: redact_url(m.group(0)), text)


@dataclass
class Job:
    url: str
    dest: Path
    kind: str  # hf | gdrive | direct
    min_size_mb: float = DEFAULT_MIN_SIZE_MB
    repo_id: str = ""
    revision: str = ""
    filename: str = ""
    total_bytes: int = 0
    status: str = "queued"  # queued | active | done | failed
    start_ts: float = 0.0
    end_ts: float = 0.0
    stage_dir: Optional[Path] = None
    error: Optional[str] = None
    reserved_bytes: int = 0  # local-stage budget this job holds while in flight


def floor_bytes(job: Job) -> int:
    return int(job.min_size_mb * 1024 * 1024)


def write_status(jobs: list) -> None:
    """Optional boot-report feed (EXECUTION.md item N1): when HF_STATUS_FILE
    is set in the environment, record every entry's final status and error so
    boot_report.py can name each failure with its reason. Never fatal; the
    deployment report falls back to a disk audit without it."""
    path = os.environ.get("HF_STATUS_FILE")
    if not path:
        return
    try:
        # Redacted here too: boot_report.py only ever reads the host out of
        # this field (boot_report.py:136), and /tmp/hf_download_status.json is
        # one more file a customer can be asked to paste.
        payload = {j.dest.name: {"status": j.status, "error": j.error,
                                 "url": redact_url(j.url)} for j in jobs}
        Path(path).write_text(json.dumps(payload, indent=1))
    except OSError as e:
        print(f"[hf-manager] could not write status file: {e}", flush=True)


def parse_manifest(path: Path) -> list[Job]:
    jobs = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" not in line:
            print(f"[hf-manager] skip malformed line: {line}", flush=True)
            continue
        fields = line.split("\t")
        if len(fields) > 3:
            print(f"[hf-manager] ignoring extra fields on line: {line}", flush=True)
        url, dest = fields[0], fields[1]
        min_size_mb = DEFAULT_MIN_SIZE_MB
        if len(fields) >= 3 and fields[2].strip():
            try:
                min_size_mb = float(fields[2])
            except ValueError:
                print(f"[hf-manager] bad min_size_mb {fields[2]!r} for {dest}; "
                      f"using {DEFAULT_MIN_SIZE_MB}", flush=True)
        m = HF_URL_RE.match(url)
        if m:
            jobs.append(Job(url=url, dest=Path(dest), kind="hf",
                            min_size_mb=min_size_mb, repo_id=m["repo"],
                            revision=m["rev"], filename=m["file"]))
        elif urlparse(url).netloc in GDRIVE_HOSTS:
            jobs.append(Job(url=url, dest=Path(dest), kind="gdrive",
                            min_size_mb=min_size_mb))
        else:
            # Everything else (presigned R2, GitHub raw, plain http) downloads
            # via aria2c. wan used to print "skip non-HF url" and drop the
            # entry; that behavior is dead (CONTRACTS.md section 1).
            jobs.append(Job(url=url, dest=Path(dest), kind="direct",
                            min_size_mb=min_size_mb))
    return jobs


def fetch_sizes(jobs: list[Job]) -> None:
    fs = HfFileSystem(token=os.environ.get("HF_TOKEN") or None)
    for j in jobs:
        try:
            info = fs.info(f"{j.repo_id}/{j.filename}", revision=j.revision)
            j.total_bytes = int(info.get("size") or 0)
        except Exception as e:
            print(f"[hf-manager] size lookup failed for {j.dest.name}: "
                  f"{scrub_urls(str(e))}", flush=True)


def ranged_size(url: str) -> int:
    """Size of a non-HF URL via a one-byte ranged GET, or 0 if unknown.

    A ranged GET, never a HEAD: a presigned R2/S3 URL is signed for GET and
    answers a HEAD with 403 (CLAUDE.md section 3). A well-behaved server
    replies 206 with `Content-Range: bytes 0-0/<total>`; one that ignores the
    Range header replies 200, and then Content-Length is the whole file.
    """
    req = Request(url, method="GET", headers={"Range": "bytes=0-0"})
    with urlopen(req, timeout=SIZE_PROBE_TIMEOUT) as resp:
        cr = resp.headers.get("Content-Range") or ""
        if "/" in cr:
            total = cr.rsplit("/", 1)[1].strip()
            if total.isdigit():
                return int(total)
        if getattr(resp, "status", None) == 200:
            return int(resp.headers.get("Content-Length") or 0)
    return 0


def fetch_direct_sizes(jobs: list[Job]) -> None:
    """Size the aria2c entries so pick_stage_dir has something to weigh.

    Costs one round trip per direct entry at boot. That is cheap next to the
    file itself, and there is no other source: a presigned R2 link is exactly
    the kind of entry that runs to tens of GB (CONTRACTS.md:85-87), and without
    a size it would take the local disk unmeasured.

    gdrive entries are deliberately not probed - Drive answers a plain GET with
    an HTML confirmation page whose Content-Length is not the file's - so they
    stay size-unknown and stage on the volume.
    """
    for j in jobs:
        try:
            j.total_bytes = ranged_size(j.url)
        except Exception as e:
            print(f"[hf-manager] size probe failed for {j.dest.name}: "
                  f"{scrub_urls(str(e))}", flush=True)
        if not j.total_bytes:
            print(f"[hf-manager] {j.dest.name}: size unknown; staging on the volume",
                  flush=True)


def fmt_bytes(n: float) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def fmt_dur(secs: float) -> str:
    secs = int(max(secs, 0))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def stage_bytes(job: Job) -> int:
    if not job.stage_dir or not job.stage_dir.exists():
        return 0
    total = 0
    for root, _, files in os.walk(job.stage_dir):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def partial_path(job: Job) -> Path:
    return job.dest.with_name(job.dest.name + ".partial")


# Bytes the in-flight jobs have already committed to LOCAL_STAGE. Guarded by
# the run lock that run_job/snapshot already share; a second lock would only
# add a way to take them in the wrong order.
_local_reserved = 0


def reset_reservations() -> None:
    global _local_reserved
    _local_reserved = 0


def release_reservation(job: Job, lock: Optional[threading.Lock] = None) -> None:
    """Give a finished job's local-stage budget back.

    By the time a job ends, whatever it wrote is real: on success the staged
    file IS the model and disk_usage already counts it, on failure the stage
    is gone and the space is back. Either way the reservation has served its
    purpose and holding it longer would double-count.
    """
    global _local_reserved
    with (lock or contextlib.nullcontext()):
        _local_reserved = max(_local_reserved - job.reserved_bytes, 0)
        job.reserved_bytes = 0


def pick_stage_dir(job: Job, lock: Optional[threading.Lock] = None) -> Path:
    """Local disk when it has room for this file, else beside the destination.

    Falling back matters: the container disk is typically 60 GB while single
    models run to 25 GB, and three of those in flight would fill it. A file
    that doesn't fit locally still downloads, just the slower way.

    Two things the naive version got wrong:

    - An unknown size (`total_bytes == 0`) used to mean "stage locally",
      i.e. exactly the entries nobody had measured skipped the guard. It now
      means the opposite: an unmeasured file does not get to risk the disk.
    - `disk_usage().free` is a snapshot, and POOL_SIZE workers read it
      concurrently. Three of them could each see 60 GB free and each commit a
      25 GB model to it, and the loser hits ENOSPC halfway down. Bytes already
      committed are subtracted from the budget under the run lock, so the
      second and third caller see what the first one took.
    """
    global _local_reserved
    need = int(job.total_bytes * LOCAL_STAGE_HEADROOM)
    try:
        LOCAL_STAGE.mkdir(parents=True, exist_ok=True)
        with (lock or contextlib.nullcontext()):
            free = shutil.disk_usage(LOCAL_STAGE).free
            budget = free - _local_reserved
            if job.total_bytes and budget > need:
                _local_reserved += job.total_bytes
                job.reserved_bytes = job.total_bytes
                return LOCAL_STAGE / job.dest.name
            reserved = _local_reserved
        if job.total_bytes:
            print(f"[hf-manager] {job.dest.name}: staging on the volume "
                  f"({fmt_bytes(free)} free locally, {fmt_bytes(reserved)} already "
                  f"committed, needs {fmt_bytes(need)})", flush=True)
        else:
            print(f"[hf-manager] {job.dest.name}: staging on the volume "
                  f"(size unknown, so the local disk is not risked)", flush=True)
    except OSError as e:
        print(f"[hf-manager] local staging unavailable ({e}); using the volume",
              flush=True)
    return job.dest.parent / ".hf_stage" / job.dest.name


def stage_is_local(stage: Path) -> bool:
    """True when this job staged on the pod's own disk rather than the volume.

    Decides whether the handoff can be deferred: a local stage means a
    cross-filesystem copy worth backgrounding, a volume stage means an instant
    same-filesystem rename.
    """
    try:
        return LOCAL_STAGE.resolve() in stage.resolve().parents
    except OSError:
        return False


def skip_if_present(job: Job) -> bool:
    """Decide this entry against what is already on the volume.

    The five cases, per spec section 2b. is_file() and stat() BOTH follow
    symlinks, which is why a naive is_file() check gets row 3 wrong: it skips
    the download and then nothing ever copies, leaving the pod depending on a
    stage dir the next restart wipes.

    | dest                         | action                                |
    |------------------------------|---------------------------------------|
    | regular file, >= floor       | done                                  |
    | regular file, < floor        | delete, refetch                       |
    | symlink, target >= floor     | done, and volume_sync still owes it    |
    | symlink, target missing/small| unlink, refetch                       |
    | absent                       | download                              |
    """
    if job.dest.is_symlink():
        # exists() follows the link, so False here means dangling.
        if not job.dest.exists() or job.dest.stat().st_size < floor_bytes(job):
            print(f"[hf-manager] removing unusable symlink: {job.dest.name}",
                  flush=True)
            job.dest.unlink(missing_ok=True)
            return False
        print(f"[hf-manager] skip {job.dest.name} "
              f"(staged, still to be copied to the volume)", flush=True)
        return True
    if not job.dest.is_file():
        return False
    size = job.dest.stat().st_size
    if size < floor_bytes(job):
        print(f"[hf-manager] deleting sub-floor file "
              f"({fmt_bytes(size)} < {job.min_size_mb:g} MB): {job.dest}", flush=True)
        job.dest.unlink()
        return False
    print(f"[hf-manager] skip {job.dest.name} ({fmt_bytes(size)} on disk)", flush=True)
    return True


def download_hf(job: Job, stage: Path) -> Path:
    # local_dir keeps everything (xet chunk cache included) inside the staging
    # dir; hf_hub_download must never touch ~/.cache/huggingface. The token is
    # user-supplied only, never baked or defaulted.
    local_path = hf_hub_download(
        repo_id=job.repo_id,
        filename=job.filename,
        revision=job.revision,
        local_dir=str(stage),
        token=os.environ.get("HF_TOKEN") or None,
    )
    return Path(local_path)


def run_downloader(cmd: list, job: Job) -> None:
    """Run a child downloader without letting its argv reach an error message.

    `check=True` raises CalledProcessError, whose __str__ embeds the whole
    command list - URL, signature and all - and run_job stores that as the
    entry's error, which snapshot() prints into the boot log. The exit code
    and the redacted URL are what triage needs.
    """
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} exited {proc.returncode} "
                           f"fetching {redact_url(job.url)}")


def download_gdrive(job: Job, stage: Path) -> Path:
    # --fuzzy accepts share links (file/d/<id>/view), not only uc?id= URLs.
    out = stage / job.dest.name
    run_downloader(["gdown", "--fuzzy", "-O", str(out), job.url], job)
    return out


def download_aria(job: Job, stage: Path) -> Path:
    cmd = [
        "aria2c", "-x", "16", "-s", "16", "-k", "1M",
        "--continue=true", "--summary-interval=0", "--console-log-level=warn",
        "-d", str(stage), "-o", job.dest.name, job.url,
    ]
    run_downloader(cmd, job)
    return stage / job.dest.name


DOWNLOADERS = {"hf": download_hf, "gdrive": download_gdrive, "direct": download_aria}


def run_job(job: Job, lock: threading.Lock) -> None:
    with lock:
        job.status = "active"
        job.start_ts = time.time()
    try:
        job.dest.parent.mkdir(parents=True, exist_ok=True)
        stage = pick_stage_dir(job, lock)
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        stage.mkdir(parents=True, exist_ok=True)
        with lock:
            job.stage_dir = stage

        src = DOWNLOADERS[job.kind](job, stage)
        if not src.is_file():
            raise RuntimeError(f"download claimed success but no file at {src}")
        size = src.stat().st_size
        # Verify at the stage, before the handoff, so a sub-floor file never
        # reaches the path a workflow loads from.
        if size < floor_bytes(job):
            raise RuntimeError(
                f"downloaded {fmt_bytes(size)}, below the {job.min_size_mb:g} MB floor")

        if src.resolve() != job.dest.resolve():
            if stage_is_local(stage):
                # NVMe-first: publish a SYMLINK and let volume_sync.py move the
                # bytes after ComfyUI is already up. Copying ~150 MB/s of
                # network volume here is what used to hold the boot for minutes.
                #
                # Safe because ComfyUI resolves models with os.path.isfile,
                # which follows symlinks, and carries no containment check on
                # that path (folder_paths.py:449-458 at v0.32.0).
                #
                # Link via a .partial name + os.replace so publishing is atomic
                # even when something already occupies dest.
                tmp = partial_path(job)
                tmp.unlink(missing_ok=True)
                os.symlink(src, tmp)
                os.replace(tmp, job.dest)
                # The stage file IS the model now; do not delete the stage.
                with lock:
                    job.status = "done"
                    job.end_ts = time.time()
                return
            # Volume-staged fallback (pick_stage_dir found no room on the local
            # disk): same filesystem, so this rename is atomic and instant.
            # There is nothing to defer.
            tmp = partial_path(job)
            shutil.move(str(src), str(tmp))
            os.replace(tmp, job.dest)
        shutil.rmtree(stage, ignore_errors=True)

        if not job.dest.is_file() or job.dest.stat().st_size < floor_bytes(job):
            raise RuntimeError(f"final verify failed at {job.dest}")

        with lock:
            job.status = "done"
            job.end_ts = time.time()
    except Exception as e:
        with lock:
            job.status = "failed"
            # Belt and braces on top of run_downloader: hf_hub_download and a
            # child's own stderr both echo URLs we did not format.
            job.error = scrub_urls(str(e))
            job.end_ts = time.time()
    finally:
        release_reservation(job, lock)


def snapshot(jobs: list[Job], started_at: float, lock: threading.Lock,
             prev_bytes: dict) -> bool:
    with lock:
        active = [j for j in jobs if j.status == "active"]
        queued = [j for j in jobs if j.status == "queued"]
        done = [j for j in jobs if j.status == "done"]
        failed = [j for j in jobs if j.status == "failed"]

    now = time.time()
    elapsed = now - started_at
    lines = [
        f"=== Downloads @ {fmt_dur(elapsed)} - "
        f"{len(done)}/{len(jobs)} done, {len(active)} active, "
        f"{len(queued)} queued, {len(failed)} failed ==="
    ]
    if active:
        lines.append("ACTIVE")
        for j in active:
            cur = stage_bytes(j)
            pct = (cur / j.total_bytes * 100) if j.total_bytes else 0.0
            prev_t, prev_b = prev_bytes.get(id(j), (j.start_ts, 0))
            dt = max(now - prev_t, 0.001)
            speed = max(cur - prev_b, 0) / dt
            eta = (j.total_bytes - cur) / speed if speed > 0 and j.total_bytes else 0
            prev_bytes[id(j)] = (now, cur)
            lines.append(
                f"  {j.dest.name:<60} {pct:5.1f}%  "
                f"{fmt_bytes(cur):>9}/{fmt_bytes(j.total_bytes):<9}  "
                f"{fmt_bytes(speed)}/s  ETA {fmt_dur(eta)}"
            )
    if queued:
        names = ", ".join(j.dest.name for j in queued[:8])
        more = "" if len(queued) <= 8 else f", +{len(queued) - 8} more"
        lines.append(f"QUEUED ({len(queued)}): {names}{more}")
    if failed:
        lines.append("FAILED")
        for j in failed:
            lines.append(f"  {j.dest.name}: {j.error}")
    print("\n".join(lines), flush=True)
    return bool(active or queued)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: hf_download_manager.py <manifest>", file=sys.stderr)
        return 2
    manifest = Path(sys.argv[1])
    if not manifest.exists():
        print(f"[hf-manager] manifest not found: {manifest}", flush=True)
        return 0
    jobs = parse_manifest(manifest)
    if not jobs:
        print("[hf-manager] empty manifest, nothing to do", flush=True)
        write_status(jobs)
        return 0

    # A pod killed mid-move leaves <name>.partial behind. It is not loadable
    # and not what any size check looks at, so it would sit there forever.
    # Sweep the whole destination directory, not just this manifest's names:
    # a model that has since been flag-disabled or renamed in the registry
    # never appears in `jobs` again, so a per-entry sweep can never reclaim it
    # (customer report, 2026-08-13). Only ever *.partial, only in directories
    # this manifest writes to.
    # Dangling symlinks go the same way. A pod restarted before volume_sync
    # finished leaves links pointing into a wiped /hf_stage. ComfyUI lists them
    # (os.walk followlinks=True, folder_paths.py:416) and then logs "doesn't
    # link anywhere" on every load, so they are dropdown litter. This runs
    # BEFORE the skip pass so a re-download is not blocked by an existing path.
    for stale_dir in {j.dest.parent for j in jobs}:
        if not stale_dir.is_dir():
            continue
        for stale in stale_dir.glob("*.partial"):
            print(f"[hf-manager] removing stale {stale.name}", flush=True)
            stale.unlink(missing_ok=True)
        for entry in stale_dir.iterdir():
            if entry.is_symlink() and not entry.exists():
                print(f"[hf-manager] removing dangling symlink {entry.name}",
                      flush=True)
                entry.unlink(missing_ok=True)
    # The stage is LIVE STATE now, not scratch: a model published as a symlink
    # lives there until volume_sync.py copies it across. Wiping it here would
    # destroy every model a previous boot had staged but not yet copied, which
    # is exactly the case skip_if_present row 3 exists to preserve. Per-job
    # stage dirs are still cleared in run_job before that job downloads.

    for j in jobs:
        if skip_if_present(j):
            j.status = "done"
    pending = [j for j in jobs if j.status == "queued"]
    if not pending:
        print(f"[hf-manager] all {len(jobs)} files already on disk", flush=True)
        write_status(jobs)
        return 0

    print(f"[hf-manager] {len(pending)} of {len(jobs)} files to download; "
          f"resolving sizes...", flush=True)
    fetch_sizes([j for j in pending if j.kind == "hf"])
    # aria2c entries get a size too, or pick_stage_dir has nothing to weigh
    # them against. Presigned R2 and CivitAI links route through this path
    # (CONTRACTS.md:85-87) and are not small.
    fetch_direct_sizes([j for j in pending if j.kind == "direct"])
    total = sum(j.total_bytes for j in pending)
    print(f"[hf-manager] total size: {fmt_bytes(total)}; "
          f"starting {POOL_SIZE}-way pool", flush=True)

    lock = threading.Lock()
    reset_reservations()
    started_at = time.time()
    prev_bytes: dict = {}

    # Watchdog: a single gated/hung transfer (e.g. an unauthorized 403 or a xet
    # stream stuck at 0 B/s) must never block the boot. If no global progress
    # happens for STALL_SECS, or the whole phase exceeds DEADLINE_SECS, abandon
    # the remaining jobs and hard-exit so ComfyUI still starts. hf_hub_download
    # threads can't be cancelled once blocked, so os._exit bypasses the pool
    # join rather than hanging on a stuck thread.
    last_finished, last_active_bytes, last_progress_at = 0, 0, time.time()

    pool = ThreadPoolExecutor(max_workers=POOL_SIZE)
    futures = [pool.submit(run_job, j, lock) for j in pending]
    try:
        while True:
            time.sleep(SNAPSHOT_INTERVAL)
            still_running = snapshot(jobs, started_at, lock, prev_bytes)
            if not still_running and all(f.done() for f in futures):
                break

            now = time.time()
            with lock:
                finished = sum(1 for j in jobs if j.status in ("done", "failed"))
                # Landed bytes are added so the total only ever rises: without
                # them a finished job's stage going to zero would leave a
                # high-water mark the next job has to climb back to before
                # anything counted as progress. The watchdog guards DOWNLOADS
                # only; the stage->volume copy is volume_sync.py's problem and
                # cannot block the boot, so it has no deadline at all.
                active_bytes = sum(stage_bytes(j) for j in jobs
                                   if j.status == "active")
                for j in jobs:
                    if j.status == "done":
                        try:
                            active_bytes += j.dest.stat().st_size
                        except OSError:
                            pass
            if finished > last_finished or active_bytes > last_active_bytes:
                last_finished, last_active_bytes, last_progress_at = finished, active_bytes, now

            stalled = (now - last_progress_at) > STALL_SECS
            expired = (now - started_at) > DEADLINE_SECS
            if stalled or expired:
                with lock:
                    stuck = [j for j in jobs if j.status in ("active", "queued")]
                    for j in stuck:
                        j.status = "failed"
                        j.error = j.error or ("stalled: no progress" if stalled else "deadline exceeded")
                reason = "no progress for %s" % fmt_dur(STALL_SECS) if stalled else "deadline exceeded"
                print(f"[hf-manager] {reason}: abandoning {len(stuck)} download(s) and continuing boot",
                      flush=True)
                snapshot(jobs, started_at, lock, prev_bytes)
                write_status(jobs)
                sys.stdout.flush()
                os._exit(1)  # leave stuck download threads behind; boot proceeds
    finally:
        pool.shutdown(wait=False)

    snapshot(jobs, started_at, lock, prev_bytes)
    write_status(jobs)
    failed = [j for j in jobs if j.status == "failed"]
    if failed:
        print(f"[hf-manager] {len(failed)} failures (boot continues; see notice + README)", flush=True)
        return 1
    print(f"[hf-manager] all {len(jobs)} downloads complete in {fmt_dur(time.time() - started_at)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
