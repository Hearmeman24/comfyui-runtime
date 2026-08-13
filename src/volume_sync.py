#!/usr/bin/env python3
"""Move staged models from the pod's local disk onto the network volume.

Runs detached, AFTER ComfyUI is already serving. The download manager publishes
each model as a symlink at its volume path pointing into /hf_stage, so ComfyUI
can load it at NVMe speed immediately; this process turns each of those links
into a real file on the volume so the model survives the pod.

Deliberately has no watchdog, no deadline and no failure exit code. It cannot
block anything, so a slow volume is a slow copy, not an incident. That is the
whole point: the blocking version of this copy used to hold the boot for
minutes and was killed mid-flight by the download manager's stall watchdog
(minimax pod, 2026-08-13).

Copy order is manifest order, which is registry order, so a pod terminated
early keeps the models that were queued first rather than a random subset.

Run: python3 volume_sync.py <manifest.tsv>
"""
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hf_download_manager import fmt_bytes, fmt_dur, parse_manifest  # noqa: E402

SUMMARY_INTERVAL = 30  # seconds; the user is in ComfyUI by now, not reading logs


def log(msg: str) -> None:
    print(f"[volume-sync] {msg}", flush=True)


def pending(manifest: Path) -> list:
    """Manifest entries still published as a symlink with a live target.

    A real file is already landed. A dangling symlink is the download manager's
    problem on the next boot, not ours, and copying from it is impossible.
    """
    out = []
    for job in parse_manifest(manifest):
        dest = job.dest
        if not dest.is_symlink():
            continue
        try:
            target = Path(os.readlink(dest))
        except OSError:
            continue
        if target.is_file():
            out.append((dest, target))
    return out


def land(dest: Path, target: Path) -> int:
    """Copy one staged model onto the volume and free the local copy.

    .partial + os.replace so the real path is never a half file, matching the
    download manager's own invariant. os.replace over a symlink replaces the
    link itself, and any handle ComfyUI already holds keeps the old inode alive
    until it closes it.
    """
    size = target.stat().st_size
    tmp = dest.with_name(dest.name + ".partial")
    tmp.unlink(missing_ok=True)
    shutil.copy2(target, tmp)
    copied = tmp.stat().st_size
    if copied != size:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"copied {copied} bytes, expected {size}")
    os.replace(tmp, dest)
    target.unlink(missing_ok=True)
    return size


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: volume_sync.py <manifest.tsv>", file=sys.stderr)
        return 2
    manifest = Path(sys.argv[1])
    if not manifest.is_file():
        log(f"no manifest at {manifest}, nothing to do")
        return 0

    todo = pending(manifest)
    if not todo:
        log("every model is already on the volume, nothing to copy")
        return 0

    total = sum(t.stat().st_size for _, t in todo)
    started = time.time()
    log(f"copying {len(todo)} model(s), {fmt_bytes(total)}, to the network "
        f"volume in the background. ComfyUI is usable now.")

    done_bytes, failures, last_summary = 0, 0, time.time()
    for i, (dest, target) in enumerate(todo, 1):
        try:
            done_bytes += land(dest, target)
            log(f"landed {dest.name} ({i}/{len(todo)})")
        except Exception as e:
            # Leave the symlink alone: it still resolves, so ComfyUI keeps
            # working off the stage for this pod's life and the next boot
            # re-downloads. Never abort the run for one file.
            failures += 1
            log(f"FAILED {dest.name}: {e} (left staged; it will re-download "
                f"on the next boot)")
        if time.time() - last_summary >= SUMMARY_INTERVAL and i < len(todo):
            log(f"{i}/{len(todo)} landed, {fmt_bytes(total - done_bytes)} left")
            last_summary = time.time()

    elapsed = fmt_dur(time.time() - started)
    if failures:
        log(f"finished with {failures} failure(s) in {elapsed}. Those models "
            f"are still on local disk only and will re-download next boot.")
    else:
        log(f"copied {fmt_bytes(done_bytes)} in {elapsed}")
        log("All models are on your volume. Safe to restart or terminate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
