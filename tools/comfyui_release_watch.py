#!/usr/bin/env python3
"""Detect a newer ComfyUI release than versions.json pins.

  comfyui_release_watch.py           print "current" or "new <tag> <sha>"
  comfyui_release_watch.py --apply   also rewrite versions.json + base/Dockerfile

Only stable (non-draft, non-prerelease) releases count, via
/releases/latest. The tag is resolved to its commit sha (annotated tags are
peeled). Writes GITHUB_OUTPUT keys release/sha/label when set, so the
scheduled workflow can branch, commit and open the PR. Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request

import versions

UPSTREAM = "comfyanonymous/ComfyUI"


def _get(path: str):
    req = urllib.request.Request("https://api.github.com" + path)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "comfyui-release-watch")
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def latest(get=_get) -> tuple[str, str]:
    tag = get(f"/repos/{UPSTREAM}/releases/latest")["tag_name"]
    obj = get(f"/repos/{UPSTREAM}/git/ref/tags/{tag}")["object"]
    while obj["type"] == "tag":
        obj = get(f"/repos/{UPSTREAM}/git/tags/{obj['sha']}")["object"]
    return tag, obj["sha"]


def _key(tag: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", tag))


def is_newer(candidate: str, current: str) -> bool:
    return _key(candidate) > _key(current)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)
    data = versions.load()
    tag, sha = latest()
    current = data["comfyui"]["release"]
    if not is_newer(tag, current):
        print(f"current: pinned {current}, latest stable {tag}")
        return 0
    print(f"new {tag} {sha} (pinned {current})")
    if args.apply:
        data = versions.set_release(tag, sha)
        versions.sync(data)
        problems = versions.check(data)
        if problems:
            print("\n".join(problems), file=sys.stderr)
            return 1
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as fh:
            fh.write(f"release={tag}\nsha={sha}\nlabel={versions.label(data)}\nprevious={current}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
