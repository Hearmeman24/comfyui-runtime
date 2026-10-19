#!/usr/bin/env python3
"""Open one base-image bump PR per template after a base release.

Reads the template list and the new base tag from versions.json, then for each
template reads its pins.json from the template's default branch:

  * pins.json has "hold"       -> SKIPPED and reported; nothing is written.
  * base_image already current -> up to date; nothing is written.
  * a fan-out PR already open  -> reported; nothing is written (idempotent).
  * otherwise                  -> branch auto/comfyui-<label>, pins.json
                                  base_image + Dockerfile ARG BASE_IMAGE
                                  rewritten, PR opened. Never merged here.

The per-template override lives in the template's own pins.json because that
file already owns the template's base_image and runtime_ref (CONTRACTS.md
section 6): a held template keeps whatever base_image / runtime_ref it pins,
the hold is reviewed in that template's history, and no second matrix can
drift from it. Shape:

  "hold": {"reason": "AI-Toolkit breaks on v0.43.0", "since": "2026-10-19"}

`reason` is required so a hold is never silent. Remove the key to rejoin.

  fanout_templates.py --dry-run   print the plan (reads GitHub, writes nothing)
  fanout_templates.py             needs GITHUB_TOKEN with contents+pull-requests
                                  write on every template repo
Stdlib only.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

import versions

API = "https://api.github.com"


class FanoutError(Exception):
    pass


@dataclass
class Result:
    repo: str
    status: str  # held | current | open | opened | planned | error
    detail: str


class GitHub:
    def __init__(self, token: str | None) -> None:
        self.token = token

    def _req(self, method: str, path: str, body: dict | None = None):
        req = urllib.request.Request(API + path, method=method,
                                     data=json.dumps(body).encode() if body is not None else None)
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("User-Agent", "comfyui-runtime-fanout")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read() or b"null")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise FanoutError(f"{method} {path}: HTTP {exc.code} {exc.read()[:200]!r}") from exc

    def read_file(self, repo: str, path: str, ref: str) -> tuple[str, str] | None:
        q = urllib.parse.quote(ref)
        data = self._req("GET", f"/repos/{repo}/contents/{path}?ref={q}")
        if not data:
            return None
        return base64.b64decode(data["content"]).decode(), data["sha"]

    def open_pr(self, repo: str, head: str) -> str | None:
        owner = repo.split("/")[0]
        prs = self._req("GET", f"/repos/{repo}/pulls?state=open&head={owner}:{urllib.parse.quote(head)}") or []
        return prs[0]["html_url"] if prs else None

    def branch_sha(self, repo: str, branch: str) -> str:
        data = self._req("GET", f"/repos/{repo}/git/ref/heads/{urllib.parse.quote(branch)}")
        if not data:
            raise FanoutError(f"{repo}: branch {branch} not found")
        return data["object"]["sha"]

    def create_branch(self, repo: str, name: str, sha: str) -> None:
        if self._req("GET", f"/repos/{repo}/git/ref/heads/{urllib.parse.quote(name)}") is None:
            self._req("POST", f"/repos/{repo}/git/refs", {"ref": f"refs/heads/{name}", "sha": sha})

    def write_file(self, repo: str, path: str, branch: str, content: str, sha: str, message: str) -> None:
        self._req("PUT", f"/repos/{repo}/contents/{path}", {
            "message": message, "branch": branch, "sha": sha,
            "content": base64.b64encode(content.encode()).decode(),
            "committer": {"name": "HearmemanAI", "email": "103748831+Hearmeman24@users.noreply.github.com"},
        })

    def create_pr(self, repo: str, head: str, base: str, title: str, body: str) -> str:
        return self._req("POST", f"/repos/{repo}/pulls",
                         {"title": title, "head": head, "base": base, "body": body})["html_url"]


def rewrite_pins(text: str, new_image: str) -> str:
    pins = json.loads(text)
    old = pins.get("base_image")
    if not old:
        raise FanoutError("pins.json has no base_image")
    return text.replace(f'"{old}"', f'"{new_image}"', 1)


def rewrite_dockerfile(text: str, new_image: str) -> str:
    new, n = re.subn(r"(?m)^ARG BASE_IMAGE=\S+$", f"ARG BASE_IMAGE={new_image}", text, count=1)
    if n != 1:
        raise FanoutError("Dockerfile has no ARG BASE_IMAGE= line")
    return new


def plan_one(gh: GitHub, tpl: dict, data: dict, *, dry_run: bool) -> Result:
    repo, base = tpl["repo"], tpl["branch"]
    new_image = versions.image(data, data["template_variant"])
    lbl = versions.label(data)
    head = f"auto/comfyui-{lbl}"
    got = gh.read_file(repo, "pins.json", base)
    if got is None:
        return Result(repo, "error", f"no pins.json on {base}")
    pins_text, pins_sha = got
    pins = json.loads(pins_text)
    hold = pins.get("hold")
    if hold is not None:
        reason = hold.get("reason") if isinstance(hold, dict) else None
        if not reason:
            return Result(repo, "error", 'pins.json "hold" must be an object with a non-empty "reason"')
        return Result(repo, "held", f"{reason} (stays on {pins.get('base_image')}, runtime_ref {pins.get('runtime_ref')})")
    if pins.get("base_image") == new_image:
        return Result(repo, "current", new_image)
    existing = gh.open_pr(repo, head)
    if existing:
        return Result(repo, "open", existing)
    df = gh.read_file(repo, "Dockerfile", base)
    if df is None:
        return Result(repo, "error", f"no Dockerfile on {base}")
    df_text, df_sha = df
    new_pins = rewrite_pins(pins_text, new_image)
    new_df = rewrite_dockerfile(df_text, new_image)
    if dry_run:
        return Result(repo, "planned", f"{pins.get('base_image')} -> {new_image} on {head}")
    msg = f"Move to ComfyUI {data['comfyui']['release']} base image"
    gh.create_branch(repo, head, gh.branch_sha(repo, base))
    gh.write_file(repo, "pins.json", head, new_pins, pins_sha, msg)
    gh.write_file(repo, "Dockerfile", head, new_df, df_sha, msg)
    body = (f"Automated fan-out from comfyui-runtime after the base image was published.\n\n"
            f"- `pins.json` `base_image` and `Dockerfile` `ARG BASE_IMAGE`: `{pins.get('base_image')}` -> `{new_image}`\n"
            f"- ComfyUI {data['comfyui']['release']} (`{data['comfyui']['sha']}`)\n"
            f"- `runtime_ref` unchanged (`{pins.get('runtime_ref')}`).\n\n"
            f"Review, let CI pass, merge, then tag the next `vN` to build and promote the image.\n"
            f"To hold this template back instead, close this PR and add "
            f'`"hold": {{"reason": "..."}}` to `pins.json`.')
    return Result(repo, "opened", gh.create_pr(repo, head, base, msg, body))


def run(gh: GitHub, data: dict, *, dry_run: bool) -> list[Result]:
    results = []
    for tpl in data["templates"]:
        try:
            results.append(plan_one(gh, tpl, data, dry_run=dry_run))
        except FanoutError as exc:
            results.append(Result(tpl["repo"], "error", str(exc)))
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    token = os.environ.get("GITHUB_TOKEN")
    if not args.dry_run and not token:
        print("FATAL: GITHUB_TOKEN is required unless --dry-run", file=sys.stderr)
        return 2
    data = versions.load()
    print(f"fan-out for {versions.image(data, data['template_variant'])}{' (dry run)' if args.dry_run else ''}")
    results = run(GitHub(token), data, dry_run=args.dry_run)
    for r in results:
        print(f"{r.status.upper():8} {r.repo}: {r.detail}")
    held = [r.repo for r in results if r.status == "held"]
    if held:
        print(f"HELD (skipped): {', '.join(held)}")
    return 1 if any(r.status == "error" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
