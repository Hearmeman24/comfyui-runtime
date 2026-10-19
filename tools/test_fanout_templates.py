#!/usr/bin/env python3
"""Fan-out dry runs against a fake GitHub, including per-template holds."""

from __future__ import annotations

import json
import unittest

import fanout_templates as f
import versions

OLD = "hearmeman/comfyui-base:cu130-comfy0.36.0-3dd559d8-torch2.11.0"
DOCKERFILE = "# syntax=docker/dockerfile:1\nARG BASE_IMAGE=%s\nFROM ${BASE_IMAGE}\n"


class FakeGitHub:
    def __init__(self, files: dict, open_prs: dict | None = None) -> None:
        self.files = files  # (repo, path) -> text
        self.open_prs = open_prs or {}
        self.writes: list = []
        self.prs: list = []
        self.branches: list = []

    def read_file(self, repo, path, ref):
        text = self.files.get((repo, path))
        return (text, "sha-" + path) if text is not None else None

    def open_pr(self, repo, head):
        return self.open_prs.get((repo, head))

    def branch_sha(self, repo, branch):
        return "base-sha"

    def create_branch(self, repo, name, sha):
        self.branches.append((repo, name))

    def write_file(self, repo, path, branch, content, sha, message):
        self.writes.append((repo, path, branch, content))

    def create_pr(self, repo, head, base, title, body):
        self.prs.append((repo, head, base, title, body))
        return f"https://github.com/{repo}/pull/1"


def pins(base=OLD, **extra):
    return json.dumps({"runtime_ref": "stable", "base_image": base, **extra}, indent=2) + "\n"


class Fanout(unittest.TestCase):
    def setUp(self) -> None:
        self.data = versions.load()
        self.data["templates"] = [
            {"repo": "o/qwen", "branch": "master"},
            {"repo": "o/minimax", "branch": "master"},
            {"repo": "o/wan", "branch": "master"},
        ]
        self.new = versions.image(self.data, self.data["template_variant"])
        self.files = {}
        for name in ("qwen", "minimax", "wan"):
            self.files[(f"o/{name}", "pins.json")] = pins()
            self.files[(f"o/{name}", "Dockerfile")] = DOCKERFILE % OLD
        # MiniMax held back on an older ComfyUI + pinned runtime (the override case)
        self.files[("o/minimax", "pins.json")] = pins(runtime_ref="v4", hold={"reason": "AI-Toolkit breaks on new ComfyUI"})
        self.files[("o/wan", "pins.json")] = pins(base=self.new)

    def results(self, gh, dry_run):
        return {r.repo: r for r in f.run(gh, self.data, dry_run=dry_run)}

    def test_dry_run_plans_skips_held_and_writes_nothing(self) -> None:
        gh = FakeGitHub(self.files)
        r = self.results(gh, dry_run=True)
        self.assertEqual(r["o/qwen"].status, "planned")
        self.assertEqual(r["o/minimax"].status, "held")
        self.assertIn("AI-Toolkit", r["o/minimax"].detail)
        self.assertIn("runtime_ref v4", r["o/minimax"].detail)
        self.assertEqual(r["o/wan"].status, "current")
        self.assertEqual((gh.writes, gh.prs, gh.branches), ([], [], []))

    def test_live_run_touches_only_unheld_outdated_templates(self) -> None:
        gh = FakeGitHub(self.files)
        r = self.results(gh, dry_run=False)
        self.assertEqual(r["o/qwen"].status, "opened")
        self.assertEqual({w[0] for w in gh.writes}, {"o/qwen"})
        written = {w[1]: w[3] for w in gh.writes}
        self.assertEqual(json.loads(written["pins.json"])["base_image"], self.new)
        self.assertEqual(json.loads(written["pins.json"])["runtime_ref"], "stable")
        self.assertIn(f"ARG BASE_IMAGE={self.new}\n", written["Dockerfile"])
        self.assertEqual(gh.prs[0][1], f"auto/comfyui-{versions.label(self.data)}")

    def test_existing_pr_is_not_duplicated(self) -> None:
        head = f"auto/comfyui-{versions.label(self.data)}"
        gh = FakeGitHub(self.files, {("o/qwen", head): "https://github.com/o/qwen/pull/7"})
        r = self.results(gh, dry_run=False)
        self.assertEqual(r["o/qwen"].status, "open")
        self.assertEqual(gh.writes, [])

    def test_hold_without_reason_is_an_error_not_a_silent_skip(self) -> None:
        self.files[("o/minimax", "pins.json")] = pins(hold=True)
        r = self.results(FakeGitHub(self.files), dry_run=True)
        self.assertEqual(r["o/minimax"].status, "error")

    def test_missing_dockerfile_arg_is_reported(self) -> None:
        self.files[("o/qwen", "Dockerfile")] = "FROM scratch\n"
        r = self.results(FakeGitHub(self.files), dry_run=True)
        self.assertEqual(r["o/qwen"].status, "error")

    def test_pins_rewrite_preserves_other_keys(self) -> None:
        text = pins(extra_key="x")
        out = json.loads(f.rewrite_pins(text, self.new))
        self.assertEqual(out, {"runtime_ref": "stable", "base_image": self.new, "extra_key": "x"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
