#!/usr/bin/env python3
"""versions.json is the single source of truth: derivation and drift checks."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import versions

ROOT = Path(__file__).resolve().parent.parent


class Versions(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "base").mkdir()
        shutil.copy(ROOT / "base" / "Dockerfile", self.tmp / "base" / "Dockerfile")
        shutil.copytree(ROOT / "torch", self.tmp / "torch")
        shutil.copy(ROOT / "versions.json", self.tmp / "versions.json")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_repo_is_coherent(self) -> None:
        self.assertEqual(versions.check(versions.load()), [])

    def test_tag_matches_legacy_grammar(self) -> None:
        data = versions.load()
        sha = data["comfyui"]["sha"]
        rel = data["comfyui"]["release"].lstrip("v")
        self.assertEqual(versions.tag(data, "cu130"), f"cu130-comfy{rel}-{sha[:8]}-torch{data['torch']}")
        self.assertTrue(versions.image(data, "cu128").startswith("hearmeman/comfyui-base:cu128-comfy"))

    def test_build_args_carry_every_pin(self) -> None:
        data = versions.load()
        args = versions.build_args(data, "cu128")
        self.assertIn(f"--build-arg=COMFYUI_REF={data['comfyui']['sha']}", args)
        self.assertIn(f"--build-arg=CUDA_BASE_IMAGE={data['cuda_variants']['cu128']}", args)
        with self.assertRaises(versions.VersionsError):
            versions.build_args(data, "cu999")

    def test_drift_is_caught_and_sync_fixes_it(self) -> None:
        path = self.tmp / "versions.json"
        data = versions.set_release("v9.9.9", "a" * 40, path)
        problems = versions.check(data, self.tmp)
        self.assertEqual(len(problems), 2, problems)
        self.assertTrue(versions.sync(data, self.tmp))
        self.assertEqual(versions.check(data, self.tmp), [])
        text = (self.tmp / "base" / "Dockerfile").read_text()
        self.assertIn("ARG COMFYUI_IMAGE_REF=9.9.9-aaaaaaaa", text)
        self.assertIn("# Latest stable release v9.9.9, resolved", text)

    def test_torch_drift_is_caught(self) -> None:
        raw = json.loads((self.tmp / "versions.json").read_text())
        raw["torch"] = "9.0.0"
        (self.tmp / "versions.json").write_text(json.dumps(raw))
        data = versions.load(self.tmp / "versions.json")
        self.assertTrue(any("torch/cu130.txt" in p for p in versions.check(data, self.tmp)))

    def test_rejects_short_sha(self) -> None:
        path = self.tmp / "versions.json"
        raw = json.loads(path.read_text())
        raw["comfyui"]["sha"] = "63be2f9b"
        path.write_text(json.dumps(raw))
        with self.assertRaises(versions.VersionsError):
            versions.load(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
