#!/usr/bin/env python3
"""Release watcher: tag peeling and version ordering, no network."""

from __future__ import annotations

import unittest

import comfyui_release_watch as w


class Watch(unittest.TestCase):
    def test_peels_annotated_tag(self) -> None:
        responses = {
            "/repos/comfyanonymous/ComfyUI/releases/latest": {"tag_name": "v0.43.0"},
            "/repos/comfyanonymous/ComfyUI/git/ref/tags/v0.43.0": {"object": {"type": "tag", "sha": "t1"}},
            "/repos/comfyanonymous/ComfyUI/git/tags/t1": {"object": {"type": "commit", "sha": "c" * 40}},
        }
        self.assertEqual(w.latest(responses.__getitem__), ("v0.43.0", "c" * 40))

    def test_lightweight_tag(self) -> None:
        responses = {
            "/repos/comfyanonymous/ComfyUI/releases/latest": {"tag_name": "v0.42.1"},
            "/repos/comfyanonymous/ComfyUI/git/ref/tags/v0.42.1": {"object": {"type": "commit", "sha": "d" * 40}},
        }
        self.assertEqual(w.latest(responses.__getitem__), ("v0.42.1", "d" * 40))

    def test_ordering_is_numeric(self) -> None:
        self.assertTrue(w.is_newer("v0.42.10", "v0.42.9"))
        self.assertTrue(w.is_newer("v1.0.0", "v0.99.99"))
        self.assertFalse(w.is_newer("v0.42.1", "v0.42.1"))
        self.assertFalse(w.is_newer("v0.41.3", "v0.42.1"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
