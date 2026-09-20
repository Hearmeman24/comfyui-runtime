#!/usr/bin/env python3
"""Behavior tests for atomic RunPod template image promotion."""

from __future__ import annotations

import unittest

from update_runpod_template import PromotionError, Template, parse_template_ids, promote


class FakeClient:
    def __init__(self, images: dict[str, str], *, mismatch_on: str | None = None) -> None:
        self.images = dict(images)
        self.names = {key: f"Template {key}" for key in images}
        self.mismatch_on = mismatch_on
        self.writes: list[tuple[str, str]] = []

    def get_template(self, template_id: str) -> Template:
        image = self.images[template_id]
        if self.mismatch_on == template_id and self.writes:
            image = "hearmeman/comfyui-qwen-template:v999"
            self.mismatch_on = None
        return Template(template_id, self.names[template_id], image)

    def set_image(self, template_id: str, image_name: str) -> None:
        self.writes.append((template_id, image_name))
        self.images[template_id] = image_name


class PromotionTests(unittest.TestCase):
    def test_updates_every_allowlisted_template(self) -> None:
        client = FakeClient({"aaaaaaaaaa": "hearmeman/comfyui-qwen-template:v1", "bbbbbbbbbb": "hearmeman/comfyui-qwen-template:v2"})
        messages: list[str] = []
        promote(client, client.images, "hearmeman/comfyui-qwen-template:v3", "hearmeman/comfyui-qwen-template", emit=messages.append)
        self.assertEqual(set(client.images.values()), {"hearmeman/comfyui-qwen-template:v3"})
        self.assertEqual(len(client.writes), 2)
        self.assertEqual(len(messages), 2)

    def test_is_idempotent(self) -> None:
        client = FakeClient({"aaaaaaaaaa": "hearmeman/comfyui-qwen-template:v3"})
        promote(client, client.images, "hearmeman/comfyui-qwen-template:v3", "hearmeman/comfyui-qwen-template", emit=lambda _: None)
        self.assertEqual(client.writes, [])

    def test_rejects_wrong_existing_repository_before_any_write(self) -> None:
        client = FakeClient({"aaaaaaaaaa": "someone/else:v1"})
        with self.assertRaises(PromotionError):
            promote(client, client.images, "hearmeman/comfyui-qwen-template:v3", "hearmeman/comfyui-qwen-template", emit=lambda _: None)
        self.assertEqual(client.writes, [])

    def test_verification_failure_rolls_back_all_changed_templates(self) -> None:
        original = {
            "aaaaaaaaaa": "hearmeman/comfyui-qwen-template:v1",
            "bbbbbbbbbb": "hearmeman/comfyui-qwen-template:v2",
        }
        client = FakeClient(original, mismatch_on="bbbbbbbbbb")
        with self.assertRaises(PromotionError):
            promote(client, client.images, "hearmeman/comfyui-qwen-template:v3", "hearmeman/comfyui-qwen-template", emit=lambda _: None)
        self.assertEqual(client.images, original)

    def test_requires_immutable_version_tag(self) -> None:
        client = FakeClient({"aaaaaaaaaa": "hearmeman/comfyui-qwen-template:v1"})
        with self.assertRaises(PromotionError):
            promote(client, client.images, "hearmeman/comfyui-qwen-template:latest", "hearmeman/comfyui-qwen-template", emit=lambda _: None)
        self.assertEqual(client.writes, [])

    def test_template_id_parser_rejects_duplicates_and_bad_ids(self) -> None:
        self.assertEqual(parse_template_ids("aaaaaaaaaa, bbbbbbbbbb"), ["aaaaaaaaaa", "bbbbbbbbbb"])
        for value in ("", "aaaaaaaaaa,aaaaaaaaaa", "not-an-id"):
            with self.subTest(value=value), self.assertRaises(PromotionError):
                parse_template_ids(value)


if __name__ == "__main__":
    unittest.main()
