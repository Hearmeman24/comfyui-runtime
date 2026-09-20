#!/usr/bin/env python3
"""Promote an immutable template image across an explicit RunPod allowlist.

Only ``imageName`` is patched.  The existing template is read before and after
the mutation, and every template already changed in this invocation is rolled
back if any update or verification fails.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Iterable


API_ROOT = "https://rest.runpod.io/v1"
IMAGE_RE = re.compile(
    r"^(?P<repo>[a-z0-9]+(?:[._/-][a-z0-9]+)*)"
    r":(?P<tag>v[0-9]+(?:\.[0-9]+)*)$"
)
TEMPLATE_ID_RE = re.compile(r"^[a-z0-9]{10}$")


class PromotionError(RuntimeError):
    """A safe promotion could not be completed."""


@dataclass(frozen=True)
class Template:
    id: str
    name: str
    image_name: str


class RunPodClient:
    def __init__(
        self,
        api_key: str,
        *,
        api_root: str = API_ROOT,
        opener: Callable[..., object] = urllib.request.urlopen,
    ) -> None:
        if not api_key:
            raise PromotionError("RUNPOD_API_KEY is required")
        self.api_key = api_key
        self.api_root = api_root.rstrip("/")
        self.opener = opener

    def _request(self, method: str, template_id: str, body: dict | None = None) -> dict:
        url = f"{self.api_root}/templates/{template_id}"
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "HearmemanAI-CircleCI-Template-Promoter/1.0",
            },
        )
        try:
            with self.opener(request, timeout=30) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            # Never echo the response body: a template response can contain env.
            raise PromotionError(
                f"RunPod {method} {template_id} returned HTTP {exc.code}"
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise PromotionError(
                f"RunPod {method} {template_id} failed: {exc.__class__.__name__}"
            ) from exc
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise PromotionError(
                f"RunPod {method} {template_id} returned invalid JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise PromotionError(
                f"RunPod {method} {template_id} returned a non-object response"
            )
        return payload

    def get_template(self, template_id: str) -> Template:
        payload = self._request("GET", template_id)
        try:
            result = Template(
                id=payload["id"],
                name=payload["name"],
                image_name=payload["imageName"],
            )
        except (KeyError, TypeError) as exc:
            raise PromotionError(
                f"RunPod GET {template_id} omitted required template fields"
            ) from exc
        if result.id != template_id:
            raise PromotionError(
                f"RunPod returned template {result.id!r} for requested ID {template_id!r}"
            )
        return result

    def set_image(self, template_id: str, image_name: str) -> None:
        self._request("PATCH", template_id, {"imageName": image_name})


def parse_template_ids(raw: str) -> list[str]:
    ids = [item.strip() for item in raw.split(",") if item.strip()]
    if not ids:
        raise PromotionError("RUNPOD_TEMPLATE_IDS must contain at least one template ID")
    if len(ids) != len(set(ids)):
        raise PromotionError("RUNPOD_TEMPLATE_IDS contains duplicates")
    invalid = [item for item in ids if not TEMPLATE_ID_RE.fullmatch(item)]
    if invalid:
        raise PromotionError(f"invalid RunPod template ID: {invalid[0]!r}")
    return ids


def image_repository(image_name: str) -> str:
    match = IMAGE_RE.fullmatch(image_name)
    if not match:
        raise PromotionError(
            f"image must be an immutable vN tag in owner/repository:vN form: {image_name!r}"
        )
    return match.group("repo")


def promote(
    client: RunPodClient,
    template_ids: Iterable[str],
    desired_image: str,
    expected_repository: str,
    *,
    emit: Callable[[str], None] = print,
) -> None:
    if image_repository(desired_image) != expected_repository:
        raise PromotionError(
            f"desired image repository does not match {expected_repository!r}"
        )

    snapshots = [client.get_template(template_id) for template_id in template_ids]
    for template in snapshots:
        current_repository = image_repository(template.image_name)
        if current_repository != expected_repository:
            raise PromotionError(
                f"template {template.id} ({template.name}) points at unexpected "
                f"repository {current_repository!r}"
            )

    changed: list[Template] = []
    try:
        for template in snapshots:
            if template.image_name == desired_image:
                emit(f"UNCHANGED {template.id} {template.name}: {desired_image}")
                continue
            client.set_image(template.id, desired_image)
            changed.append(template)
            actual = client.get_template(template.id).image_name
            if actual != desired_image:
                raise PromotionError(
                    f"template {template.id} verification returned {actual!r}, "
                    f"expected {desired_image!r}"
                )
            emit(
                f"UPDATED {template.id} {template.name}: "
                f"{template.image_name} -> {desired_image}"
            )
    except Exception as exc:
        rollback_failures: list[str] = []
        for template in reversed(changed):
            try:
                client.set_image(template.id, template.image_name)
                restored = client.get_template(template.id).image_name
                if restored != template.image_name:
                    raise PromotionError("rollback verification mismatch")
                emit(f"ROLLED BACK {template.id} {template.name}: {template.image_name}")
            except Exception:
                rollback_failures.append(template.id)
        suffix = ""
        if rollback_failures:
            suffix = f"; rollback failed for: {', '.join(rollback_failures)}"
        raise PromotionError(f"promotion failed: {exc}{suffix}") from exc


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--image", required=True, help="Immutable image, e.g. owner/repo:v12")
    result.add_argument("--expected-repository", required=True, help="Allowed owner/repository")
    result.add_argument(
        "--template-ids",
        default=os.environ.get("RUNPOD_TEMPLATE_IDS", ""),
        help="Comma-separated RunPod template IDs (default: RUNPOD_TEMPLATE_IDS)",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        client = RunPodClient(os.environ.get("RUNPOD_API_KEY", ""))
        promote(
            client,
            parse_template_ids(args.template_ids),
            args.image,
            args.expected_repository,
        )
    except PromotionError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
