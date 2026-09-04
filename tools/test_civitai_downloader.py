#!/usr/bin/env python3
"""Stdlib regression suite for the vendored CivitAI downloader.

The production dependency is installed lazily by src/civitai_downloads.sh.
CI intentionally stays stdlib-only, so this suite supplies import-only stubs
when requests/urllib3 are unavailable and injects fake sessions for every test.
No network, CivitAI token, aria2 transfer, or model bytes are required.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
VENDOR = REPO / "vendor" / "civitai_downloader"
SOURCE = VENDOR / "download_with_aria.py"
EXPECTED_SOURCE_SHA256 = (
    "c48a41b4095f6d93dda58063ed6426a8ae00e297cb7bbebb64ae0d7cdf8dc1d6"
)
EXPECTED_REQUIREMENTS_SHA256 = (
    "0d0d2938e5443f318fc97363f42cffe3309df1b718cc2bf713263a68cb7bbee5"
)
SECRET = "sk-civitai-CANARY-DO-NOT-LOG-123456"


def install_dependency_stubs_if_needed() -> None:
    try:
        import requests  # noqa: F401
        from urllib3.util.retry import Retry  # noqa: F401

        return
    except ImportError:
        pass

    requests = types.ModuleType("requests")

    class RequestException(Exception):
        pass

    class Session:
        pass

    requests.RequestException = RequestException
    requests.Session = Session
    requests.Response = object

    adapters = types.ModuleType("requests.adapters")
    adapters.HTTPAdapter = object
    requests.adapters = adapters

    urllib3 = types.ModuleType("urllib3")
    urllib3_util = types.ModuleType("urllib3.util")
    retry_module = types.ModuleType("urllib3.util.retry")

    class Retry:
        def __init__(self, *args, **kwargs):
            pass

        def get_retry_after(self, response):
            return None

        def get_backoff_time(self):
            return 0

    retry_module.Retry = Retry
    urllib3.util = urllib3_util
    urllib3_util.retry = retry_module
    sys.modules.update(
        {
            "requests": requests,
            "requests.adapters": adapters,
            "urllib3": urllib3,
            "urllib3.util": urllib3_util,
            "urllib3.util.retry": retry_module,
        }
    )


def load_vendor():
    install_dependency_stubs_if_needed()
    spec = importlib.util.spec_from_file_location("vendored_civitai_downloader", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


dwa = load_vendor()


def resource(payload=b"verified model payload"):
    return dwa.ResolvedCivitAIResource(
        original="version:20",
        model_id="10",
        version_id="20",
        file_id="30",
        filename="model.safetensors",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest().upper(),
        file_type="Model",
        file_format="SafeTensor",
    )


class ProvenanceTests(unittest.TestCase):
    def test_snapshot_checksum_and_metadata_match(self):
        self.assertEqual(hashlib.sha256(SOURCE.read_bytes()).hexdigest(), EXPECTED_SOURCE_SHA256)
        checksums = (VENDOR / "SHA256SUMS").read_text(encoding="utf-8")
        self.assertIn(f"{EXPECTED_SOURCE_SHA256}  {SOURCE.name}", checksums)
        requirements = VENDOR / "requirements.txt"
        self.assertEqual(
            hashlib.sha256(requirements.read_bytes()).hexdigest(),
            EXPECTED_REQUIREMENTS_SHA256,
        )
        self.assertIn(
            f"{EXPECTED_REQUIREMENTS_SHA256}  {requirements.name}", checksums
        )
        provenance = (VENDOR / "UPSTREAM.md").read_text(encoding="utf-8")
        self.assertIn("fcba2f29e23f424984bba3f2654c59b9370ee905", provenance)
        self.assertEqual(
            requirements.read_text(encoding="utf-8").strip(),
            "requests==2.34.2",
        )
        self.assertIn("MIT", (VENDOR / "LICENSE-NOTICE.md").read_text(encoding="utf-8"))


class IdentifierAndLoggingTests(unittest.TestCase):
    def test_model_version_air_and_urls_are_preserved(self):
        cases = {
            "model:2834417": ("model", "2834417", None, None),
            "version:3268303": ("version", None, "3268303", None),
            "urn:air:minimaxh3:lora:civitai:2834417@3268303+3152083": (
                "air",
                "2834417",
                "3268303",
                "3152083",
            ),
            "https://civitai.com/api/download/models/3268303?fileId=3152083": (
                "version",
                None,
                "3268303",
                "3152083",
            ),
        }
        for original, expected in cases.items():
            with self.subTest(original=original):
                parsed = dwa.parse_civitai_reference(original)
                self.assertEqual(parsed.original, original)
                self.assertEqual(
                    (parsed.kind, parsed.model_id, parsed.version_id, parsed.file_id),
                    expected,
                )

    def test_legacy_cli_aliases_still_parse(self):
        args = dwa.build_parser().parse_args(["--model", "version:20", "-t", "token"])
        self.assertEqual(args.identifier, "version:20")
        self.assertEqual(args.token, "token")

    def test_success_is_exactly_two_structured_lines(self):
        stream = io.StringIO()
        logger = dwa.StructuredLogger(stream)
        payload = b"model"
        resolved = resource(payload)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / resolved.filename
            path.write_bytes(payload)
            downloader = dwa.CivitAIDownloader(
                "token", td, logger=logger, session=object()
            )
            downloader.resolve_identifier = lambda identifier: resolved
            downloader._download_resource = lambda *args, **kwargs: dwa.DownloadOutcome(
                path=path,
                artifacts=(path,),
                status="downloaded",
                size_bytes=len(payload),
                sha256=resolved.sha256,
            )
            downloader.download("version:20")

        lines = stream.getvalue().splitlines()
        self.assertEqual(len(lines), 2, stream.getvalue())
        self.assertTrue(lines[0].startswith("INFO resolve "), lines)
        self.assertTrue(lines[1].startswith("OK ready "), lines)

    def test_errors_stay_single_line_and_redacted(self):
        stream = io.StringIO()
        dwa.StructuredLogger(stream).error(
            "failure",
            stage="download",
            message=f"bad\nhttps://example.invalid/?token={SECRET}",
        )
        self.assertEqual(len(stream.getvalue().splitlines()), 1)
        self.assertNotIn(SECRET, stream.getvalue())


class SecurityAndRecoveryTests(unittest.TestCase):
    def test_token_aliases_are_read_from_environment(self):
        for name in ("CIVITAI_TOKEN", "civitai_token"):
            with self.subTest(name=name), mock.patch.dict(
                os.environ, {name: SECRET}, clear=True
            ):
                self.assertEqual(dwa.get_token(None), SECRET)

    def test_signed_url_is_not_in_aria_argv_or_child_environment(self):
        captured = {}

        class Sink:
            def __init__(self):
                self.value = ""
                self.closed = False

            def write(self, value):
                self.value += value

            def close(self):
                self.closed = True

        class FakeProcess:
            def __init__(self, cmd, **kwargs):
                captured["cmd"] = cmd
                captured["env"] = kwargs["env"]
                self.stdin = Sink()
                captured["stdin"] = self.stdin
                self.stdout = io.StringIO(
                    f"failed https://example.invalid/?Authorization={SECRET}\n"
                )

            def wait(self):
                return 1

        source_url = f"https://example.invalid/?Authorization={SECRET}"
        with (
            mock.patch.dict(
                os.environ,
                {"CIVITAI_TOKEN": SECRET, "civitai_token": SECRET},
                clear=True,
            ),
            mock.patch.object(dwa.subprocess, "Popen", FakeProcess),
        ):
            result = dwa.CivitAIDownloader(
                "token", tempfile.mkdtemp(), session=object()
            )._run_aria2c(["aria2c", "--input-file=-"], source_url)

        self.assertNotIn(source_url, captured["cmd"])
        self.assertNotIn("CIVITAI_TOKEN", captured["env"])
        self.assertNotIn("civitai_token", captured["env"])
        self.assertTrue(captured["stdin"].value.startswith(source_url + "\n"))
        self.assertNotIn(SECRET, " ".join(result.diagnostic_tail))

    def test_partial_with_control_file_resumes(self):
        payload = b"verified model payload"
        resolved = resource(payload)
        with tempfile.TemporaryDirectory() as td:
            output = Path(td)
            key = hashlib.sha256(resolved.filename.encode("utf-8")).hexdigest()[:12]
            staging = output / f".civitai-{resolved.file_id}-{key}.part"
            control = Path(f"{staging}{dwa.ARIA2_EXT}")
            staging.write_bytes(payload[:5])
            control.write_bytes(b"aria state")
            downloader = dwa.CivitAIDownloader(
                "token", output, session=object()
            )
            downloader._resolve_download_url = lambda url: "https://files.invalid/signed"
            captured = {}

            def fake_aria(cmd, source_url):
                captured["control_exists"] = control.exists()
                captured["partial_size"] = staging.stat().st_size
                staging.write_bytes(payload)
                control.unlink()
                return dwa.AriaResult(0, ())

            downloader._run_aria2c = fake_aria
            outcome = downloader._download_resource(resolved)

            self.assertTrue(captured["control_exists"])
            self.assertEqual(captured["partial_size"], 5)
            self.assertEqual(outcome.status, "resumed")
            self.assertEqual(outcome.path.read_bytes(), payload)


if __name__ == "__main__":
    unittest.main()
