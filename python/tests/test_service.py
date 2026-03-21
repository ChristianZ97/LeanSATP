"""Tests for LeanSATP service device fallback behavior."""

from __future__ import annotations

import json
import unittest
from contextlib import redirect_stderr
from io import BytesIO, StringIO
from unittest.mock import patch

from leansatp_runtime import service


class _DummyServer:
    def __init__(self, engine: service.SATPInferenceEngine):
        self.engine = engine


class _TestRequestHandler(service._SATPRequestHandler):
    def __init__(self, request_bytes: bytes, engine: service.SATPInferenceEngine):
        self._request_bytes = request_bytes
        self.response = BytesIO()
        super().__init__(
            request=None,
            client_address=("127.0.0.1", 0),
            server=_DummyServer(engine),
        )

    def setup(self) -> None:
        self.rfile = BytesIO(self._request_bytes)
        self.wfile = self.response

    def finish(self) -> None:
        return


class SATPInferenceEngineTests(unittest.TestCase):
    def test_startup_cuda_failure_falls_back_to_cpu(self) -> None:
        calls: list[str] = []

        def fake_load_policy(_checkpoint: str, _cache: str, device: str | None = None):
            calls.append(device or "auto")
            if device == "cuda":
                raise RuntimeError(
                    "CUDA error: no kernel image is available for execution on the device"
                )
            return ("cpu-model", "cpu", False)

        stderr = StringIO()
        with (
            patch.object(service, "preferred_device", return_value="cuda"),
            patch.object(service, "load_policy", side_effect=fake_load_policy),
            redirect_stderr(stderr),
        ):
            engine = service.SATPInferenceEngine("checkpoint", "cache")

        self.assertEqual(calls, ["cuda", "cpu"])
        self.assertEqual(engine.preferred_device, "cuda")
        self.assertEqual(engine.active_device, "cpu")
        self.assertEqual(engine.model_and_device, ("cpu-model", "cpu", False))
        self.assertIn("falling back from cuda to cpu during startup", stderr.getvalue())

    def test_inference_cuda_failure_retries_once_on_cpu(self) -> None:
        load_calls: list[str] = []
        tactic_devices: list[str] = []

        def fake_load_policy(_checkpoint: str, _cache: str, device: str | None = None):
            load_calls.append(device or "auto")
            return (f"{device}-model", device, False)

        def fake_policy_tactic(model_and_device, *_args, **_kwargs):
            tactic_devices.append(model_and_device[1])
            if model_and_device[1] == "cuda":
                raise RuntimeError(
                    "CUDA error: no kernel image is available for execution on the device"
                )
            return "aesop"

        stderr = StringIO()
        with (
            patch.object(service, "preferred_device", return_value="cuda"),
            patch.object(service, "load_policy", side_effect=fake_load_policy),
            patch.object(service, "policy_tactic", side_effect=fake_policy_tactic),
            redirect_stderr(stderr),
        ):
            engine = service.SATPInferenceEngine("checkpoint", "cache")
            result = engine.infer(goal="True")

        self.assertEqual(result["tactic"], "aesop")
        self.assertEqual(load_calls, ["cuda", "cpu"])
        self.assertEqual(tactic_devices, ["cuda", "cpu"])
        self.assertEqual(engine.active_device, "cpu")
        self.assertIn(
            "falling back from cuda to cpu during inference", stderr.getvalue()
        )

    def test_cpu_downgrade_is_sticky_for_later_requests(self) -> None:
        load_calls: list[str] = []
        tactic_devices: list[str] = []

        def fake_load_policy(_checkpoint: str, _cache: str, device: str | None = None):
            load_calls.append(device or "auto")
            return (f"{device}-model", device, False)

        first_cuda_failure = True

        def fake_policy_tactic(model_and_device, *_args, **_kwargs):
            nonlocal first_cuda_failure
            tactic_devices.append(model_and_device[1])
            if model_and_device[1] == "cuda" and first_cuda_failure:
                first_cuda_failure = False
                raise RuntimeError("CUDA error: no kernel image is available")
            return "aesop"

        with (
            patch.object(service, "preferred_device", return_value="cuda"),
            patch.object(service, "load_policy", side_effect=fake_load_policy),
            patch.object(service, "policy_tactic", side_effect=fake_policy_tactic),
        ):
            engine = service.SATPInferenceEngine("checkpoint", "cache")
            first = engine.infer(goal="True")
            second = engine.infer(goal="True")

        self.assertEqual(first["tactic"], "aesop")
        self.assertEqual(second["tactic"], "aesop")
        self.assertEqual(load_calls, ["cuda", "cpu"])
        self.assertEqual(tactic_devices, ["cuda", "cpu", "cpu"])
        self.assertEqual(engine.active_device, "cpu")

    def test_non_cuda_exception_does_not_trigger_fallback(self) -> None:
        load_calls: list[str] = []

        def fake_load_policy(_checkpoint: str, _cache: str, device: str | None = None):
            load_calls.append(device or "auto")
            return (f"{device}-model", device, False)

        with (
            patch.object(service, "preferred_device", return_value="cuda"),
            patch.object(service, "load_policy", side_effect=fake_load_policy),
            patch.object(
                service,
                "policy_tactic",
                side_effect=ValueError("invalid theorem state"),
            ),
        ):
            engine = service.SATPInferenceEngine("checkpoint", "cache")
            with self.assertRaisesRegex(ValueError, "invalid theorem state"):
                engine.infer(goal="True")

        self.assertEqual(load_calls, ["cuda"])
        self.assertEqual(engine.active_device, "cuda")


class SATPHTTPServerSmokeTests(unittest.TestCase):
    def test_http_infer_smoke_uses_cpu_after_cuda_fallback(self) -> None:
        load_calls: list[str] = []

        def fake_load_policy(_checkpoint: str, _cache: str, device: str | None = None):
            load_calls.append(device or "auto")
            return (f"{device}-model", device, False)

        def fake_policy_tactic(model_and_device, *_args, **_kwargs):
            if model_and_device[1] == "cuda":
                raise RuntimeError("CUDA error: no kernel image is available")
            return "aesop"

        with (
            patch.object(service, "preferred_device", return_value="cuda"),
            patch.object(service, "load_policy", side_effect=fake_load_policy),
            patch.object(service, "policy_tactic", side_effect=fake_policy_tactic),
        ):
            engine = service.SATPInferenceEngine("checkpoint", "cache")
            payload = json.dumps({"goal": "True"}).encode("utf-8")
            raw_request = (
                b"POST /infer HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\n"
                b"Content-Type: application/json\r\n"
                + f"Content-Length: {len(payload)}\r\n\r\n".encode("utf-8")
                + payload
            )
            handler = _TestRequestHandler(raw_request, engine)
            raw_response = handler.response.getvalue()
            body = json.loads(raw_response.split(b"\r\n\r\n", 1)[1].decode("utf-8"))

        self.assertTrue(body["ok"])
        self.assertEqual(body["tactic"], "aesop")
        self.assertEqual(engine.active_device, "cpu")
        self.assertEqual(load_calls, ["cuda", "cpu"])


if __name__ == "__main__":
    unittest.main()
