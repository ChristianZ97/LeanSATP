"""Tests for LeanSATP server device fallback behavior."""

from __future__ import annotations

import errno
import json
import sys
import warnings
import unittest
from contextlib import redirect_stderr, redirect_stdout
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

    def test_full_proof_trace_is_pretty_printed(self) -> None:
        def fake_load_policy(_checkpoint: str, _cache: str, device: str | None = None):
            return (f"{device}-model", device, False)

        tactic = "aesop (config := { maxRuleApplications := 42 })"
        stderr = StringIO()
        with (
            patch.object(service, "preferred_device", return_value="cpu"),
            patch.object(service, "load_policy", side_effect=fake_load_policy),
            patch.object(service, "policy_tactic", return_value=tactic),
            redirect_stderr(stderr),
        ):
            engine = service.SATPInferenceEngine("checkpoint", "cache")
            result = engine.infer(goal="True")

        output = stderr.getvalue()
        self.assertEqual(result["tactic"], tactic)
        self.assertIn("INFO     [LeanSATP] Full proof (cpu):", output)
        self.assertIn("    theorem satp_goal", output)
        self.assertIn("      : True := by", output)
        self.assertIn("      aesop (config := { maxRuleApplications := 42 })", output)

    def test_full_proof_trace_can_be_colorized(self) -> None:
        stream = StringIO()
        service.print_full_proof_trace(
            "theorem satp_goal\n  : True := by",
            "aesop (config := { maxRuleApplications := 42 })",
            device="cpu",
            stream=stream,
            enable_color=True,
        )

        trace = stream.getvalue()
        self.assertIn("\033[", trace)
        self.assertIn("[LeanSATP] Full proof", trace)
        self.assertIn("theorem", trace)
        self.assertIn("aesop", trace)


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

        stderr = StringIO()
        with (
            patch.object(service, "preferred_device", return_value="cuda"),
            patch.object(service, "load_policy", side_effect=fake_load_policy),
            patch.object(service, "policy_tactic", side_effect=fake_policy_tactic),
            redirect_stderr(stderr),
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
        self.assertIn("INFO     → request POST /infer", stderr.getvalue())
        self.assertIn("INFO     ← response 200 POST /infer", stderr.getvalue())


class SATPServerLifecycleTests(unittest.TestCase):
    def test_suppress_startup_noise_hides_library_chatter(self) -> None:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stderr(stderr), redirect_stdout(stdout):
            with service._suppress_startup_noise():
                print("library stdout noise")
                print("library stderr noise", file=sys.stderr)
                warnings.warn("Found GPU0 incompatible CUDA capability", UserWarning)

        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "")

    def test_serve_shutdown_is_graceful_on_keyboard_interrupt(self) -> None:
        class FakeEngine:
            preferred_device = "cuda"
            active_device = "cpu"
            retrieval_enabled = False

            def summary(self) -> str:
                return (
                    "SATP inference engine initialized with: "
                    "PREFERRED_DEVICE=[bold]cuda[/bold], "
                    "ACTIVE_DEVICE=[bold]cpu[/bold], "
                    "RETRIEVAL=[bold]disabled[/bold]"
                )

        fake_engine = FakeEngine()
        signal_calls: list[tuple[int, object]] = []
        server_events: list[str] = []

        class FakeServer:
            engine = None

            def server_bind(self) -> None:
                server_events.append("bound")

            def server_activate(self) -> None:
                server_events.append("activated")

            def serve_forever(self) -> None:
                raise KeyboardInterrupt

            def server_close(self) -> None:
                server_events.append("closed")

        stderr = StringIO()
        with (
            patch.object(service, "SATPInferenceEngine", return_value=fake_engine),
            patch.object(service, "_SATPHTTPServer", return_value=FakeServer()),
            patch.object(
                service.signal, "getsignal", side_effect=["old-int", "old-term"]
            ),
            patch.object(
                service.signal,
                "signal",
                side_effect=lambda sig, handler: signal_calls.append((sig, handler)),
            ),
            redirect_stderr(stderr),
        ):
            exit_code = service.serve(host="127.0.0.1", port=5177)

        output = stderr.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("INFO     Started server process [", output)
        self.assertIn("INFO     Waiting for application startup.", output)
        self.assertIn(
            "SATP inference engine initialized with:",
            output,
        )
        self.assertIn("INFO     Application startup complete.", output)
        self.assertIn("LeanSATP Server", output)
        self.assertIn("http://127.0.0.1:5177", output)
        self.assertIn("INFO     Try me with:", output)
        self.assertIn("curl --request POST \\", output)
        self.assertIn("--url http://localhost:5177/infer \\", output)
        self.assertIn('--data \'{"goal":"True"}\' | jq', output)
        self.assertNotIn("tactic_name", output)
        self.assertIn("INFO     Shutting down", output)
        self.assertIn("INFO     Waiting for application shutdown.", output)
        self.assertIn("INFO     Application shutdown complete.", output)
        self.assertIn("INFO     Finished server process [", output)
        self.assertEqual(server_events, ["bound", "activated", "closed"])
        self.assertEqual(signal_calls[0][0], service.signal.SIGINT)
        self.assertEqual(signal_calls[1][0], service.signal.SIGTERM)
        self.assertEqual(signal_calls[2], (service.signal.SIGINT, "old-int"))
        self.assertEqual(signal_calls[3], (service.signal.SIGTERM, "old-term"))

    def test_serve_port_in_use_logs_help_and_exits_cleanly(self) -> None:
        server_events: list[str] = []

        class FakeServer:
            engine = None

            def server_bind(self) -> None:
                server_events.append("bind")
                raise OSError(errno.EADDRINUSE, "Address already in use")

            def server_close(self) -> None:
                server_events.append("closed")

        stderr = StringIO()
        with (
            patch.object(service, "_SATPHTTPServer", return_value=FakeServer()),
            patch.object(service, "SATPInferenceEngine") as mock_engine,
            redirect_stderr(stderr),
        ):
            exit_code = service.serve(host="127.0.0.1", port=5177)

        output = stderr.getvalue()
        self.assertEqual(exit_code, 1)
        mock_engine.assert_not_called()
        self.assertIn("INFO     Started server process [", output)
        self.assertIn("INFO     Waiting for application startup.", output)
        self.assertIn(
            "WARNING  [LeanSATP] Port 5177 is already in use on 127.0.0.1.",
            output,
        )
        self.assertIn(
            "WARNING  [LeanSATP] Another process is already listening on the SATP server port.",
            output,
        )
        self.assertIn(
            "WARNING  [LeanSATP] If that is an existing SATP server, reuse it instead of starting a second copy.",
            output,
        )
        self.assertIn(
            "INFO     [LeanSATP] To inspect the current listener: lsof -i :5177",
            output,
        )
        self.assertIn(
            "INFO     [LeanSATP] To stop it and restart SATP: fuser -k 5177/tcp",
            output,
        )
        self.assertIn("INFO     Waiting for application shutdown.", output)
        self.assertIn("INFO     Application shutdown complete.", output)
        self.assertIn("INFO     Finished server process [", output)
        self.assertNotIn("Traceback", output)
        self.assertEqual(server_events, ["bind", "closed"])

    def test_serve_missing_checkpoint_logs_help_and_exits_cleanly(self) -> None:
        server_events: list[str] = []

        class FakeServer:
            engine = None

            def server_bind(self) -> None:
                server_events.append("bound")

            def server_close(self) -> None:
                server_events.append("closed")

        stderr = StringIO()
        with (
            patch.object(service, "_SATPHTTPServer", return_value=FakeServer()),
            patch.object(
                service,
                "SATPInferenceEngine",
                side_effect=FileNotFoundError(
                    "checkpoint not found: /tmp/cache/best_checkpoint.pt; "
                    "run ./setup.sh to download it"
                ),
            ),
            redirect_stderr(stderr),
        ):
            exit_code = service.serve(host="127.0.0.1", port=5177)

        output = stderr.getvalue()
        self.assertEqual(exit_code, 1)
        self.assertIn("INFO     Started server process [", output)
        self.assertIn("INFO     Waiting for application startup.", output)
        self.assertIn(
            "ERROR    [LeanSATP] checkpoint not found: /tmp/cache/best_checkpoint.pt; run ./setup.sh to download it",
            output,
        )
        self.assertIn(
            "INFO     [LeanSATP] Run ./setup.sh from the repository root to install dependencies, fetch mathlib, and download the checkpoint.",
            output,
        )
        self.assertIn("INFO     Waiting for application shutdown.", output)
        self.assertIn("INFO     Application shutdown complete.", output)
        self.assertIn("INFO     Finished server process [", output)
        self.assertNotIn("Traceback", output)
        self.assertEqual(server_events, ["bound", "closed"])


if __name__ == "__main__":
    unittest.main()
