import logging
import os
import sys
import types
import unittest
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

import backend


class GpuStatusTests(unittest.TestCase):
    def make_torch(self, *, available=True, total_memory=4 * 1024**3, free_memory=3 * 1024**3):
        torch = types.ModuleType("torch")
        torch.cuda = Mock()
        torch.cuda.is_available.return_value = available
        torch.cuda.current_device.return_value = 0
        torch.cuda.get_device_properties.return_value = types.SimpleNamespace(
            name="Test NVIDIA GPU",
            total_memory=total_memory,
        )
        torch.cuda.mem_get_info.return_value = (free_memory, total_memory)
        return torch

    def test_gpu_status_requires_verified_name_and_memory(self):
        torch = self.make_torch()
        with patch("backend.importlib.util.find_spec", return_value=object()), patch.dict(
            sys.modules, {"torch": torch}
        ):
            status = backend._gpu_status()

        self.assertTrue(status["cuda_ready"])
        self.assertEqual(status["gpu_name"], "Test NVIDIA GPU")
        self.assertEqual(status["total_vram_gb"], 4.0)
        self.assertEqual(status["free_vram_gb"], 3.0)

    def test_gpu_status_does_not_report_ready_when_memory_probe_fails(self):
        torch = self.make_torch()
        torch.cuda.get_device_properties.side_effect = RuntimeError("device query failed")
        with patch("backend.importlib.util.find_spec", return_value=object()), patch.dict(
            sys.modules, {"torch": torch}
        ), self.assertLogs(backend.LOGGER, level=logging.ERROR):
            status = backend._gpu_status()

        self.assertFalse(status["cuda_ready"])
        self.assertIsNone(status["gpu_name"])
        self.assertIsNone(status["total_vram_gb"])
        self.assertIn("device query failed", status["message"])

    def test_render_runtime_never_uses_server_gpu_as_visitor_gpu(self):
        with patch.dict(os.environ, {"RENDER": "true"}), patch(
            "backend._gpu_status"
        ) as gpu_probe:
            runtime = backend._training_runtime()

        gpu_probe.assert_not_called()
        self.assertTrue(runtime["render_hosted"])
        self.assertFalse(runtime["cuda_available"])
        self.assertFalse(runtime["training_ready"])
        self.assertIsNone(runtime["gpu_name"])
        self.assertEqual(runtime["gpu_memory_bytes"], 0)
        with patch.dict(os.environ, {"RENDER": "true"}):
            response = TestClient(backend.app).post(
                "/api/training/start",
                json={"dataset_upload_id": "dataset", "model_upload_id": "model"},
            )
        self.assertEqual(response.status_code, 503)
        self.assertIn("cannot access your computer's GPU", response.json()["detail"])

    def test_local_backend_cors_allows_only_configured_render_site(self):
        application = FastAPI()
        backend._configure_web_cors(application, ["https://automata.example"])

        @application.get("/api/health")
        def health():
            return {"status": "ok"}

        client = TestClient(application)
        origin = "https://automata.example"
        preflight = client.options(
            "/api/health",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Private-Network": "true",
            },
        )
        self.assertEqual(preflight.status_code, 200)
        self.assertEqual(preflight.headers["access-control-allow-origin"], origin)
        self.assertEqual(preflight.headers["access-control-allow-private-network"], "true")

        denied = client.get("/api/health", headers={"Origin": "https://untrusted.example"})
        self.assertNotIn("access-control-allow-origin", denied.headers)

    def test_local_backend_cors_rejects_non_https_origins(self):
        with self.assertRaises(RuntimeError):
            backend._configure_web_cors(FastAPI(), ["http://automata.example"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
