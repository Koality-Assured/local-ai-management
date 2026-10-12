"""Unit tests for direct local inference chat module and CLI integration."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from local_ai.chat import (
    complete_chat,
    is_loopback_url,
    load_file_context,
    stream_chat,
)
from local_ai.cli import main


class TestChatModule(unittest.TestCase):
    """Test direct inference engine utilities."""

    def test_is_loopback_url(self) -> None:
        self.assertTrue(is_loopback_url("http://127.0.0.1:11434/v1"))
        self.assertTrue(is_loopback_url("http://localhost:8080/v1"))
        self.assertTrue(is_loopback_url("http://0.0.0.0:11434/v1"))
        self.assertTrue(is_loopback_url("http://[::1]:11434/v1"))

        self.assertFalse(is_loopback_url("http://192.168.1.100:11434/v1"))
        self.assertFalse(is_loopback_url("https://api.openai.com/v1"))
        self.assertFalse(is_loopback_url("https://example.com/api"))

    def test_load_file_context(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p1 = Path(td) / "test1.txt"
            p1.write_text("Hello from file 1", encoding="utf-8")
            p2 = Path(td) / "test2.py"
            p2.write_text("def foo(): pass", encoding="utf-8")

            ctx = load_file_context([str(p1), str(p2)])
            self.assertIn("--- Context File: test1.txt ---", ctx)
            self.assertIn("Hello from file 1", ctx)
            self.assertIn("--- Context File: test2.py ---", ctx)
            self.assertIn("def foo(): pass", ctx)

    def test_load_file_context_missing(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_file_context(["/nonexistent/path/never_existed_xyz.txt"])

    def test_stream_chat_rejects_remote_by_default(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            list(
                stream_chat(
                    messages=[{"role": "user", "content": "hi"}],
                    base_url="https://remote-api.com/v1",
                    allow_remote=False,
                )
            )
        self.assertIn("not a loopback address", str(ctx.exception))

    def test_stream_chat_sse_mock(self) -> None:
        sse_lines = [
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
            b"\n",
            b'data: {"choices":[{"delta":{"content":" World"}}]}\n',
            b"\n",
            b"data: [DONE]\n",
        ]
        mock_resp = MagicMock()
        mock_resp.__iter__.return_value = sse_lines
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None

        with patch("urllib.request.urlopen", return_value=mock_resp):
            chunks = list(
                stream_chat(
                    messages=[{"role": "user", "content": "test"}],
                    base_url="http://127.0.0.1:11434/v1",
                )
            )
        self.assertEqual(chunks, ["Hello", " World"])

    def test_complete_chat_mock(self) -> None:
        sse_lines = [
            b'data: {"choices":[{"delta":{"content":"Complete"}}]}\n',
            b'data: {"choices":[{"delta":{"content":" response."}}]}\n',
            b"data: [DONE]\n",
        ]
        mock_resp = MagicMock()
        mock_resp.__iter__.return_value = sse_lines
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None

        with patch("urllib.request.urlopen", return_value=mock_resp):
            resp = complete_chat(
                messages=[{"role": "user", "content": "test"}],
                base_url="http://127.0.0.1:11434/v1",
            )
        self.assertEqual(resp, "Complete response.")


class TestChatCli(unittest.TestCase):
    """Test CLI commands for chat."""

    def test_cli_chat_oneshot_json(self) -> None:
        sse_lines = [
            b'data: {"choices":[{"delta":{"content":"JSON test passed."}}]}\n',
            b"data: [DONE]\n",
        ]
        mock_resp = MagicMock()
        mock_resp.__iter__.return_value = sse_lines
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None

        out = io.StringIO()
        with patch("urllib.request.urlopen", return_value=mock_resp), patch("sys.stdout", out):
            code = main(["chat", "-q", "hello", "--preset", "security", "--json"])
        self.assertEqual(code, 0)
        data = json.loads(out.getvalue())
        self.assertEqual(data["preset"], "security")
        self.assertEqual(data["response"], "JSON test passed.")

    def test_cli_chat_with_context_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "secret.c"
            f.write_text("int secret_func() { return 42; }", encoding="utf-8")

            sse_lines = [
                b'data: {"choices":[{"delta":{"content":"Vulnerability inspected."}}]}\n',
                b"data: [DONE]\n",
            ]
            mock_resp = MagicMock()
            mock_resp.__iter__.return_value = sse_lines
            mock_resp.__enter__.return_value = mock_resp
            mock_resp.__exit__.return_value = None

            out = io.StringIO()
            with patch("urllib.request.urlopen", return_value=mock_resp), patch("sys.stdout", out):
                code = main(["chat", "-f", str(f), "-q", "Audit this", "--no-stream"])
            self.assertEqual(code, 0)
            self.assertIn("Vulnerability inspected.", out.getvalue())


if __name__ == "__main__":
    unittest.main()
