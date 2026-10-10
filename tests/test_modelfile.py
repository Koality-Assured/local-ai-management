"""Unit tests for modelfile generator."""

from __future__ import annotations

import unittest

from local_ai.modelfile import generate_llama_server_command, generate_ollama_modelfile


class TestModelfileGenerator(unittest.TestCase):
    """Test synthesis of Modelfiles and startup commands."""

    def test_generate_ollama_modelfile(self) -> None:
        mf = generate_ollama_modelfile(
            base_model="qwen2.5-coder:14b",
            context_tokens=32768,
            temperature=0.2,
            system_prompt="You are a senior coding assistant.",
        )
        self.assertIn("FROM qwen2.5-coder:14b", mf)
        self.assertIn("PARAMETER num_ctx 32768", mf)
        self.assertIn("PARAMETER temperature 0.2", mf)
        self.assertIn("SYSTEM \"\"\"You are a senior coding assistant.\"\"\"", mf)

    def test_generate_llama_server_command(self) -> None:
        cmd = generate_llama_server_command(
            gguf_path="models/test.gguf",
            gpu_layers=36,
            context_tokens=16384,
            cpu_threads=16,
            quantize_kv=True,
        )
        self.assertIn("./llama-server", cmd)
        self.assertIn("-m models/test.gguf", cmd)
        self.assertIn("-ngl 36", cmd)
        self.assertIn("-c 16384", cmd)
        self.assertIn("-t 16", cmd)
        self.assertIn("-ctk q8_0 -ctv q8_0", cmd)


if __name__ == "__main__":
    unittest.main()
