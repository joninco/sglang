"""
Unit tests for the Anthropic Messages API serving layer.
"""

import sys
import unittest
from unittest.mock import MagicMock

# Stub out sgl_kernel before any sglang import so the test runs on
# CPU-only runners without the real CUDA library.
for _mod in ("sgl_kernel", "sgl_kernel.kvcacheio"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from sglang.srt.entrypoints.anthropic.protocol import (
    AnthropicContentBlock,
    AnthropicMessagesRequest,
)
from sglang.srt.entrypoints.anthropic.serving import AnthropicServing
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu-only")


def _get_system_content(chat_req):
    """Extract system message content from a ChatCompletionRequest."""
    msg = chat_req.messages[0]
    return msg.content if hasattr(msg, "content") else msg["content"]


class TestBillingHeaderStripping(unittest.TestCase):
    """Verify that per-request billing headers are stripped from the system
    prompt so they don't break RadixAttention prefix caching."""

    def setUp(self):
        self.serving = AnthropicServing.__new__(AnthropicServing)
        self.serving.openai_serving_chat = MagicMock()

    def _make_request(self, system_blocks):
        return AnthropicMessagesRequest(
            model="test-model",
            max_tokens=100,
            messages=[{"role": "user", "content": "hello"}],
            system=system_blocks,
        )

    def test_billing_header_stripped(self):
        """Billing header block should not appear in the converted system message."""
        request = self._make_request(
            [
                AnthropicContentBlock(
                    type="text",
                    text="x-anthropic-billing-header: cc_version=2.1.80; cch=abc123;",
                ),
                AnthropicContentBlock(
                    type="text",
                    text="You are a helpful assistant.",
                ),
            ]
        )
        chat_req = self.serving._convert_to_chat_completion_request(request)
        content = _get_system_content(chat_req)
        self.assertNotIn("billing-header", content)
        self.assertIn("helpful assistant", content)

    def test_system_prompt_stable_across_requests(self):
        """Two requests with different billing headers but same content
        should produce identical system messages."""
        req1 = self._make_request(
            [
                AnthropicContentBlock(
                    type="text",
                    text="x-anthropic-billing-header: cc_version=2.1.80; cch=aaa111;",
                ),
                AnthropicContentBlock(type="text", text="You are a helpful assistant."),
            ]
        )
        req2 = self._make_request(
            [
                AnthropicContentBlock(
                    type="text",
                    text="x-anthropic-billing-header: cc_version=2.1.80; cch=bbb222;",
                ),
                AnthropicContentBlock(type="text", text="You are a helpful assistant."),
            ]
        )
        chat1 = self.serving._convert_to_chat_completion_request(req1)
        chat2 = self.serving._convert_to_chat_completion_request(req2)
        self.assertEqual(
            _get_system_content(chat1),
            _get_system_content(chat2),
        )

    def test_no_billing_header_unchanged(self):
        """System prompt without billing header should pass through normally."""
        request = self._make_request(
            [
                AnthropicContentBlock(type="text", text="You are a helpful assistant."),
                AnthropicContentBlock(type="text", text="Be concise."),
            ]
        )
        chat_req = self.serving._convert_to_chat_completion_request(request)
        content = _get_system_content(chat_req)
        self.assertIn("helpful assistant", content)
        self.assertIn("Be concise", content)

    def test_string_system_prompt_unchanged(self):
        """A plain string system prompt should pass through unmodified."""
        request = AnthropicMessagesRequest(
            model="test-model",
            max_tokens=100,
            messages=[{"role": "user", "content": "hello"}],
            system="You are a helpful assistant.",
        )
        chat_req = self.serving._convert_to_chat_completion_request(request)
        content = _get_system_content(chat_req)
        self.assertEqual(content, "You are a helpful assistant.")


if __name__ == "__main__":
    unittest.main()
