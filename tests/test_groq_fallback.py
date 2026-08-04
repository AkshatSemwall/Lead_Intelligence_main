"""
Tests for the Groq fallback resilience layer added to LLMClient.

Priority chain verified:
  1. Gemini (primary)
  2. Groq   (fallback when Gemini fails)
  3. _fallback_response() (deterministic, when both LLMs fail)
"""
from __future__ import annotations

import asyncio
import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from backend.services.gemini_client import LLMClient


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_settings(*, groq_api_key: str = "test-groq-key") -> Any:
    return types.SimpleNamespace(
        llm_provider="gemini",
        gemini_api_key="test-gemini-key",
        gemini_model="gemini-1.5-flash",
        groq_api_key=groq_api_key,
        groq_model="llama-3.3-70b-versatile",
    )


def _groq_response(text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ── tests ────────────────────────────────────────────────────────────────────

def test_groq_called_when_gemini_fails() -> None:
    """When every Gemini model fails, Groq must be tried next."""
    settings = _make_settings()
    client = LLMClient(settings=settings)
    groq_resp = _groq_response("Groq says hello")

    async def run():
        with patch.object(client, "_get_client", return_value=MagicMock()), \
             patch("asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(
                side_effect=RuntimeError("Gemini quota exceeded")
            )
            with patch("google.generativeai.configure"), \
                 patch("google.generativeai.GenerativeModel"):
                mock_groq_client = AsyncMock()
                mock_groq_client.chat.completions.create = AsyncMock(return_value=groq_resp)
                with patch("groq.AsyncGroq", return_value=mock_groq_client):
                    result = await client.generate("test prompt")
        assert result == "Groq says hello"

    asyncio.run(run())


def test_deterministic_fallback_when_both_fail() -> None:
    """When both Gemini and Groq fail, the deterministic response is returned."""
    settings = _make_settings()
    client = LLMClient(settings=settings)

    async def run():
        with patch.object(client, "_get_client", return_value=MagicMock()), \
             patch("asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(
                side_effect=RuntimeError("Gemini down")
            )
            with patch("google.generativeai.configure"), \
                 patch("google.generativeai.GenerativeModel"):
                mock_groq_client = AsyncMock()
                mock_groq_client.chat.completions.create = AsyncMock(
                    side_effect=RuntimeError("Groq also down")
                )
                with patch("groq.AsyncGroq", return_value=mock_groq_client):
                    result = await client.generate("test prompt")
        assert "Fallback response generated" in result

    asyncio.run(run())


def test_groq_skipped_when_no_api_key() -> None:
    """When GROQ_API_KEY is empty, Groq must be skipped entirely."""
    settings = _make_settings(groq_api_key="")
    client = LLMClient(settings=settings)

    async def run():
        with patch.object(client, "_get_client", return_value=MagicMock()), \
             patch("asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(
                side_effect=RuntimeError("Gemini down")
            )
            with patch("google.generativeai.configure"), \
                 patch("google.generativeai.GenerativeModel"):
                with patch("groq.AsyncGroq") as mock_groq_cls:
                    result = await client.generate("test prompt")
                    mock_groq_cls.assert_not_called()
        assert "Fallback response generated" in result

    asyncio.run(run())


def test_settings_has_groq_fields() -> None:
    """Settings class must expose groq_api_key and groq_model fields."""
    from backend.config import Settings

    s = Settings(
        _env_file=None,
        gemini_api_key="dummy",
        llm_provider="gemini",
        groq_api_key="gsk_test",
        groq_model="llama-3.3-70b-versatile",
    )
    assert s.groq_api_key == "gsk_test"
    assert s.groq_model == "llama-3.3-70b-versatile"


def test_groq_api_key_stripped_by_validator() -> None:
    """groq_api_key must be stripped of whitespace by the shared validator."""
    from backend.config import Settings

    s = Settings(
        _env_file=None,
        gemini_api_key="dummy",
        llm_provider="gemini",
        groq_api_key="  gsk_test  ",
    )
    assert s.groq_api_key == "gsk_test"


def test_groq_default_model() -> None:
    """groq_model must default to llama-3.3-70b-versatile when not set."""
    from backend.config import Settings

    s = Settings(
        _env_file=None,
        gemini_api_key="dummy",
        llm_provider="gemini",
    )
    assert s.groq_model == "llama-3.3-70b-versatile"
