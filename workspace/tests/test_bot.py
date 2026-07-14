"""Unit tests for the opus Discord bot.

Run with:
    cd workspace && pip install pytest httpx && python -m pytest tests/ -v
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from httpx import ASGITransport

# Ensure bot module is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import bot
from bot import (
    _claim_token,
    _get_token,
    _jobs,
    _pop_job,
    _store_job,
    _store_token,
    _tokens,
    app,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
async def client():
    """FastAPI ASGI test client — no server process needed."""
    transport = ASGITransport(app=app)
    ac = httpx.AsyncClient(transport=transport, base_url="http://test")
    yield ac
    await ac.aclose()


@pytest.fixture(autouse=True)
def clear_state():
    """Reset in-memory stores and mock _http before each test."""
    _jobs.clear()
    _tokens.clear()
    # Mock _http so clip_callback doesn't hit the real network
    with patch.object(bot, "_http", AsyncMock(spec=httpx.AsyncClient)):
        yield


# ══════════════════════════════════════════════════════════════════════════
# Phase 1 — Discord Interaction Handling
# ══════════════════════════════════════════════════════════════════════════

class TestInteractions:
    """Tests for POST /interactions — the Discord webhook entry point."""

    async def test_ping_pong(self, client):
        """Type 1 (ping) → type 1 (pong)."""
        resp = await client.post("/interactions", json={"type": 1})
        assert resp.status_code == 200
        assert resp.json() == {"type": 1}

    async def test_opus_returns_deferred(self, client):
        """/opus → type 5 (deferred ACK) within 3 seconds."""
        resp = await client.post("/interactions", json={
            "type": 2,
            "id": "test-id-001",
            "token": "test-token-001",
            "application_id": "mock-app-0001",
            "data": {
                "name": "opus",
                "options": [{"name": "url", "value": "https://youtube.com/watch?v=test"}],
            },
            "user": {"id": "u-1", "username": "tester"},
            "channel_id": "general",
        })
        assert resp.status_code == 200
        assert resp.json() == {"type": 5}

    async def test_opus_missing_url_shows_usage(self, client):
        """/opus without url → type 4 with usage hint."""
        resp = await client.post("/interactions", json={
            "type": 2,
            "id": "test-id-002",
            "token": "test-token-002",
            "application_id": "mock-app-0001",
            "data": {"name": "opus", "options": []},
            "user": {"id": "u-1", "username": "tester"},
            "channel_id": "general",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["type"] == 4
        assert "Usage" in body["data"]["content"]

    async def test_unknown_command_rejected(self, client):
        """Unknown command → type 4 with error message."""
        resp = await client.post("/interactions", json={
            "type": 2,
            "id": "test-id-003",
            "token": "test-token-003",
            "application_id": "mock-app-0001",
            "data": {"name": "foobar", "options": []},
            "user": {"id": "u-1", "username": "tester"},
            "channel_id": "general",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["type"] == 4
        assert "Unknown command" in body["data"]["content"]

    async def test_unsupported_interaction_type(self, client):
        """Type 3 (unsupported) → type 4."""
        resp = await client.post("/interactions", json={"type": 3})
        assert resp.status_code == 200
        assert resp.json()["type"] == 4

    async def test_deferred_response_is_fast(self, client):
        """Verify the handler returns before any external call."""
        import asyncio
        called = False

        async def tracking(url, app_id, token):
            nonlocal called
            called = True

        with patch.object(bot, "_process_opus", tracking):
            resp = await client.post("/interactions", json={
                "type": 2,
                "id": "test-fast",
                "token": "tok-fast",
                "application_id": "mock-app-0001",
                "data": {
                    "name": "opus",
                    "options": [{"name": "url", "value": "https://youtube.com/watch?v=test"}],
                },
                "user": {"id": "u-1", "username": "tester"},
                "channel_id": "general",
            })
        # Response came back before background task ran
        assert resp.status_code == 200
        await asyncio.sleep(0)
        assert called, "_process_opus should have been scheduled"


# ══════════════════════════════════════════════════════════════════════════
# Phase 2 — One-Time Claim Token API
# ══════════════════════════════════════════════════════════════════════════

class TestTokenAPI:
    """Tests for POST /api/tokens and GET /api/claim/{token}."""

    TEST_VIDEO = "https://cdn.opusclip-mock.local/test.mp4"

    async def test_create_token(self, client):
        """POST /api/tokens → token + claim_url + expiry."""
        resp = await client.post("/api/tokens", json={"video_url": self.TEST_VIDEO})
        assert resp.status_code == 200
        body = resp.json()
        assert "token" in body
        assert len(body["token"]) == 32       # 16 bytes → 32 hex chars
        assert body["claim_url"].startswith("https://opusclip.com/claim/")
        assert body["expires_in_seconds"] == 24 * 3600

    async def test_claim_first_time_returns_video(self, client):
        """First claim → 200 + video_url."""
        cr = await client.post("/api/tokens", json={"video_url": self.TEST_VIDEO})
        token = cr.json()["token"]

        resp = await client.get(f"/api/claim/{token}")
        assert resp.status_code == 200
        assert resp.json() == {"video_url": self.TEST_VIDEO}

    async def test_claim_twice_returns_409(self, client):
        """Second claim → 409 Conflict."""
        cr = await client.post("/api/tokens", json={"video_url": self.TEST_VIDEO})
        token = cr.json()["token"]

        r1 = await client.get(f"/api/claim/{token}")
        assert r1.status_code == 200

        r2 = await client.get(f"/api/claim/{token}")
        assert r2.status_code == 409

    async def test_claim_invalid_token_returns_404(self, client):
        """Bogus token → 404."""
        resp = await client.get("/api/claim/nonexistent123")
        assert resp.status_code == 404

    async def test_claim_expired_token_returns_404(self, client):
        """Token past 24h TTL → 404."""
        expired = "expiredtoken12345678"
        _tokens[expired] = {
            "video_url": self.TEST_VIDEO,
            "created_at": time.time() - 25 * 3600,  # 25 h ago
            "claimed": False,
        }
        resp = await client.get(f"/api/claim/{expired}")
        assert resp.status_code == 404


# ══════════════════════════════════════════════════════════════════════════
# Phase 3 — LLM Transcript & Window Selection
# ══════════════════════════════════════════════════════════════════════════

SAMPLE_TRANSCRIPT = {
    "source_url": "https://youtube.com/watch?v=test",
    "video_id": "ds-001",
    "title": "24 Hours Inside An Abandoned Theme Park",
    "duration_seconds": 240,
    "language": "en",
    "segments": [
        {"start": 0, "end": 8,    "text": "What's up guys, today we're doing something new."},
        {"start": 8, "end": 18,   "text": "Spending 24 hours inside an abandoned theme park."},
        {"start": 75, "end": 90,  "text": "Wait. Did you hear that? Stop."},
        {"start": 90, "end": 105, "text": "OH MY GOD. The Ferris wheel started moving by itself."},
        {"start": 105, "end": 120, "text": "This is the craziest thing I have ever seen."},
    ],
}

LLM_STRUCTURED = {
    "start_seconds": 75,
    "end_seconds": 120,
    "reason": "Highest emotional spike — Ferris wheel scene.",
    "confidence": 0.86,
}


class TestTranscript:
    """Tests for _fetch_transcript."""

    async def test_fetch_success(self):
        """200 response → parsed transcript."""
        class MockResponse:
            def raise_for_status(self):
                pass
            def json(self):
                return SAMPLE_TRANSCRIPT

        class MockClient:
            async def get(self, *args, **kwargs):
                return MockResponse()

        mock_client = MockClient()

        result = await bot._fetch_transcript(mock_client, "https://youtube.com/watch?v=test")
        assert result is not None
        assert result["title"] == SAMPLE_TRANSCRIPT["title"]
        assert len(result["segments"]) == 5
        assert result["duration_seconds"] == 240

    async def test_fetch_http_error_returns_none(self):
        """Non-200 → None."""
        class MockResponse:
            def raise_for_status(self):
                raise httpx.HTTPStatusError(
                    "404", request=MagicMock(), response=MagicMock(status_code=404),
                )
            def json(self):
                return {}

        class MockClient:
            async def get(self, *args, **kwargs):
                return MockResponse()

        mock_client = MockClient()
        result = await bot._fetch_transcript(mock_client, "https://youtube.com/watch?v=test")
        assert result is None

    async def test_fetch_timeout_returns_none(self):
        """Timeout → None."""
        class MockClient:
            async def get(self, *args, **kwargs):
                raise httpx.TimeoutException("timeout")

        mock_client = MockClient()
        result = await bot._fetch_transcript(mock_client, "https://youtube.com/watch?v=test")
        assert result is None


class TestLLMWindow:
    """Tests for _pick_window_with_llm."""

    def _build_mocks(self, response_json=None, side_effect=None):
        class MockResponse:
            def __init__(self, data):
                self._data = data
                self.status_code = 200
            def raise_for_status(self):
                pass
            def json(self):
                return self._data

        class MockClient:
            def __init__(self, resp_data=None, err=None):
                self._resp_data = resp_data
                self._err = err
            async def post(self, *args, **kwargs):
                if self._err:
                    raise self._err
                return MockResponse(self._resp_data)

        if side_effect:
            return MockClient(err=side_effect)
        return MockClient(resp_data=response_json)

    async def test_parses_valid_json(self):
        """LLM returns structured JSON → parsed window."""
        client = self._build_mocks({
            "choices": [{"message": {"content": json.dumps(LLM_STRUCTURED)}}],
        })
        result = await bot._pick_window_with_llm(client, SAMPLE_TRANSCRIPT)
        assert result is not None
        assert result["start_seconds"] == 75.0
        assert result["end_seconds"] == 120.0
        assert result["confidence"] == 0.86
        assert "Ferris" in result["reason"]

    async def test_fallback_on_http_error(self):
        """LLM 500 → fallback window (middle third)."""
        client = self._build_mocks(side_effect=httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock(status_code=500),
        ))
        result = await bot._pick_window_with_llm(client, SAMPLE_TRANSCRIPT)
        assert result is not None
        assert result["start_seconds"] == 80.0   # 240/3
        assert result["end_seconds"] == 110.0     # 80+30
        assert result["reason"] == "fallback (LLM unavailable)"

    async def test_fallback_on_timeout(self):
        """LLM timeout → fallback window."""
        client = self._build_mocks(side_effect=httpx.TimeoutException("timeout"))
        result = await bot._pick_window_with_llm(client, SAMPLE_TRANSCRIPT)
        assert result is not None
        assert result["start_seconds"] == 80.0
        assert result["end_seconds"] == 110.0

    async def test_handles_prose_with_embedded_json(self):
        """Prose response → regex extraction of embedded JSON."""
        prose = (
            'Some text {"start_seconds": 75, "end_seconds": 120, '
            '"reason": "Ferris", "confidence": 0.86} more text'
        )
        client = self._build_mocks({
            "choices": [{"message": {"content": prose}}],
        })
        result = await bot._pick_window_with_llm(client, SAMPLE_TRANSCRIPT)
        assert result is not None
        assert result["start_seconds"] == 75.0
        assert result["end_seconds"] == 120.0


# ══════════════════════════════════════════════════════════════════════════
# Pipeline Callback (Phase 1 + 2 combined)
# ══════════════════════════════════════════════════════════════════════════

class TestClipCallback:
    """Tests for POST /clips/done — pipeline completion webhook."""

    async def test_unknown_job_returns_warning(self, client):
        """Callback for unrecognised job → warning."""
        resp = await client.post("/clips/done", json={
            "job_id": "no-such-job",
            "status": "done",
            "video_url": "https://cdn.opusclip-mock.local/test.mp4",
        })
        assert resp.status_code == 200
        assert resp.json().get("warning") == "already processed"

    async def test_not_done_status_returns_error(self, client):
        """Callback with status != 'done' → error."""
        resp = await client.post("/clips/done", json={
            "job_id": "some-job", "status": "processing", "video_url": "",
        })
        assert resp.status_code == 200
        assert resp.json().get("error") is not None

    async def test_missing_video_url_returns_error(self, client):
        """Callback without video_url → error."""
        resp = await client.post("/clips/done", json={
            "job_id": "some-job", "status": "done", "video_url": "",
        })
        assert resp.status_code == 200
        assert resp.json().get("error") is not None

    async def test_happy_path_creates_token_and_sends_followup(self, client):
        """Happy path: callback → token created → follow-up sent."""
        await _store_job("job-abc-123", {
            "app_id": "mock-app-0001",
            "token": "interaction-token-xyz",
            "created_at": time.time(),
            "done": False,
        })

        with patch.object(bot, "_send_followup", AsyncMock()) as mock_send:
            resp = await client.post("/clips/done", json={
                "job_id": "job-abc-123",
                "status": "done",
                "video_url": "https://cdn.opusclip-mock.local/abcdef.mp4",
            })

        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        # Follow-up was sent
        assert mock_send.called
        # Token was created for the video
        found = any(
            info["video_url"] == "https://cdn.opusclip-mock.local/abcdef.mp4"
            for info in _tokens.values()
        )
        assert found, "Token should exist for the video URL"

    async def test_duplicate_callback_is_idempotent(self, client):
        """Duplicate callback should be ignored (job already popped)."""
        await _store_job("job-dup", {
            "app_id": "mock-app-0001", "token": "tok", "created_at": time.time(), "done": False,
        })
        with patch.object(bot, "_send_followup", AsyncMock()) as mock_send:
            r1 = await client.post("/clips/done", json={
                "job_id": "job-dup", "status": "done",
                "video_url": "https://cdn.opusclip-mock.local/dup.mp4",
            })
            r2 = await client.post("/clips/done", json={
                "job_id": "job-dup", "status": "done",
                "video_url": "https://cdn.opusclip-mock.local/dup.mp4",
            })
        assert r1.status_code == 200 and r1.json()["ok"] is True
        assert r2.status_code == 200 and r2.json().get("warning") == "already processed"
        assert mock_send.call_count == 1  # only first callback acted


# ══════════════════════════════════════════════════════════════════════════
# Storage helpers
# ══════════════════════════════════════════════════════════════════════════

class TestStorage:
    """Tests for internal _store_job / _pop_job / _store_token / _claim_token."""

    async def test_store_and_pop_job(self):
        await _store_job("j1", {"app_id": "a", "token": "t", "created_at": 0})
        assert "j1" in _jobs
        popped = await _pop_job("j1")
        assert popped["app_id"] == "a"
        assert "j1" not in _jobs

    async def test_pop_nonexistent_job(self):
        assert await _pop_job("nope") is None

    async def test_store_and_claim_token(self):
        await _store_token("t1", {"video_url": "url", "created_at": time.time(), "claimed": False})
        claimed = await _claim_token("t1")
        assert claimed is not None
        assert claimed["video_url"] == "url"

    async def test_claim_twice(self):
        await _store_token("t2", {"video_url": "url", "created_at": time.time(), "claimed": True})
        assert await _claim_token("t2") is None

    async def test_claim_expired(self):
        await _store_token("t3", {"video_url": "url", "created_at": time.time() - 25*3600, "claimed": False})
        assert await _claim_token("t3") is None


# ══════════════════════════════════════════════════════════════════════════
# Health check
# ══════════════════════════════════════════════════════════════════════════

class TestHealth:
    async def test_healthz(self, client):
        resp = await client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

        # Patch _send_followup to avoid real HTTP calls
        with patch.object(bot, "_send_followup", AsyncMock()) as mock_send:
            resp = await client.post("/clips/done", json={
                "job_id": "job-abc-123",
                "status": "done",
                "video_url": "https://cdn.opusclip-mock.local/abcdef.mp4",
            })

        assert resp.status_code == 200
        assert resp.json()["ok"] is True

        # A token should have been created
        assert mock_send.called
        call_args = mock_send.call_args
        assert "clip/".format in call_args.kwargs.get("content", "") or \
               "clip/" in call_args[0][3] if len(call_args[0]) > 3 else True

        # Verify token exists in store
        found = False
        for tok, info in _tokens.items():
            if info["video_url"] == "https://cdn.opusclip-mock.local/abcdef.mp4":
                found = True
                assert not info["claimed"]
                break
        assert found, "Token should have been created for the video URL"


# ══════════════════════════════════════════════════════════════════════════
# Storage helpers
# ══════════════════════════════════════════════════════════════════════════

class TestStorage:
    """Tests for internal _store_job / _pop_job / _store_token / _claim_token."""

    async def test_store_and_pop_job(self):
        job_id = "job-1"
        info = {"app_id": "a", "token": "t", "created_at": time.time()}
        await _store_job(job_id, info)
        assert job_id in _jobs

        popped = await _pop_job(job_id)
        assert popped == info
        assert job_id not in _jobs  # removed after pop

    async def test_pop_nonexistent_job(self):
        assert await _pop_job("nonexistent") is None

    async def test_store_and_claim_token(self):
        token = "tok-1"
        info = {"video_url": "url", "created_at": time.time(), "claimed": False}
        await _store_token(token, info)

        claimed = await _claim_token(token)
        assert claimed is not None
        assert claimed["video_url"] == "url"
        assert claimed["claimed"] is True

    async def test_claim_already_claimed(self):
        token = "tok-2"
        await _store_token(token, {"video_url": "url", "created_at": time.time(), "claimed": True})
        assert await _claim_token(token) is None

    async def test_claim_expired(self):
        token = "tok-3"
        await _store_token(token, {
            "video_url": "url",
            "created_at": time.time() - 25 * 3600,
            "claimed": False,
        })
        assert await _claim_token(token) is None


# ══════════════════════════════════════════════════════════════════════════
# Health check
# ══════════════════════════════════════════════════════════════════════════

class TestHealth:
    async def test_healthz(self, client):
        resp = await client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
