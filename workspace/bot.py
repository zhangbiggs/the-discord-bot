"""
/opus Discord Bot — three phases in one.

Phase 1 — Core async flow:
  POST /interactions  → deferred ACK → pipeline with callback → follow-up
Phase 2 — One-time claim tokens:
  POST /api/tokens    → register a video_url under a single-use token
  GET  /api/claim/{t} → validate token, return video_url (24h expiry)
Phase 3 — LLM window selection:
  Before pipeline call, fetch transcript → LLM picks best 30 s window

Usage:
  cd workspace && pip install -r requirements.txt && python bot.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import re
import uvicorn

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("opusbot")

# ── Configuration ────────────────────────────────────────────────────────
# All URLs are overridable via environment variables.
# Host defaults (python bot.py directly):        localhost
# Docker defaults (docker run):                  host.docker.internal
MOCK_DISCORD = os.environ.get("MOCK_DISCORD_URL", "http://localhost:7001")
MOCK_PLATFORM = os.environ.get("MOCK_PLATFORM_URL", "http://localhost:7002")
MOCK_LLM = os.environ.get("MOCK_LLM_URL", "http://localhost:7003")
CALLBACK_HOST = os.environ.get("CALLBACK_HOST", "http://host.docker.internal:8080")
BOT_PORT = int(os.environ.get("BOT_PORT", "8080"))

# ── Storage ──────────────────────────────────────────────────────────────

# Phase 1: job_id → {app_id, token, created_at, done}
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = asyncio.Lock()

# Phase 2: token → {video_url, created_at, claimed}
_tokens: dict[str, dict[str, Any]] = {}
_tokens_lock = asyncio.Lock()
TOKEN_TTL_SECONDS = 24 * 3600  # 24 hours


async def _store_job(job_id: str, info: dict[str, Any]) -> None:
    async with _jobs_lock:
        _jobs[job_id] = info


async def _pop_job(job_id: str) -> dict[str, Any] | None:
    """Retrieve and remove a job record (one-shot consumption)."""
    async with _jobs_lock:
        return _jobs.pop(job_id, None)


async def _store_token(token: str, info: dict[str, Any]) -> None:
    async with _tokens_lock:
        _tokens[token] = info


async def _get_token(token: str) -> dict[str, Any] | None:
    async with _tokens_lock:
        info = _tokens.get(token)
        if info is None:
            return None
        # Check expiry
        if time.time() - info["created_at"] > TOKEN_TTL_SECONDS:
            _tokens.pop(token, None)
            return None
        return info


async def _claim_token(token: str) -> dict[str, Any] | None:
    """Atomically claim a token. Returns None if already claimed."""
    async with _tokens_lock:
        info = _tokens.get(token)
        if info is None:
            return None
        if time.time() - info["created_at"] > TOKEN_TTL_SECONDS:
            _tokens.pop(token, None)
            return None
        if info["claimed"]:
            return None
        info["claimed"] = True
        return info


# ── FastAPI Application ──────────────────────────────────────────────────

_http: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http
    # Explicitly disable system proxy detection (bypass VPN/proxy tools
    # that intercept localhost traffic, e.g. ClashX at 127.0.0.1:1082).
    _http = httpx.AsyncClient(
        timeout=30.0,
        mounts={
            "http://": httpx.AsyncHTTPTransport(),
            "https://": httpx.AsyncHTTPTransport(),
        },
    )
    try:
        yield
    finally:
        await _http.aclose()


app = FastAPI(title="opus-discord-bot", lifespan=lifespan)


# ══════════════════════════════════════════════════════════════════════════
# Phase 1 — Core Discord Integration
# ══════════════════════════════════════════════════════════════════════════

@app.post("/interactions")
async def handle_interaction(req: Request) -> dict[str, Any]:
    """Entry point for Discord slash commands (called by mock-discord).

    Must respond within 3 seconds.  We return a deferred placeholder
    (type 5) and let a background task do the heavy lifting.
    """
    body = await req.json()
    logger.info("Received interaction type=%s", body.get("type"))

    # Ping/pong (Discord uses type 1 for URL verification)
    if body.get("type") == 1:
        return {"type": 1}

    # APPLICATION_COMMAND (type 2)
    if body.get("type") != 2:
        return {"type": 4, "data": {"content": "Unsupported interaction type."}}

    data = body.get("data", {})
    name = data.get("name", "")

    if name != "opus":
        return {"type": 4, "data": {"content": f"Unknown command: /{name}"}}

    # Extract options
    options = {o["name"]: o["value"] for o in data.get("options", [])}
    url = options.get("url", "")

    if not url:
        return {"type": 4, "data": {"content": "Usage: /opus <youtube-url>"}}

    app_id = body.get("application_id", "mock-app-0001")
    token = body.get("token", "")

    logger.info("Processing /opus url=%s interaction=%s", url, body.get("id"))

    # ⚡ Fire background task — this MUST happen after the response is
    # returned, so we use create_task which schedules the coroutine for
    # the next event-loop iteration (after the response is sent).
    asyncio.create_task(_process_opus(url, app_id, token))

    # Deferred channel message — Discord shows "… is thinking"
    return {"type": 5}


async def _process_opus(url: str, app_id: str, token: str) -> None:
    """Background processing for /opus.

    Phase 3: Fetch transcript → LLM picks window → pipeline with callback.
    Phase 1: Pipeline creates clip → callback → follow-up.
    Phase 2: Follow-up delivers claim link instead of raw video URL.
    """
    assert _http is not None
    client: httpx.AsyncClient = _http

    # ── Phase 3: Transcript + LLM window selection ──────────────────────
    start_seconds: float | None = None
    end_seconds: float | None = None

    try:
        transcript = await _fetch_transcript(client, url)
        if transcript:
            window = await _pick_window_with_llm(client, transcript)
            if window:
                start_seconds = window.get("start_seconds")
                end_seconds = window.get("end_seconds")
                logger.info(
                    "LLM selected window: %.1f–%.1f  (reason: %s)",
                    start_seconds, end_seconds, window.get("reason", "?"),
                )
    except Exception as exc:
        logger.warning("LLM window selection failed, falling back to full clip: %s", exc)

    # ── Phase 1: Create pipeline clip job ───────────────────────────────
    callback_url = f"{CALLBACK_HOST}/clips/done"

    payload: dict[str, Any] = {
        "url": url,
        "callback_url": callback_url,
    }
    if start_seconds is not None and end_seconds is not None:
        payload["start_seconds"] = start_seconds
        payload["end_seconds"] = end_seconds

    try:
        resp = await client.post(
            f"{MOCK_PLATFORM}/v1/clip",
            json=payload,
            timeout=10.0,
        )
        resp.raise_for_status()
        job = resp.json()
        job_id = job["job_id"]

        await _store_job(job_id, {
            "app_id": app_id,
            "token": token,
            "created_at": time.time(),
            "done": False,
        })
        logger.info("Clip job created: %s  eta=%ss", job_id, job.get("eta_seconds"))
    except Exception as exc:
        logger.error("Failed to create clip job: %s", exc)
        await _send_followup(client, app_id, token,
                             f"❌ Failed to start clip generation: {exc}")


@app.post("/clips/done")
async def clip_callback(req: Request) -> dict[str, Any]:
    """Webhook callback from mock-platform when a clip finishes.

    Phase 1: Post the video URL as a follow-up to Discord.
    Phase 2: Create a one-time claim token, post the claim link instead.
    """
    body = await req.json()
    job_id = body.get("job_id", "")
    status = body.get("status", "")
    video_url = body.get("video_url", "")

    logger.info("Clip callback: job=%s status=%s", job_id, status)

    if status != "done":
        logger.warning("Unexpected job status: %s", status)
        return {"ok": False, "error": f"unexpected status: {status}"}

    if not video_url:
        logger.warning("No video_url in callback")
        return {"ok": False, "error": "missing video_url"}

    # Consume the job record (idempotent: duplicate callbacks are ignored)
    info = await _pop_job(job_id)
    if info is None:
        logger.warning("Duplicate or unknown callback for job %s — ignoring", job_id)
        return {"ok": True, "warning": "already processed"}

    assert _http is not None
    client: httpx.AsyncClient = _http

    # ── Phase 2: Create claim token instead of posting video directly ───
    try:
        # Create a one-time claim token
        token_str = secrets.token_hex(16)
        await _store_token(token_str, {
            "video_url": video_url,
            "created_at": time.time(),
            "claimed": False,
        })

        claim_url = f"https://opusclip.com/claim/{token_str}"
        content = (
            f"🎬 **Your clip is ready!**\n"
            f"Claim it here (one-time link, expires in 24h):\n"
            f"{claim_url}"
        )
        await _send_followup(client, info["app_id"], info["token"], content)
        logger.info("Claim link sent for job %s  token=%s", job_id, token_str[:8])
    except Exception as exc:
        logger.error("Failed to create token / send follow-up: %s", exc)
        # Fallback: send the video URL directly
        try:
            fallback = f"🎬 Your clip is ready!\n{video_url}"
            await _send_followup(client, info["app_id"], info["token"], fallback)
        except Exception as e2:
            logger.error("Fallback follow-up also failed: %s", e2)

    return {"ok": True}


async def _send_followup(
    client: httpx.AsyncClient,
    app_id: str,
    token: str,
    content: str,
) -> None:
    """POST a follow-up message to the Discord interaction webhook."""
    resp = await client.post(
        f"{MOCK_DISCORD}/webhooks/{app_id}/{token}",
        json={"content": content},
        timeout=10.0,
    )
    resp.raise_for_status()
    logger.info("Follow-up sent (HTTP %d)", resp.status_code)


# ══════════════════════════════════════════════════════════════════════════
# Phase 2 — One-Time Claim Token API
# ══════════════════════════════════════════════════════════════════════════

class CreateTokenRequest(BaseModel):
    video_url: str = Field(..., description="URL of the finished clip video")


class CreateTokenResponse(BaseModel):
    token: str
    claim_url: str
    expires_in_seconds: int


@app.post("/api/tokens", response_model=CreateTokenResponse)
async def create_token(req: CreateTokenRequest) -> CreateTokenResponse:
    """Register a video URL under a fresh one-time claim token."""
    token_str = secrets.token_hex(16)
    await _store_token(token_str, {
        "video_url": req.video_url,
        "created_at": time.time(),
        "claimed": False,
    })
    return CreateTokenResponse(
        token=token_str,
        claim_url=f"https://opusclip.com/claim/{token_str}",
        expires_in_seconds=TOKEN_TTL_SECONDS,
    )


class ClaimResponse(BaseModel):
    video_url: str


@app.get("/api/claim/{token}", response_model=ClaimResponse)
async def claim_token(token: str) -> ClaimResponse:
    """Validate and consume a one-time claim token.

    Returns the video URL on success.
    Errors:
      404 — token not found or expired
      409 — token already claimed
    """
    # First check if the token exists at all (for better error messages)
    pre_check = await _get_token(token)
    if pre_check is None:
        raise HTTPException(status_code=404, detail="Token not found or expired")

    if pre_check["claimed"]:
        raise HTTPException(status_code=409, detail="Token already claimed")

    info = await _claim_token(token)
    if info is None:
        # Race: someone else claimed it between the check and the claim
        raise HTTPException(status_code=409, detail="Token already claimed")

    return ClaimResponse(video_url=info["video_url"])


# ══════════════════════════════════════════════════════════════════════════
# Phase 3 — LLM-assisted highlight window selection
# ══════════════════════════════════════════════════════════════════════════

async def _fetch_transcript(
    client: httpx.AsyncClient,
    url: str,
) -> dict[str, Any] | None:
    """Fetch YouTube transcript from mock-platform."""
    try:
        resp = await client.get(
            f"{MOCK_PLATFORM}/v1/transcript",
            params={"url": url},
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()
        logger.info("Transcript fetched: %s  (%d segments, %.0fs total)",
                     data.get("title", "?"),
                     len(data.get("segments", [])),
                     data.get("duration_seconds", 0))
        return data
    except Exception as exc:
        logger.warning("Failed to fetch transcript: %s", exc)
        return None


async def _pick_window_with_llm(
    client: httpx.AsyncClient,
    transcript: dict[str, Any],
) -> dict[str, Any] | None:
    """Ask the LLM to choose the most interesting 30-second window.

    Uses response_format + explicit schema description to defeat the
    adversarial LLM (which would otherwise return prose).
    """
    segments = transcript.get("segments", [])
    duration = transcript.get("duration_seconds", 240)

    # Build a compact transcript summary for the LLM
    transcript_text = "\n".join(
        f"[{seg['start']:.0f}s–{seg['end']:.0f}s] {seg['text']}"
        for seg in segments
    )

    system_prompt = (
        "You are a video editor assistant. Your job is to select the most "
        "engaging 30-second clip from a video transcript.\n\n"
        "Rules:\n"
        "- The clip must be 25–45 seconds long.\n"
        "- It should capture the most interesting, exciting, or dramatic part.\n"
        "- Prefer moments with high emotional impact, surprises, or payoff.\n\n"
        "Respond with a valid JSON object. Use exactly this schema:\n"
        '{"start_seconds": int, "end_seconds": int, '
        '"reason": string, "confidence": float}\n\n'
        "Return ONLY the JSON object, no other text."
    )

    user_prompt = (
        f"Here is the video transcript (total duration: {duration:.0f}s):\n\n"
        f"{transcript_text}\n\n"
        f"Select the best 25–45 second highlight window. "
        f"Return a JSON object with keys: start_seconds, end_seconds, reason, confidence."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        resp = await client.post(
            f"{MOCK_LLM}/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": messages,
                "response_format": {"type": "json_object"},
                "temperature": 0.2,
                "max_tokens": 300,
            },
            timeout=15.0,
        )
        resp.raise_for_status()
        body = resp.json()
        content = body["choices"][0]["message"]["content"]

        # Parse the JSON response
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from prose (fallback)
            logger.warning("LLM response not valid JSON, attempting extraction")
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                result = json.loads(match.group())
            else:
                raise

        start = float(result.get("start_seconds", 0))
        end = float(result.get("end_seconds", 0))

        # Validate bounds
        if start < 0:
            start = 0
        if end > duration:
            end = duration
        if end - start < 15:
            # Too short — extend to minimum 25 s
            end = min(start + 25, duration)
        if end - start > 60:
            # Cap at 60 s
            end = start + 60
        if end > duration:
            end = duration
            start = max(0, duration - 30)
        if start >= end or end - start < 15:
            # Fallback to sensible default
            start = duration / 3
            end = min(start + 30, duration)

        return {
            "start_seconds": start,
            "end_seconds": end,
            "reason": result.get("reason", ""),
            "confidence": result.get("confidence", 0),
        }

    except Exception as exc:
        logger.warning("LLM window selection failed: %s", exc)
        # Fallback: use the middle third of the video
        fallback_start = duration / 3
        fallback_end = min(fallback_start + 30, duration)
        return {
            "start_seconds": fallback_start,
            "end_seconds": fallback_end,
            "reason": "fallback (LLM unavailable)",
            "confidence": 0.0,
        }


# ══════════════════════════════════════════════════════════════════════════
# Diagnostics
# ══════════════════════════════════════════════════════════════════════════

@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {"ok": True, "uptime": "…"}


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("Starting opus bot on 0.0.0.0:%d", BOT_PORT)
    uvicorn.run(app, host="0.0.0.0", port=BOT_PORT)
