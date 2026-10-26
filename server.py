"""FastAPI server for the System 1 vs. System 2 Claims Router Demo.

Serves the interactive single-page dashboard and API endpoints:
  - GET  /                     -> index.html
  - GET  /api/claims           -> Seed claims, schema definitions, default cascade threshold, cloud availability
  - GET  /api/telemetry        -> Live RTX 4090 telemetry & llama-server health
  - POST /api/evaluate-single  -> Side-by-side evaluation of a single claim (S1, S2, Cascade)
  - POST /api/evaluate-batch   -> SSE stream of the batch race across S1, S2, and Cascade
  - POST /api/simulate-cascade -> Instant cascade recalculation at any threshold from pre-computed results
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, List, Literal, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

import backends as b
import config
from claims_data import ROUTING_SCHEMA, SEED_CLAIMS, format_claim_state


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up client on startup
    client = b.get_http_client()
    try:
        # Quick non-blocking ping to model server
        resp = await client.get(config.LLAMA_HEALTH_URL, timeout=2.0)
        if resp.status_code == 200:
            # Warm up prefix cache with first seed claim in background
            asyncio.create_task(b.route_local_cygnet(SEED_CLAIMS[0]))
    except Exception:
        pass
    yield
    # Cleanup on shutdown
    await b.close_http_client()


app = FastAPI(
    title="System 1 vs System 2 Claims Router Demo",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HTML_PATH = Path(__file__).resolve().parent / "index.html"


# ---------------------------------------------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------------------------------------------
class SingleEvaluationRequest(BaseModel):
    claim_id: Optional[str] = None
    claim: Optional[dict[str, Any]] = None
    backend: Literal["local", "cloud"] = "local"
    modes: List[Literal["s1", "s2", "cascade"]] = Field(default_factory=lambda: ["s1", "s2", "cascade"])
    threshold: float = Field(default=config.DEFAULT_THRESHOLD, ge=0.50, le=0.99)


class BatchEvaluationRequest(BaseModel):
    count: Literal[10, 25, 50, 100] = 25
    backend: Literal["local", "cloud"] = "local"
    threshold: float = Field(default=config.DEFAULT_THRESHOLD, ge=0.50, le=0.99)
    concurrency: int = Field(default=4, ge=1, le=8)


class SimulateCascadeRequest(BaseModel):
    threshold: float = Field(..., ge=0.50, le=0.99)
    s1: List[dict[str, Any]]
    s2: List[dict[str, Any]]
    claims: List[dict[str, Any]]


# ---------------------------------------------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    if not HTML_PATH.exists():
        return HTMLResponse("<h1>index.html not found. Building dashboard...</h1>", status_code=503)
    return FileResponse(HTML_PATH, media_type="text/html")


@app.get("/api/claims")
async def get_claims_metadata():
    """Returns seed claims for the inspector, routing schema, and runtime config."""
    return {
        "schema": ROUTING_SCHEMA,
        "seed_claims": SEED_CLAIMS,
        "default_threshold": config.DEFAULT_THRESHOLD,
        "temperature": config.CYGNET_TEMPERATURE,
        "has_openai_key": bool(config.OPENAI_API_KEY),
        "claims_per_day": config.CLAIMS_PER_DAY,
    }


@app.get("/api/telemetry")
async def get_telemetry():
    """Returns live GPU metrics and model server connectivity."""
    gpu = b.get_gpu_telemetry()
    llama_ok = False
    client = b.get_http_client()
    try:
        r = await client.get(config.LLAMA_HEALTH_URL, timeout=1.5)
        llama_ok = r.status_code == 200
    except Exception:
        llama_ok = False

    return {
        "gpu": gpu,
        "llama_server": {
            "status": "online" if llama_ok else "offline",
            "url": config.LLAMA_BASE_URL,
            "model": config.LOCAL_MODEL_ALIAS,
        },
        "cloud": {
            "configured": bool(config.OPENAI_API_KEY),
            "model_decisions": config.OPENAI_DECISIONS_MODEL,
            "model_chat": config.OPENAI_CHAT_MODEL,
        },
    }


@app.post("/api/evaluate-single")
async def evaluate_single(req: SingleEvaluationRequest):
    """Evaluates a single claim across the requested modes."""
    # Resolve claim object
    claim = req.claim
    if not claim and req.claim_id:
        seed = next((c for c in SEED_CLAIMS if c["claim_id"] == req.claim_id), None)
        if seed:
            claim = seed.copy()
    if not claim:
        raise HTTPException(status_code=400, detail="Must provide either a valid claim_id or a claim payload.")

    if req.backend == "cloud" and not config.OPENAI_API_KEY:
        raise HTTPException(
            status_code=400,
            detail="OpenAI API key is not configured. Set OPENAI_API_KEY in .env to use Cloud mode.",
        )

    out: dict[str, Any] = {
        "claim": {
            "claim_id": claim.get("claim_id", "CUSTOM"),
            "ground_truth_queue": claim.get("ground_truth_queue"),
            "ground_truth_missing_docs": claim.get("ground_truth_missing_docs"),
            "ground_truth_urgency": claim.get("ground_truth_urgency"),
        },
        "backend": req.backend,
        "threshold": req.threshold,
    }

    try:
        # Run requested evaluations
        if "s1" in req.modes:
            out["s1"] = await b.route_system1(claim, req.backend)
        if "s2" in req.modes:
            out["s2"] = await b.route_system2(claim, req.backend)
        if "cascade" in req.modes:
            if "s1" in out and "s2" in out:
                out["cascade"] = b.combine_cascade(out["s1"], out["s2"], req.threshold)
            else:
                out["cascade"] = await b.route_cascade(claim, req.threshold, req.backend)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Inference error ({type(exc).__name__}): {exc}")

    return out


@app.post("/api/evaluate-batch")
async def evaluate_batch(req: BatchEvaluationRequest):
    """Streams live batch evaluation progress via Server-Sent Events (SSE)."""
    if req.backend == "cloud" and not config.OPENAI_API_KEY:
        raise HTTPException(
            status_code=400,
            detail="OpenAI API key is not configured. Set OPENAI_API_KEY in .env to use Cloud mode.",
        )

    async def event_generator() -> AsyncIterator[str]:
        try:
            async for ev in b.run_batch(
                count=req.count,
                backend=req.backend,
                threshold=req.threshold,
                concurrency=req.concurrency,
            ):
                yield f"data: {json.dumps(ev)}\n\n"
        except Exception as exc:
            err = {"event": "error", "message": f"{type(exc).__name__}: {str(exc)}"}
            yield f"data: {json.dumps(err)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/simulate-cascade")
async def simulate_cascade_endpoint(req: SimulateCascadeRequest):
    """Recomputes cascade metrics at a new threshold without running any model inference."""
    try:
        sim = b.simulate_cascade(req.s1, req.s2, req.claims, req.threshold)
        return {
            "threshold": req.threshold,
            "summary": sim["summary"],
            "results": [
                {
                    "claim_id": r["claim_id"],
                    "routing_queue": r["routing_queue"],
                    "escalated_to_llm": r.get("escalated_to_llm", False),
                    "confidence": r.get("confidence"),
                    "latency_ms": r["latency_ms"],
                }
                for r in sim["results"]
            ],
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Simulation error: {exc}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8080, reload=False, access_log=True)
