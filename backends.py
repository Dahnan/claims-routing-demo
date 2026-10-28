"""Unified routing engine: Cygnet (System 1), Gemma 4 12B (System 2), confidence cascade, OpenAI cloud, telemetry.

Both local paths hit the SAME llama-server instance (same weights), so latency and cost differences come from the
inference paradigm, not from parameter count. Timers (``time.perf_counter``) run server-side next to the GPU.
"""
from __future__ import annotations

import asyncio
import json
import statistics
import time
from typing import Any, AsyncIterator, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

import config
import cygnet_readout
from claims_data import ROUTING_SCHEMA, SEED_CLAIMS, format_claim_state, get_batch_claims

# ---------------------------------------------------------------------------------------------------------------
# HTTP client (strictly 127.0.0.1 for local: avoids Windows IPv6 'localhost' lag)
# ---------------------------------------------------------------------------------------------------------------
_http_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=5.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=32),
        )
    return _http_client


async def close_http_client() -> None:
    global _http_client
    if _http_client is not None and not _http_client.is_closed:
        await _http_client.aclose()
    _http_client = None


def _cost(in_tok: int, out_tok: int, kind: Literal["decisions", "chat"]) -> float:
    p = config.PRICING[kind]
    return round((in_tok * p["input"] + out_tok * p["output"]) / 1_000_000, 6)


# ---------------------------------------------------------------------------------------------------------------
# 1. GPU telemetry (nvml.dll via nvidia-ml-py) + energy sampler
# ---------------------------------------------------------------------------------------------------------------
_nvml_ready: bool | None = None
_gpu_handle: Any = None


def _nvml() -> Any | None:
    """Initialise NVML once. Returns the pynvml module, or None when no NVIDIA GPU/driver is available."""
    global _nvml_ready, _gpu_handle
    if _nvml_ready is False:
        return None
    try:
        import pynvml

        if _nvml_ready is None:
            pynvml.nvmlInit()
            _gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            _nvml_ready = True
        return pynvml
    except Exception:
        _nvml_ready = False
        return None


def get_gpu_telemetry() -> dict[str, Any]:
    """Live VRAM / utilisation / power. When NVML is unavailable every metric is None (never a made-up number)."""
    nv = _nvml()
    if nv is None:
        return {"available": False, "gpu_name": None, "vram_used_gb": None, "vram_total_gb": None,
                "gpu_util_pct": None, "power_watts": None}
    mem = nv.nvmlDeviceGetMemoryInfo(_gpu_handle)
    name = nv.nvmlDeviceGetName(_gpu_handle)
    return {
        "available": True,
        "gpu_name": name.decode() if isinstance(name, bytes) else name,
        "vram_used_gb": round(mem.used / 1024**3, 2),
        "vram_total_gb": round(mem.total / 1024**3, 1),
        "gpu_util_pct": nv.nvmlDeviceGetUtilizationRates(_gpu_handle).gpu,
        "power_watts": round(nv.nvmlDeviceGetPowerUsage(_gpu_handle) / 1000.0, 1),
    }


class PowerSampler:
    """Async context manager: samples GPU power every ``interval`` s and integrates to Watt-hours."""

    def __init__(self, interval: float = 0.1) -> None:
        self.interval = interval
        self.watt_hours: float | None = None
        self.avg_watts: float | None = None
        self._samples: list[tuple[float, float]] = []
        self._task: asyncio.Task | None = None

    async def _run(self) -> None:
        nv = _nvml()
        while nv is not None:
            self._samples.append((time.perf_counter(), nv.nvmlDeviceGetPowerUsage(_gpu_handle) / 1000.0))
            await asyncio.sleep(self.interval)

    async def __aenter__(self) -> "PowerSampler":
        self._task = asyncio.create_task(self._run())
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        s = self._samples
        if len(s) >= 2:
            joules = sum((t1 - t0) * (w0 + w1) / 2 for (t0, w0), (t1, w1) in zip(s, s[1:]))
            self.watt_hours = joules / 3600.0
            self.avg_watts = joules / (s[-1][0] - s[0][0])


# ---------------------------------------------------------------------------------------------------------------
# 2. SYSTEM 1 (LOCAL): Cygnet one-token readout on Gemma 4 12B
# ---------------------------------------------------------------------------------------------------------------
QUEUES = list(ROUTING_SCHEMA["routing_queue"]["options"])
URGENCY = list(ROUTING_SCHEMA["urgency_tier"]["options"])


async def route_local_cygnet(claim: dict[str, Any]) -> dict[str, Any]:
    """All 3 questions as 1-token forward passes, sharing the claim-state prompt prefix."""
    client = get_http_client()
    state = format_claim_state(claim)
    t0 = time.perf_counter()
    # Sequential passes on the same slot allow llama-server (--cache-reuse 256)
    # to reuse the ~700-token claim state KV cache across the 3 questions.
    q = await cygnet_readout.read_question(client, state, ROUTING_SCHEMA["routing_queue"])
    d = await cygnet_readout.read_question(client, state, ROUTING_SCHEMA["missing_clinical_docs"])
    u = await cygnet_readout.read_question(client, state, ROUTING_SCHEMA["urgency_tier"])
    latency_ms = (time.perf_counter() - t0) * 1000.0
    # Input tokens represent the shared claim state prompt (~750 tokens), matching Decisions API
    in_tok = max(r["usage"]["input_tokens"] for r in (q, d, u))
    out_tok = sum(r["usage"]["output_tokens"] for r in (q, d, u))
    return {
        "engine": "Local System 1: Cygnet (Gemma 4 12B one-token readout)",
        "mode": "system1_local",
        "claim_id": claim["claim_id"],
        "routing_queue": q["choice"],
        "confidence": q["p_max"],  # calibrated top probability: the cascade gate
        "confidence_norm": q["confidence"],  # recipe's (K*pmax-1)/(K-1)
        "queue_probabilities": q["probabilities"],
        "missing_clinical_docs": d["choice"],
        "missing_docs_prob": d["noul"],
        "urgency_tier": u["choice"],
        "urgency_probabilities": u["probabilities"],
        "urgency_score": u["score"],
        "clinical_rationale": None,  # a System 1 model does not generate prose
        "latency_ms": round(latency_ms, 1),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "estimated_cloud_cost_usd": _cost(in_tok, out_tok, "decisions"),
    }


# ---------------------------------------------------------------------------------------------------------------
# 3. SYSTEM 2 (LOCAL): Gemma 4 12B autoregressive structured JSON
# ---------------------------------------------------------------------------------------------------------------
class LLMClaimRoutingOutput(BaseModel):
    """Decision fields come first so they are generated before the rationale that explains them."""
    model_config = ConfigDict(extra="forbid")

    routing_queue: Literal[
        "auto_adjudicate", "clinical_nurse_review", "coding_audit_ncci", "cob_subrogation", "siu_fraud_review"
    ]
    missing_clinical_docs: Literal["yes", "no"]
    urgency_tier: Literal["standard_30_day", "expedited_72_hr", "stat_24_hr"]
    clinical_rationale: str = Field(
        description="Exactly 2 sentences of clinical and coding reasoning justifying the routing decision."
    )


def _system2_prompt() -> str:
    parts = ["You are a senior US health insurance claims adjudication router. Analyze the claim and return a JSON "
             "object with your routing decision. Definitions:"]
    for name, spec in ROUTING_SCHEMA.items():
        parts.append(f"\n{name}: {spec['instructions']}")
        for key, desc in spec["options"].items():
            parts.append(f"  - {key}: {desc}")
    return "\n".join(parts)


def _parse_json_content(content: str | None, message: dict[str, Any]) -> dict[str, Any]:
    if not content:
        raise RuntimeError(
            "System 2 returned empty content (reasoning_content length "
            f"{len(message.get('reasoning_content') or '')}): is the model server running with thinking off?"
        )
    return LLMClaimRoutingOutput.model_validate_json(content).model_dump()


async def route_local_gemma4(claim: dict[str, Any]) -> dict[str, Any]:
    client = get_http_client()
    body = {
        "model": config.LOCAL_MODEL_ALIAS,
        "messages": [
            {"role": "system", "content": _system2_prompt()},
            {"role": "user", "content": format_claim_state(claim)},
        ],
        "temperature": 0.1,
        "max_tokens": 400,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "claim_routing", "strict": True, "schema": LLMClaimRoutingOutput.model_json_schema()},
        },
    }
    t0 = time.perf_counter()
    resp = await client.post(config.LLAMA_CHAT_URL, json=body)
    resp.raise_for_status()
    latency_ms = (time.perf_counter() - t0) * 1000.0
    data = resp.json()
    msg = data["choices"][0]["message"]
    parsed = _parse_json_content(msg.get("content"), msg)
    usage = data.get("usage") or {}
    in_tok, out_tok = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
    return {
        "engine": "Local System 2: Gemma 4 12B (structured JSON generation)",
        "mode": "system2_local",
        "claim_id": claim["claim_id"],
        **{k: parsed[k] for k in ("routing_queue", "missing_clinical_docs", "urgency_tier", "clinical_rationale")},
        "confidence": None,  # generated text carries no calibrated multi-class probabilities
        "confidence_norm": None,
        "queue_probabilities": None,
        "missing_docs_prob": None,
        "urgency_probabilities": None,
        "urgency_score": None,
        "latency_ms": round(latency_ms, 1),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "estimated_cloud_cost_usd": _cost(in_tok, out_tok, "chat"),
    }


# ---------------------------------------------------------------------------------------------------------------
# 4. COMMERCIAL CLOUD: OpenAI Decisions API (System 1 style) and GPT-6 Sol chat (System 2 style)
# ---------------------------------------------------------------------------------------------------------------
def _openai_headers() -> dict[str, str]:
    if not config.OPENAI_API_KEY:
        raise RuntimeError("Set OPENAI_API_KEY (in .env or the environment) to use Commercial Cloud mode.")
    return {"Authorization": f"Bearer {config.OPENAI_API_KEY}", "Content-Type": "application/json"}


def _decisions_payload(claim: dict[str, Any]) -> dict[str, Any]:
    """Request shape validated against the live API: choices use value+description, levels use label+description."""
    s = ROUTING_SCHEMA
    return {
        "model": config.OPENAI_DECISIONS_MODEL,
        "input": format_claim_state(claim),
        "questions": [
            {"name": "routing_queue", "type": "choice", "instructions": s["routing_queue"]["instructions"],
             "choices": [{"value": k, "description": v} for k, v in s["routing_queue"]["options"].items()]},
            {"name": "missing_clinical_docs", "type": "predicate",
             "instructions": s["missing_clinical_docs"]["instructions"]},
            {"name": "urgency_tier", "type": "score", "instructions": s["urgency_tier"]["instructions"],
             "levels": [{"label": k, "description": v} for k, v in s["urgency_tier"]["options"].items()]},
        ],
    }


def _first(d: dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if d.get(k) is not None:
            return d[k]
    return None


def _answer(data: dict[str, Any], name: str) -> dict[str, Any]:
    """Find a named answer in a Decisions response. The response shape could not be verified live (the test account
    had no credits), so this accepts the plausible layouts and fails loudly on anything else."""
    container = _first(data, "answers", "results", "decisions", "output")
    if isinstance(container, dict) and isinstance(container.get(name), dict):
        return container[name]
    if isinstance(container, list):
        for item in container:
            if isinstance(item, dict) and item.get("name") == name:
                return item
    raise RuntimeError(f"Unrecognised Decisions API response (no answer named {name!r}): {json.dumps(data)[:500]}")


def _probs(ans: dict[str, Any], labels: list[str]) -> dict[str, float]:
    raw = _first(ans, "probabilities", "probs", "distribution")
    if isinstance(raw, dict):
        out = {k: float(raw.get(k, 0.0)) for k in labels}
    elif isinstance(raw, list) and raw and isinstance(raw[0], dict):
        by = {str(_first(e, "label", "value", "name")): float(_first(e, "probability", "p") or 0.0) for e in raw}
        out = {k: by.get(k, 0.0) for k in labels}
    elif isinstance(raw, list) and len(raw) == len(labels):
        out = {k: float(p) for k, p in zip(labels, raw)}
    else:
        raise RuntimeError(f"Decisions answer has no probabilities: {json.dumps(ans)[:300]}")
    s = sum(out.values()) or 1.0
    return {k: v / s for k, v in out.items()}


async def route_openai_cloud(claim: dict[str, Any], use_decisions_api: bool = True) -> dict[str, Any]:
    headers = _openai_headers()
    client = get_http_client()
    if use_decisions_api:
        t0 = time.perf_counter()
        resp = await client.post(f"{config.OPENAI_BASE_URL}/decisions", headers=headers, json=_decisions_payload(claim))
        resp.raise_for_status()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        data = resp.json()
        qa, da, ua = (_answer(data, n) for n in ("routing_queue", "missing_clinical_docs", "urgency_tier"))
        qp = _probs(qa, QUEUES)
        top = max(qp, key=qp.__getitem__)
        up = _probs(ua, URGENCY)
        p_docs = _first(da, "probability", "p", "noul")
        if p_docs is None:
            raise RuntimeError(f"Decisions predicate answer has no probability: {json.dumps(da)[:300]}")
        usage = data.get("usage") or {}
        in_tok = int(_first(usage, "input_tokens", "prompt_tokens") or 0)
        out_tok = int(_first(usage, "output_tokens", "completion_tokens") or 0)
        k = len(QUEUES)
        return {
            "engine": f"Cloud System 1: OpenAI Decisions API ({config.OPENAI_DECISIONS_MODEL})",
            "mode": "system1_cloud", "claim_id": claim["claim_id"],
            "routing_queue": top, "confidence": qp[top],
            "confidence_norm": max(0.0, min(1.0, (k * qp[top] - 1) / (k - 1))),
            "queue_probabilities": qp,
            "missing_clinical_docs": "yes" if float(p_docs) >= 0.5 else "no", "missing_docs_prob": float(p_docs),
            "urgency_tier": max(up, key=up.__getitem__), "urgency_probabilities": up,
            "urgency_score": sum(i * up[u] for i, u in enumerate(URGENCY)),
            "clinical_rationale": None,
            "latency_ms": round(latency_ms, 1), "input_tokens": in_tok, "output_tokens": out_tok,
            "estimated_cloud_cost_usd": _cost(in_tok, out_tok, "decisions"),
        }
    body = {
        "model": config.OPENAI_CHAT_MODEL,
        "messages": [{"role": "system", "content": _system2_prompt()},
                     {"role": "user", "content": format_claim_state(claim)}],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "claim_routing", "strict": True, "schema": LLMClaimRoutingOutput.model_json_schema()}},
    }
    t0 = time.perf_counter()
    resp = await client.post(f"{config.OPENAI_BASE_URL}/chat/completions", headers=headers, json=body)
    resp.raise_for_status()
    latency_ms = (time.perf_counter() - t0) * 1000.0
    data = resp.json()
    msg = data["choices"][0]["message"]
    parsed = _parse_json_content(msg.get("content"), msg)
    usage = data.get("usage") or {}
    in_tok, out_tok = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
    return {
        "engine": f"Cloud System 2: OpenAI {config.OPENAI_CHAT_MODEL} (Chat Completions)",
        "mode": "system2_cloud", "claim_id": claim["claim_id"],
        **{k: parsed[k] for k in ("routing_queue", "missing_clinical_docs", "urgency_tier", "clinical_rationale")},
        "confidence": None, "confidence_norm": None, "queue_probabilities": None, "missing_docs_prob": None,
        "urgency_probabilities": None, "urgency_score": None,
        "latency_ms": round(latency_ms, 1), "input_tokens": in_tok, "output_tokens": out_tok,
        "estimated_cloud_cost_usd": _cost(in_tok, out_tok, "chat"),
    }


# ---------------------------------------------------------------------------------------------------------------
# 5. Backend selection + CONFIDENCE CASCADE
# ---------------------------------------------------------------------------------------------------------------
Backend = Literal["local", "cloud"]


async def route_system1(claim: dict[str, Any], backend: Backend = "local") -> dict[str, Any]:
    return await (route_local_cygnet(claim) if backend == "local" else route_openai_cloud(claim, True))


async def route_system2(claim: dict[str, Any], backend: Backend = "local") -> dict[str, Any]:
    return await (route_local_gemma4(claim) if backend == "local" else route_openai_cloud(claim, False))


def combine_cascade(s1: dict[str, Any], s2: dict[str, Any] | None, threshold: float) -> dict[str, Any]:
    """Pure cascade logic: System 1 answers when p_max >= threshold, otherwise System 2 adjudicates."""
    if s2 is None or s1["confidence"] >= threshold:
        return {**s1, "mode": "cascade_s1", "escalated_to_llm": False,
                "engine": f"Cascade: handled by System 1 (confidence {s1['confidence']:.2f} >= {threshold:.2f})"}
    return {
        "engine": f"Cascade: escalated to System 2 (System 1 confidence {s1['confidence']:.2f} < {threshold:.2f})",
        "mode": "cascade_escalated", "escalated_to_llm": True, "claim_id": s1["claim_id"],
        "routing_queue": s2["routing_queue"], "missing_clinical_docs": s2["missing_clinical_docs"],
        "urgency_tier": s2["urgency_tier"], "clinical_rationale": s2["clinical_rationale"],
        "confidence": s1["confidence"], "confidence_norm": s1["confidence_norm"],
        "queue_probabilities": s1["queue_probabilities"], "missing_docs_prob": s1["missing_docs_prob"],
        "urgency_probabilities": s1["urgency_probabilities"], "urgency_score": s1["urgency_score"],
        "latency_ms": round(s1["latency_ms"] + s2["latency_ms"], 1),
        "input_tokens": s1["input_tokens"] + s2["input_tokens"],
        "output_tokens": s1["output_tokens"] + s2["output_tokens"],
        "estimated_cloud_cost_usd": round(s1["estimated_cloud_cost_usd"] + s2["estimated_cloud_cost_usd"], 6),
    }


async def route_cascade(claim: dict[str, Any], threshold: float = config.DEFAULT_THRESHOLD,
                        backend: Backend = "local") -> dict[str, Any]:
    s1 = await route_system1(claim, backend)
    if s1["confidence"] >= threshold:
        return combine_cascade(s1, None, threshold)
    return combine_cascade(s1, await route_system2(claim, backend), threshold)


# ---------------------------------------------------------------------------------------------------------------
# 6. BATCH RACE
# ---------------------------------------------------------------------------------------------------------------
def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    return round(v[min(len(v) - 1, max(0, round(q * (len(v) - 1))))], 1)


def summarize(results: list[dict[str, Any]], claims: list[dict[str, Any]], *,
              energy_wh: float | None = None, wall_s: float | None = None) -> dict[str, Any]:
    """Aggregate accuracy / latency / cost for one lane (results aligned with claims)."""
    n = len(results)
    gt = {c["claim_id"]: c for c in claims}
    hit = lambda key, gkey: sum(1 for r in results if r[key] == gt[r["claim_id"]][gkey])  # noqa: E731
    lat = [r["latency_ms"] for r in results]
    cost = sum(r["estimated_cloud_cost_usd"] for r in results)
    per_claim = cost / n if n else 0.0
    return {
        "count": n,
        "accuracy": round(hit("routing_queue", "ground_truth_queue") / n, 4) if n else None,
        "docs_accuracy": round(hit("missing_clinical_docs", "ground_truth_missing_docs") / n, 4) if n else None,
        "urgency_accuracy": round(hit("urgency_tier", "ground_truth_urgency") / n, 4) if n else None,
        "latency_p50_ms": _pct(lat, 0.5), "latency_p95_ms": _pct(lat, 0.95),
        "latency_mean_ms": round(statistics.fmean(lat), 1) if lat else None,
        "escalation_rate": round(sum(1 for r in results if r.get("escalated_to_llm")) / n, 4) if n else None,
        "input_tokens": sum(r["input_tokens"] for r in results),
        "output_tokens": sum(r["output_tokens"] for r in results),
        "cost_usd": round(cost, 6),
        "annual_cost_usd": round(per_claim * config.CLAIMS_PER_DAY * 365, 2),
        "energy_wh": None if energy_wh is None else round(energy_wh, 3),
        "wall_seconds": None if wall_s is None else round(wall_s, 2),
    }


def simulate_cascade(s1: list[dict[str, Any]], s2: list[dict[str, Any]], claims: list[dict[str, Any]],
                     threshold: float) -> dict[str, Any]:
    """Re-derive the cascade from already-measured System 1 / System 2 results at any threshold (no inference)."""
    results = [combine_cascade(a, b, threshold) for a, b in zip(s1, s2)]
    return {"results": results, "summary": summarize(results, claims)}


async def _run_lane(fn: Any, claims: list[dict[str, Any]], backend: Backend, concurrency: int,
                    lane: str) -> AsyncIterator[dict[str, Any]]:
    # Cap local concurrency to 4 slots to avoid llama-server contention
    effective_concurrency = min(concurrency, 4) if backend == "local" else concurrency
    sem = asyncio.Semaphore(effective_concurrency)
    out: list[dict[str, Any] | None] = [None] * len(claims)

    async def one(i: int, c: dict[str, Any]) -> int:
        async with sem:
            out[i] = await fn(c, backend)
        return i

    async with PowerSampler() as power:
        t0 = time.perf_counter()
        tasks = [asyncio.create_task(one(i, c)) for i, c in enumerate(claims)]
        done = 0
        try:
            for fut in asyncio.as_completed(tasks):
                i = await fut
                done += 1
                r = out[i]
                yield {"event": "progress", "lane": lane, "done": done, "total": len(claims),
                       "claim_id": r["claim_id"], "routing_queue": r["routing_queue"],
                       "correct": r["routing_queue"] == claims[i]["ground_truth_queue"],
                       "latency_ms": r["latency_ms"]}
        finally:
            for t in tasks:
                t.cancel()
        wall = time.perf_counter() - t0
    summary = summarize(out, claims, energy_wh=power.watt_hours, wall_s=wall)  # type: ignore[arg-type]
    yield {"event": "lane_done", "lane": lane, "summary": summary, "results": out,
           "avg_watts": None if power.avg_watts is None else round(power.avg_watts, 1)}


async def run_batch(count: int = 100, backend: Backend = "local", threshold: float = config.DEFAULT_THRESHOLD,
                    concurrency: int = 4) -> AsyncIterator[dict[str, Any]]:
    """Race System 1, System 2 and the Cascade over a deterministic batch, streaming progress events.

    Lanes run one after another (so they never contend for the GPU and distort each other's latency). The cascade
    lane is derived from the two measured lanes, which is exact: it runs the same System 1 call and, on escalation,
    the same System 2 call, and its latency is their sum.
    """
    claims = get_batch_claims(count)
    lanes: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, Any] = {}
    yield {"event": "start", "count": len(claims), "backend": backend, "threshold": threshold}
    for lane, fn in (("s2", route_system2), ("s1", route_system1)):
        yield {"event": "lane_start", "lane": lane}
        async for ev in _run_lane(fn, claims, backend, concurrency, lane):
            if ev["event"] == "lane_done":
                lanes[lane], summaries[lane] = ev["results"], ev["summary"]
                yield {k: v for k, v in ev.items() if k != "results"}
            else:
                yield ev
    casc = simulate_cascade(lanes["s1"], lanes["s2"], claims, threshold)
    summaries["cascade"] = casc["summary"]
    yield {"event": "lane_done", "lane": "cascade", "summary": casc["summary"]}
    yield {"event": "done", "summaries": summaries, "threshold": threshold, "claims_per_day": config.CLAIMS_PER_DAY,
           "claims": [{"claim_id": c["claim_id"], "ground_truth_queue": c["ground_truth_queue"],
                       "difficulty": c["difficulty"]} for c in claims],
           "s1": [{k: r[k] for k in ("claim_id", "routing_queue", "confidence", "latency_ms",
                                     "estimated_cloud_cost_usd")} for r in lanes["s1"]],
           "s2": [{k: r[k] for k in ("claim_id", "routing_queue", "latency_ms",
                                     "estimated_cloud_cost_usd")} for r in lanes["s2"]]}


# ---------------------------------------------------------------------------------------------------------------
# 7. CLI SMOKE TEST: `python backends.py`
# ---------------------------------------------------------------------------------------------------------------
async def _smoke_test() -> None:
    print("=" * 78)
    print("CLAIMS ROUTER SMOKE TEST (local llama-server)")
    print("=" * 78)
    gpu = get_gpu_telemetry()
    if gpu["available"]:
        print(f"GPU : {gpu['gpu_name']} | VRAM {gpu['vram_used_gb']}/{gpu['vram_total_gb']} GB | {gpu['power_watts']} W")
    else:
        print("GPU : NVML unavailable (telemetry disabled)")
    print(f"Cascade threshold {config.DEFAULT_THRESHOLD} | Cygnet temperature {config.CYGNET_TEMPERATURE}\n")

    print("Warming up (KV cache / prompt prefix)...")
    await route_local_cygnet(SEED_CLAIMS[0])
    await route_local_gemma4(SEED_CLAIMS[0])

    print(f"\n{'claim':9} {'truth':23} {'System 1 pick':23} {'p_max':>6} {'ms':>6}  cascade")
    s1_ms: list[float] = []
    for c in SEED_CLAIMS:
        s1 = await route_local_cygnet(c)
        s1_ms.append(s1["latency_ms"])
        ok = "OK " if s1["routing_queue"] == c["ground_truth_queue"] else "BAD"
        esc = "-> System 2" if s1["confidence"] < config.DEFAULT_THRESHOLD else "auto"
        print(f"{c['claim_id']:9} {c['ground_truth_queue']:23} {s1['routing_queue']:23} {s1['confidence']:6.3f} "
              f"{s1['latency_ms']:6.0f}  {ok} {esc}")

    test = SEED_CLAIMS[1]
    s2 = await route_local_gemma4(test)
    print(f"\nSystem 2 on {test['claim_id']}: {s2['routing_queue']} in {s2['latency_ms']:.0f} ms "
          f"({s2['output_tokens']} output tokens)\n  rationale: {s2['clinical_rationale']}")
    p50 = statistics.median(s1_ms)
    print(f"\nSystem 1 median {p50:.0f} ms vs System 2 {s2['latency_ms']:.0f} ms = "
          f"{s2['latency_ms'] / max(p50, 1.0):.1f}x faster")
    await close_http_client()


if __name__ == "__main__":
    asyncio.run(_smoke_test())