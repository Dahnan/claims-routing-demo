import os
import sys
import math
import json
import time
import asyncio
from typing import Dict, Any, List
import httpx
from pydantic import BaseModel, Field
from cygnet-recipe import Cygnet

from claims_data import ROUTING_SCHEMA, SEED_CLAIMS, format_claim_state

# Initialize Cygnet using your local model weights or llama-server endpoint
cygnet = CygnetEvaluator(
    endpoint="http://127.0.0.1:8000"
    device="cuda"
)

# Cygnet evaluates the question and returns calibrated probabilities directly:
result = cygnet.evaluate(
    state=claim_text,
    question=ROUTING_SCHEMA["routing_queue"]["instructions"],
    options=ROUTING_SCHEMA["routing_queue"]["options"]
)

# result contains: {"choice": "cob_subrogation", "probabilities": {...}, "confidence": 0.94}

# Fix Windows asyncio event loop policy if needed
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Persistent loopback HTTP client (strictly 127.0.0.1 to avoid Windows IPv6 localhost lag)
LOCAL_LLAMA_URL = "http://127.0.0.1:8000/v1"
CYGNET_CALIBRATION_TEMP = 1.052  # Cygnet post-hoc temperature scaling for Gemma 4 12B

_http_client: httpx.AsyncClient | None = None

def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=5.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=32)
        )
    return _http_client


# =====================================================================
# 1. LIVE RTX 4090 TELEMETRY (Windows nvml.dll via nvidia-ml-py)
# =====================================================================
_nvml_initialized = False
_gpu_handle = None

def get_gpu_telemetry() -> Dict[str, Any]:
    global _nvml_initialized, _gpu_handle
    try:
        import pynvml
        if not _nvml_initialized:
            pynvml.nvmlInit()
            _gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            _nvml_initialized = True
        
        mem = pynvml.nvmlDeviceGetMemoryInfo(_gpu_handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(_gpu_handle)
        power_mw = pynvml.nvmlDeviceGetPowerUsage(_gpu_handle)
        name = pynvml.nvmlDeviceGetName(_gpu_handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")
            
        return {
            "gpu_name": name,
            "vram_used_gb": round(mem.used / (1024 ** 3), 2),
            "vram_total_gb": round(mem.total / (1024 ** 3), 1),
            "gpu_util_pct": util.gpu,
            "power_watts": round(power_mw / 1000.0, 1),
            "available": True
        }
    except Exception:
        return {
            "gpu_name": "NVIDIA GeForce RTX 4090 (Simulated/Offline)",
            "vram_used_gb": 15.8,
            "vram_total_gb": 24.0,
            "gpu_util_pct": 0,
            "power_watts": 48.0,
            "available": False
        }


# =====================================================================
# 2. SYSTEM 1 (LOCAL): CYGNET 1-TOKEN LOGPROB EXTRACTOR ON GEMMA 4 12B
# =====================================================================
async def _eval_single_cygnet_question(
    client: httpx.AsyncClient,
    claim_text: str,
    q_name: str,
    q_spec: Dict[str, Any]
) -> Dict[str, Any]:
    options_dict: Dict[str, str] = q_spec["options"]
    option_keys = list(options_dict.keys())
    letters = [chr(ord("A") + i) for i in range(len(option_keys))]
    letter_to_key = dict(zip(letters, option_keys))

    options_block = "\n".join(
        f"{letter}) {key}: {options_dict[key]}"
        for letter, key in zip(letters, option_keys)
    )

    prompt = (
        f"### CLAIM STATE\n{claim_text}\n\n"
        f"### DECISION RUBRIC ({q_name})\n{q_spec['instructions']}\n\n"
        f"### OPTIONS\n{options_block}\n\n"
        f"Respond with ONLY the single capital letter ({', '.join(letters)}) corresponding to the best option."
    )

    payload = {
        "model": "cygnet:12b",
        "messages": [
            {"role": "system", "content": "You are Cygnet, a calibrated System 1 decision router. Output only a single option letter."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": 20,
        "cache_prompt": True
    }

    resp = await client.post(f"{LOCAL_LLAMA_URL}/chat/completions", json=payload)
    resp.raise_for_status()
    data = resp.json()

    # Extract top_logprobs from the single generated token
    raw_logprobs: Dict[str, float] = {k: -15.0 for k in option_keys}
    try:
        top_lp_list = data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        for entry in top_lp_list:
            tok = entry["token"].strip().upper()
            if tok in letter_to_key:
                mapped_key = letter_to_key[tok]
                # Keep the highest logprob if both "A" and " A" appear
                if entry["logprob"] > raw_logprobs[mapped_key]:
                    raw_logprobs[mapped_key] = float(entry["logprob"])
    except (KeyError, IndexError, TypeError):
        # Fallback if logprobs structure differs
        chosen_tok = data["choices"][0]["message"]["content"].strip().upper()[:1]
        chosen_key = letter_to_key.get(chosen_tok, option_keys[0])
        raw_logprobs[chosen_key] = 0.0

    # Apply Cygnet calibration temperature (T = 1.052) and softmax over declared options
    scaled_logits = {k: lp / CYGNET_CALIBRATION_TEMP for k, lp in raw_logprobs.items()}
    max_logit = max(scaled_logits.values())
    exp_scores = {k: math.exp(val - max_logit) for k, val in scaled_logits.items()}
    total_exp = sum(exp_scores.values())
    probs = {k: round(exp_scores[k] / total_exp, 4) for k in option_keys}

    top_choice = max(probs, key=probs.get)
    return {
        "question": q_name,
        "choice": top_choice,
        "confidence": probs[top_choice],
        "probabilities": probs,
        "prompt_tokens": data.get("usage", {}).get("prompt_tokens", 320)
    }


async def route_local_cygnet(claim: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluates all 3 routing questions via Local Cygnet (1-token forward passes) on 127.0.0.1:8000."""
    client = get_http_client()
    claim_text = format_claim_state(claim)

    t0 = time.perf_counter()
    tasks = [
        _eval_single_cygnet_question(client, claim_text, q_name, q_spec)
        for q_name, q_spec in ROUTING_SCHEMA.items()
    ]
    results = await asyncio.gather(*tasks)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    ans_map = {r["question"]: r for r in results}
    q_res = ans_map["routing_queue"]
    doc_res = ans_map["missing_clinical_docs"]
    urg_res = ans_map["urgency_tier"]

    return {
        "engine": "Local System 1: Cygnet (Gemma 4 12B Logprobs)",
        "mode": "system1_local",
        "claim_id": claim["claim_id"],
        "routing_queue": q_res["choice"],
        "confidence": q_res["confidence"],
        "queue_probabilities": q_res["probabilities"],
        "missing_clinical_docs": doc_res["choice"],
        "missing_docs_prob": doc_res["probabilities"].get("yes", 0.0),
        "urgency_tier": urg_res["choice"],
        "clinical_rationale": None,  # System 1 models do not waste tokens generating prose
        "latency_ms": round(latency_ms, 1),
        "input_tokens": q_res["prompt_tokens"],
        "output_tokens": 3,  # 1 token per question
        "estimated_cloud_cost_usd": round((q_res["prompt_tokens"] / 1_000_000) * 0.10, 6)
    }


# =====================================================================
# 3. SYSTEM 2 (LOCAL): GEMMA 4 12B AUTOREGRESSIVE JSON GENERATION
# =====================================================================
class LLMClaimRoutingOutput(BaseModel):
    clinical_rationale: str = Field(
        description="2-sentence clinical and coding explanation justifying the routing decision."
    )
    routing_queue: str = Field(
        description="One of: auto_adjudicate, clinical_nurse_review, coding_audit_ncci, cob_subrogation, siu_fraud_review"
    )
    missing_clinical_docs: str = Field(description="One of: yes, no")
    urgency_tier: str = Field(description="One of: standard_30_day, expedited_72_hr, stat_24_hr")


async def route_local_gemma4(claim: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluates the claim via full System 2 autoregressive JSON generation on 127.0.0.1:8000."""
    client = get_http_client()
    claim_text = format_claim_state(claim)

    rubric_summary = json.dumps(ROUTING_SCHEMA, indent=2)
    system_prompt = (
        "You are a senior US health insurance claims adjudication router. "
        "Analyze the claim and return a valid JSON object matching the required schema.\n"
        f"RUBRIC DEFINITIONS:\n{rubric_summary}"
    )

    payload = {
        "model": "cygnet:12b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": claim_text}
        ],
        "temperature": 0.1,
        "max_tokens": 220,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "claim_routing",
                "strict": True,
                "schema": LLMClaimRoutingOutput.model_json_schema()
            }
        }
    }

    t0 = time.perf_counter()
    resp = await client.post(f"{LOCAL_LLAMA_URL}/chat/completions", json=payload)
    resp.raise_for_status()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    data = resp.json()
    raw_content = data["choices"][0]["message"]["content"]
    parsed = json.loads(raw_content)
    usage = data.get("usage", {})
    in_tok = usage.get("prompt_tokens", 480)
    out_tok = usage.get("completion_tokens", 115)

    return {
        "engine": "Local System 2: Gemma 4 12B (Structured JSON)",
        "mode": "system2_local",
        "claim_id": claim["claim_id"],
        "routing_queue": parsed["routing_queue"],
        "confidence": None,  # Generative JSON does not provide calibrated multi-class probabilities
        "queue_probabilities": None,
        "missing_clinical_docs": parsed["missing_clinical_docs"],
        "missing_docs_prob": None,
        "urgency_tier": parsed["urgency_tier"],
        "clinical_rationale": parsed["clinical_rationale"],
        "latency_ms": round(latency_ms, 1),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "estimated_cloud_cost_usd": round((in_tok * 0.10 + out_tok * 0.50) / 1_000_000, 6)
    }


# =====================================================================
# 4. CONFIDENCE CASCADE (SYSTEM 1 -> THRESHOLD GATE -> SYSTEM 2)
# =====================================================================
async def route_cascade(claim: Dict[str, Any], threshold: float = 0.90) -> Dict[str, Any]:
    """Runs System 1 (Cygnet) first; escalates to System 2 (Gemma 4) only if confidence < threshold."""
    s1_res = await route_local_cygnet(claim)
    if s1_res["confidence"] >= threshold:
        s1_res["escalated_to_llm"] = False
        s1_res["engine"] = f"Cascade: Handled by Cygnet (Conf {s1_res['confidence']:.2f} >= {threshold})"
        return s1_res

    # Low confidence edge case -> Escalate to System 2 LLM for clinical reasoning
    s2_res = await route_local_gemma4(claim)
    return {
        "engine": f"Cascade: Escalated to Gemma 4 (Cygnet Conf {s1_res['confidence']:.2f} < {threshold})",
        "mode": "cascade_escalated",
        "escalated_to_llm": True,
        "claim_id": claim["claim_id"],
        "routing_queue": s2_res["routing_queue"],
        "confidence": s1_res["confidence"],
        "queue_probabilities": s1_res["queue_probabilities"],
        "missing_clinical_docs": s2_res["missing_clinical_docs"],
        "missing_docs_prob": s1_res["missing_docs_prob"],
        "urgency_tier": s2_res["urgency_tier"],
        "clinical_rationale": s2_res["clinical_rationale"],
        "latency_ms": round(s1_res["latency_ms"] + s2_res["latency_ms"], 1),
        "input_tokens": s1_res["input_tokens"] + s2_res["input_tokens"],
        "output_tokens": s2_res["output_tokens"],
        "estimated_cloud_cost_usd": round(
            s1_res["estimated_cloud_cost_usd"] + s2_res["estimated_cloud_cost_usd"], 6
        )
    }


# =====================================================================
# 5. COMMERCIAL CLOUD TOGGLE: OPENAI DECISIONS API & GPT-6 LUNA
# =====================================================================
async def route_openai_cloud(claim: Dict[str, Any], use_decisions_api: bool = True) -> Dict[str, Any]:
    """Optional commercial cloud switch for OpenAI /v1/decisions (gpt-6-luna-decisions) vs gpt-6-luna."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY environment variable to use the Commercial Cloud mode.")

    client = get_http_client()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    claim_text = format_claim_state(claim)

    if use_decisions_api:
        # OpenAI /v1/decisions endpoint (gpt-6-luna-decisions: $0.10/1M input, $0 output)
        payload = {
            "model": "gpt-6-luna-decisions",
            "input": claim_text,
            "questions": [
                {
                    "name": "routing_queue",
                    "type": "choice",
                    "instructions": ROUTING_SCHEMA["routing_queue"]["instructions"],
                    "choices": ROUTING_SCHEMA["routing_queue"]["options"]
                },
                {
                    "name": "missing_clinical_docs",
                    "type": "predicate",
                    "instructions": ROUTING_SCHEMA["missing_clinical_docs"]["instructions"]
                },
                {
                    "name": "urgency_tier",
                    "type": "score",
                    "instructions": ROUTING_SCHEMA["urgency_tier"]["instructions"],
                    "levels": ROUTING_SCHEMA["urgency_tier"]["options"]
                }
            ]
        }
        t0 = time.perf_counter()
        resp = await client.post("https://api.openai.com/v1/decisions", headers=headers, json=payload)
        resp.raise_for_status()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        data = resp.json()
        ans = data["answers"]
        in_tok = data.get("usage", {}).get("input_tokens", 350)
        return {
            "engine": "Cloud System 1: OpenAI Decisions API (gpt-6-luna-decisions)",
            "mode": "system1_cloud",
            "claim_id": claim["claim_id"],
            "routing_queue": ans["routing_queue"]["choice"],
            "confidence": ans["routing_queue"]["probability"],
            "queue_probabilities": ans["routing_queue"]["probabilities"],
            "missing_clinical_docs": "yes" if ans["missing_clinical_docs"]["probability"] >= 0.5 else "no",
            "missing_docs_prob": ans["missing_clinical_docs"]["probability"],
            "urgency_tier": ans["urgency_tier"]["level"],
            "clinical_rationale": None,
            "latency_ms": round(latency_ms, 1),
            "input_tokens": in_tok,
            "output_tokens": 0,
            "estimated_cloud_cost_usd": round((in_tok / 1_000_000) * 0.10, 6)
        }
    else:
        # OpenAI /v1/chat/completions with gpt-6-luna ($0.10/1M in, $0.50/1M out)
        payload = {
            "model": "gpt-6-luna",
            "messages": [
                {"role": "system", "content": f"Route this claim using this schema:\n{json.dumps(ROUTING_SCHEMA)}"},
                {"role": "user", "content": claim_text}
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "claim_routing",
                    "strict": True,
                    "schema": LLMClaimRoutingOutput.model_json_schema()
                }
            }
        }
        t0 = time.perf_counter()
        resp = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        data = resp.json()
        parsed = json.loads(data["choices"][0]["message"]["content"])
        usage = data.get("usage", {})
        in_tok = usage.get("prompt_tokens", 480)
        out_tok = usage.get("completion_tokens", 120)
        return {
            "engine": "Cloud System 2: OpenAI GPT-6 Luna (Chat Completions)",
            "mode": "system2_cloud",
            "claim_id": claim["claim_id"],
            "routing_queue": parsed["routing_queue"],
            "confidence": None,
            "queue_probabilities": None,
            "missing_clinical_docs": parsed["missing_clinical_docs"],
            "missing_docs_prob": None,
            "urgency_tier": parsed["urgency_tier"],
            "clinical_rationale": parsed["clinical_rationale"],
            "latency_ms": round(latency_ms, 1),
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "estimated_cloud_cost_usd": round((in_tok * 0.10 + out_tok * 0.50) / 1_000_000, 6)
        }


# =====================================================================
# 6. CLI SMOKE TEST (Run `python backends.py` to verify your RTX 4090!)
# =====================================================================
async def _smoke_test():
    print("=" * 72)
    print("RTX 4090 HEALTH CLAIMS ROUTER - SMOKE TEST")
    print("=" * 72)
    gpu = get_gpu_telemetry()
    print(f"GPU Detected : {gpu['gpu_name']}")
    print(f"VRAM Usage   : {gpu['vram_used_gb']} GB / {gpu['vram_total_gb']} GB | Power: {gpu['power_watts']} W\n")

    test_claim = SEED_CLAIMS[1]  # CLM-1002: Tibial fracture at grocery store (Subrogation)
    print(f"Testing Claim: {test_claim['claim_id']} - {test_claim['title']}")
    print(f"Ground Truth : {test_claim['ground_truth_queue']}\n")

    # Warm-up pass so CUDA graphs & KV prefix cache are hot
    print("Warming up CUDA kernels & prompt prefix cache...")
    await route_local_cygnet(SEED_CLAIMS[0])

    print("\n1. Running Local System 1 (Cygnet on Gemma 4 12B)...")
    s1 = await route_local_cygnet(test_claim)
    print(f"   -> Queue Chosen : {s1['routing_queue']} (Confidence: {s1['confidence']*100:.1f}%)")
    print(f"   -> Probabilities: {json.dumps(s1['queue_probabilities'])}")
    print(f"   -> Missing Docs : {s1['missing_clinical_docs']} (Prob: {s1['missing_docs_prob']*100:.1f}%)")
    print(f"   -> Latency      : {s1['latency_ms']} ms")

    print("\n2. Running Local System 2 (Gemma 4 12B Structured JSON)...")
    s2 = await route_local_gemma4(test_claim)
    print(f"   -> Queue Chosen : {s2['routing_queue']}")
    print(f"   -> Rationale    : {s2['clinical_rationale']}")
    print(f"   -> Latency      : {s2['latency_ms']} ms")
    print(f"\nSpeedup Factor : {s2['latency_ms'] / max(s1['latency_ms'], 1.0):.1f}x faster with System 1!")
    print("=" * 72)

if __name__ == "__main__":
    asyncio.run(_smoke_test())