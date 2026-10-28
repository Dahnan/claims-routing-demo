"""Central configuration for the System 1 vs System 2 claims router demo.

Every value can be overridden through environment variables or a repo-root ``.env`` file.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")


def _f(name: str, default: float) -> float:
    return float(os.getenv(name, default))


# --- Local model server (llama-server, strictly loopback to avoid Windows IPv6 localhost lag) ---
LLAMA_BASE_URL = os.getenv("LLAMA_BASE_URL", "http://127.0.0.1:8000")
LLAMA_CHAT_URL = f"{LLAMA_BASE_URL}/v1/chat/completions"
LLAMA_HEALTH_URL = f"{LLAMA_BASE_URL}/health"
LOCAL_MODEL_ALIAS = os.getenv("LOCAL_MODEL_ALIAS", "cygnet:12b")

# --- Cygnet readout (cygnet-recipe: one calibration temperature, p^(1/T) renormalised) ---
CYGNET_TEMPERATURE = _f("CYGNET_TEMPERATURE", 3.4)
TOP_LOGPROBS = int(os.getenv("CYGNET_TOP_LOGPROBS", 20))

# --- Cascade: measured in Phase 0 on the seed claims (clear-cut >= 0.94, CLM-1008 ~0.53) ---
DEFAULT_THRESHOLD = _f("CASCADE_THRESHOLD", 0.90)

# --- OpenAI commercial cloud toggle ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_DECISIONS_MODEL = os.getenv("OPENAI_DECISIONS_MODEL", "gpt-6-luna")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-6-sol")

# --- Pricing, USD per 1M tokens. Local runs report the *equivalent* API cost of the same paradigm. ---
# System 1 (Decisions API): single forward pass, zero generation ($0.10 / 1M input, $0.00 output)
# System 2 (Generative LLM - GPT-6 Sol): commercial frontier tier ($2.00 / 1M input, $10.00 / 1M output)
PRICING = {
    "decisions": {
        "input": _f("PRICING_DECISIONS_INPUT", 0.10),
        "output": _f("PRICING_DECISIONS_OUTPUT", 0.00),
    },
    "chat": {
        "input": _f("PRICING_CHAT_INPUT", 2.00),
        "output": _f("PRICING_CHAT_OUTPUT", 10.00),
    },
}

# --- Scale economics ---
CLAIMS_PER_DAY = int(os.getenv("CLAIMS_PER_DAY", 1_000_000))
