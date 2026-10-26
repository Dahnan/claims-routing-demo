"""Cygnet one-token option readout, ported to async httpx over llama-server.

Ported from blockbrain-ai/cygnet-recipe (``shim/cygnet_shim.py`` and ``shim/decision_server.py``, MIT). The
one-token readout idea is NInfer's (github.com/igorls/ninfer); see cygnet-recipe/CREDITS.md. The recipe targets
vLLM; this port targets llama.cpp's ``llama-server``, which differs in two ways:

  * no ``structured_outputs.choice`` masking. Measured on llama-server: the answer position is not masked, but
    with thinking off every option letter appears in the top-20 logprobs, so the mass is read the same way and
    missing letters get 0 (a GBNF grammar was tried and changed nothing, so none is sent).
  * Gemma-4 *thinks by default* on llama.cpp, which turns the answer token into ``<|channel>``. Every request
    therefore sends ``chat_template_kwargs.enable_thinking = false`` (the recipe's own advice for llama.cpp).

What it does, per question (one forward pass, one output token, cost is input tokens only):
  1. presents the state, instructions and options as lettered choices (A., B., ...);
  2. reads the top-20 logprobs at the answer slot and SUMS the probability mass per letter (Gemma-4 has more
     than one token per letter, so a dict keyed by text would drop mass);
  3. renormalises over the options and applies ONE calibration temperature, ``p^(1/T)`` renormalised.

Typed questions: ``choice`` (labelled options), ``noul`` (yes/no, options shown "false" first, the order the
recipe was measured in), ``score`` (ordered levels, also reported as a probability-weighted level).
"""
from __future__ import annotations

import math
import re
from typing import Any, Sequence

import httpx

import config

# Verbatim from cygnet_shim.SYSTEM: "the scaffolding is the fix" (a bare prompt lost ~10 points).
SYSTEM = (
    "You are a calibration engine. You never answer in prose. You are given a state, a question and "
    "a numbered set of options, and you choose exactly one option. You reply with that option's "
    "LETTER and nothing else \u2014 a single character, no words, no punctuation, no explanation."
)

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# A letter can surface as "A", " A", "A." or SentencePiece "\u2581A"; all of them count toward that letter.
_TOK_LETTER = re.compile(r"^[\s(\[{'\"\u2581]*([A-Za-z])[\s.,:)\]}'\"]*$")

# (label, description) in presentation order
Item = tuple[str, str]


class ReadoutError(RuntimeError):
    """The model server failed, or no option-letter distribution could be recovered at the answer slot."""


def build_prompt(state: str, instructions: str, items: Sequence[Item]) -> str:
    """cygnet_shim.build_prompt layout. The claim state comes first so the three questions share a KV prefix."""
    lines = [state.rstrip(), "", instructions.rstrip(), "", "Options:"]
    for i, (_label, desc) in enumerate(items):
        lines.append(f"{LETTERS[i]}. {desc}")
    lines += ["", "Answer with the letter of exactly one option, and nothing else:"]
    return "\n".join(lines)


def letter_probs(top_logprobs: Sequence[tuple[str, float]], n_letters: int) -> list[float] | None:
    """Sum probability mass per option letter from the answer slot's top logprobs, renormalised over the options.

    Returns None only when no option letter appeared at all. A letter missing from a top-20 that does contain
    others carries ~0 mass and gets 0.0.
    """
    acc: dict[int, float] = {}
    for tok, lp in top_logprobs:
        m = _TOK_LETTER.match(tok)
        if not m:
            continue
        idx = LETTERS.index(m.group(1).upper())
        if idx >= n_letters:
            continue
        acc[idx] = acc.get(idx, 0.0) + math.exp(lp)
    total = sum(acc.values())
    if not acc or total <= 0:
        return None
    return [acc.get(i, 0.0) / total for i in range(n_letters)]


def temper(probs: Sequence[float], temperature: float | None = None) -> list[float]:
    """cygnet_shim calibration: p^(1/T) renormalised, with the recipe's 1e-12 floor. T = 1 is a no-op."""
    t = config.CYGNET_TEMPERATURE if temperature is None else temperature
    if t == 1.0:
        return list(probs)
    z = [max(p, 1e-12) ** (1.0 / t) for p in probs]
    s = sum(z)
    return [v / s for v in z]


def confidence(probs: Sequence[float]) -> float:
    """decision_server.confidence: (K * p_max - 1) / (K - 1), clipped to [0, 1] (1 for a single option)."""
    k = len(probs)
    return 1.0 if k == 1 else max(0.0, min(1.0, (k * max(probs) - 1.0) / (k - 1.0)))


def _top_logprobs(data: dict[str, Any]) -> list[tuple[str, float]]:
    try:
        entries = data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        # a list of pairs, not a dict keyed by text: duplicate letter tokens must all be kept
        return [(e["token"], float(e["logprob"])) for e in entries]
    except (KeyError, IndexError, TypeError) as exc:
        raise ReadoutError("llama-server returned no top_logprobs at the answer slot") from exc


async def read_pass(
    client: httpx.AsyncClient,
    state: str,
    instructions: str,
    items: Sequence[Item],
    *,
    url: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
) -> tuple[list[float], dict[str, int]]:
    """One forward pass over at most 20 options. Returns calibrated probabilities (in item order) and usage."""
    if len(items) > config.TOP_LOGPROBS:
        raise ReadoutError(f"{len(items)} options exceeds the {config.TOP_LOGPROBS} logprobs one pass returns")
    if len(items) == 1:
        return [1.0], {"input_tokens": 0, "output_tokens": 0}
    body = {
        "model": model or config.LOCAL_MODEL_ALIAS,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_prompt(state, instructions, items)},
        ],
        "max_tokens": 1,
        "temperature": 1.0,
        "logprobs": True,
        "top_logprobs": config.TOP_LOGPROBS,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        resp = await client.post(url or config.LLAMA_CHAT_URL, json=body)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as exc:
        raise ReadoutError(f"llama-server request failed: {type(exc).__name__}: {exc}") from exc
    raw = letter_probs(_top_logprobs(data), len(items))
    if raw is None:
        raise ReadoutError("could not recover an option-letter distribution from the answer slot")
    usage = data.get("usage") or {}
    return temper(raw, temperature), {
        "input_tokens": int(usage.get("prompt_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or 0),
    }


def items_for(spec: dict[str, Any]) -> list[Item]:
    """Presentation order for a ROUTING_SCHEMA question. noul is always shown false ('no') first."""
    options: dict[str, str] = spec["options"]
    if spec["type"] == "noul":
        return [("no", options["no"]), ("yes", options["yes"])]
    return list(options.items())


async def read_question(
    client: httpx.AsyncClient, state: str, spec: dict[str, Any], **kwargs: Any
) -> dict[str, Any]:
    """Answer one typed question. Always returns label-keyed ``probabilities``, ``choice``, ``p_max``, ``confidence``."""
    qtype = spec["type"]
    if qtype not in ("choice", "noul", "score"):
        raise ReadoutError(f"unsupported question type {qtype!r}")
    items = items_for(spec)
    probs, usage = await read_pass(client, state, spec["instructions"], items, **kwargs)
    s = sum(probs)
    probs = [p / s for p in probs]  # cygnet_shim renormalises once more; so do we
    labels = [label for label, _ in items]
    dist = {label: p for label, p in zip(labels, probs)}
    best = max(dist, key=dist.__getitem__)
    out: dict[str, Any] = {
        "type": qtype,
        "choice": best,
        "probabilities": dist,
        "p_max": dist[best],
        "confidence": confidence(probs),
        "usage": usage,
    }
    if qtype == "noul":
        out["noul"] = dist["yes"]  # P(true)
    elif qtype == "score":
        out["score"] = sum(i * p for i, p in enumerate(probs))  # probability-weighted level index
    return out
