"""Unit tests for the ported Cygnet readout (no GPU: llama-server is faked with httpx.MockTransport)."""
import json
import math

import httpx
import pytest

import cygnet_readout as cr
from claims_data import ROUTING_SCHEMA


def lp(**tokens):
    return [(t.replace("_SP", "\u2581"), math.log(p)) for t, p in tokens.items()]


def test_letter_mass_is_summed_over_duplicate_tokens():
    # Gemma-4 has two tokens that decode to the same letter; both must count, not just the last/max.
    top = [("A", math.log(0.30)), ("\u2581A", math.log(0.20)), ("B", math.log(0.50))]
    probs = cr.letter_probs(top, 2)
    assert probs == pytest.approx([0.5, 0.5])


def test_letter_tokens_with_punctuation_and_prefixes_are_recognised():
    top = [(" A", math.log(0.25)), ("B.", math.log(0.25)), ("(C", math.log(0.25)), ("D", math.log(0.25)),
           ("The", math.log(0.9)), ("1", math.log(0.9))]  # non-letters are ignored
    assert cr.letter_probs(top, 4) == pytest.approx([0.25] * 4)


def test_letters_beyond_option_count_are_ignored_and_missing_letters_get_zero():
    top = [("A", math.log(0.6)), ("E", math.log(0.4))]  # only 3 options: E is not an option
    assert cr.letter_probs(top, 3) == pytest.approx([1.0, 0.0, 0.0])


def test_no_letters_returns_none():
    assert cr.letter_probs([("<|channel>", 0.0), ("###", -17.0)], 5) is None


def test_temperature_matches_the_recipe_readout_example():
    # cygnet-recipe README: pickup 99.95% -> 82.7% at T=3.4 over 4 options (tail mass under 0.01%).
    raw = [0.9995, 0.0003, 0.0002, 9e-6]
    tempered = cr.temper(raw, 3.4)
    assert tempered[0] == pytest.approx(0.827, abs=0.01)
    assert tempered[1] == pytest.approx(0.0758, abs=0.01)
    assert sum(tempered) == pytest.approx(1.0)


def test_temperature_one_is_identity_and_floor_prevents_zero_division():
    assert cr.temper([0.7, 0.3], 1.0) == [0.7, 0.3]
    assert sum(cr.temper([1.0, 0.0], 3.4)) == pytest.approx(1.0)


def test_confidence_metric():
    assert cr.confidence([1.0, 0.0, 0.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert cr.confidence([0.2] * 5) == pytest.approx(0.0)
    assert cr.confidence([1.0]) == 1.0


def test_prompt_layout_has_state_first_and_lettered_options():
    prompt = cr.build_prompt("STATE", "Pick one.", [("x", "first desc"), ("y", "second desc")])
    assert prompt.startswith("STATE\n\nPick one.\n\nOptions:\nA. first desc\nB. second desc")
    assert prompt.endswith("nothing else:")


def test_noul_items_always_show_false_first():
    items = cr.items_for(ROUTING_SCHEMA["missing_clinical_docs"])
    assert [label for label, _ in items] == ["no", "yes"]


# ---------------------------- end-to-end through a faked llama-server ----------------------------
def fake_server(letter_logprobs: dict[str, float], captured: list | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if captured is not None:
            captured.append(body)
        entries = [{"token": t, "logprob": v} for t, v in letter_logprobs.items()]
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "A"}, "logprobs": {"content": [{"top_logprobs": entries}]}}],
            "usage": {"prompt_tokens": 450, "completion_tokens": 1},
        })
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_choice_question_returns_label_keyed_calibrated_distribution():
    captured: list = []
    client = fake_server({"D": -0.0, "B": -15.5, "A": -16.2, " D": -16.3, "C": -16.4, "E": -18.0}, captured)
    res = await cr.read_question(client, "claim", ROUTING_SCHEMA["routing_queue"])
    assert res["choice"] == "cob_subrogation"  # 4th option = D
    assert set(res["probabilities"]) == set(ROUTING_SCHEMA["routing_queue"]["options"])
    assert sum(res["probabilities"].values()) == pytest.approx(1.0)
    assert 0.9 < res["p_max"] < 1.0  # T=3.4 softens the near-1.0 raw mass but not below the gate
    assert res["usage"] == {"input_tokens": 450, "output_tokens": 1}
    req = captured[0]
    assert req["max_tokens"] == 1 and req["logprobs"] is True
    assert req["chat_template_kwargs"] == {"enable_thinking": False}  # root cause of the empty responses


async def test_noul_question_reports_probability_of_yes():
    # false ('no') is letter A, true ('yes') is letter B
    client = fake_server({"B": math.log(0.97), "A": math.log(0.03)})
    res = await cr.read_question(client, "claim", ROUTING_SCHEMA["missing_clinical_docs"])
    assert res["choice"] == "yes" and res["noul"] > 0.5 and res["type"] == "noul"


async def test_score_question_reports_weighted_level():
    client = fake_server({"A": math.log(0.1), "B": math.log(0.1), "C": math.log(0.8)})
    res = await cr.read_question(client, "claim", ROUTING_SCHEMA["urgency_tier"])
    assert res["choice"] == "stat_24_hr" and 1.0 < res["score"] <= 2.0


async def test_thinking_marker_instead_of_letters_raises_readout_error():
    # exactly what llama-server returned with thinking on: only '<|channel>' at the answer slot
    client = fake_server({"<|channel>": -0.0, " >": -16.5, "###": -17.2})
    with pytest.raises(cr.ReadoutError, match="option-letter distribution"):
        await cr.read_question(client, "claim", ROUTING_SCHEMA["routing_queue"])


async def test_http_failure_becomes_readout_error():
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(503, json={"error": "loading"})))
    with pytest.raises(cr.ReadoutError, match="request failed"):
        await cr.read_question(client, "claim", ROUTING_SCHEMA["routing_queue"])
