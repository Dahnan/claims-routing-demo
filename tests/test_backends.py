"""Unit tests for backends.py (pure functions and mocked endpoints, no GPU needed)."""
import json
import pytest
import backends as b
from claims_data import SEED_CLAIMS


def test_combine_cascade_unambiguous_keeps_s1():
    s1 = {
        "claim_id": "CLM-1001",
        "routing_queue": "auto_adjudicate",
        "confidence": 0.95,
        "confidence_norm": 0.93,
        "queue_probabilities": {"auto_adjudicate": 0.95},
        "missing_clinical_docs": "no",
        "missing_docs_prob": 0.02,
        "urgency_tier": "standard_30_day",
        "urgency_probabilities": {"standard_30_day": 0.98},
        "urgency_score": 0.05,
        "latency_ms": 120.0,
        "input_tokens": 400,
        "output_tokens": 3,
        "estimated_cloud_cost_usd": 0.00004,
    }
    result = b.combine_cascade(s1, None, threshold=0.90)
    assert result["mode"] == "cascade_s1"
    assert result["escalated_to_llm"] is False
    assert result["routing_queue"] == "auto_adjudicate"
    assert result["confidence"] == 0.95
    assert result["latency_ms"] == 120.0


def test_combine_cascade_ambiguous_escalates_to_s2():
    s1 = {
        "claim_id": "CLM-1008",
        "routing_queue": "coding_audit_ncci",
        "confidence": 0.55,
        "confidence_norm": 0.44,
        "queue_probabilities": {"coding_audit_ncci": 0.55, "clinical_nurse_review": 0.35},
        "missing_clinical_docs": "yes",
        "missing_docs_prob": 0.92,
        "urgency_tier": "expedited_72_hr",
        "urgency_probabilities": {"expedited_72_hr": 0.75},
        "urgency_score": 1.1,
        "latency_ms": 110.0,
        "input_tokens": 400,
        "output_tokens": 3,
        "estimated_cloud_cost_usd": 0.00004,
    }
    s2 = {
        "claim_id": "CLM-1008",
        "routing_queue": "clinical_nurse_review",
        "missing_clinical_docs": "yes",
        "urgency_tier": "expedited_72_hr",
        "clinical_rationale": "Medical necessity requires orthopedic nurse review of unattached op report.",
        "latency_ms": 1500.0,
        "input_tokens": 500,
        "output_tokens": 120,
        "estimated_cloud_cost_usd": 0.00011,
    }
    result = b.combine_cascade(s1, s2, threshold=0.90)
    assert result["mode"] == "cascade_escalated"
    assert result["escalated_to_llm"] is True
    assert result["routing_queue"] == "clinical_nurse_review"
    assert result["clinical_rationale"] == s2["clinical_rationale"]
    assert result["confidence"] == 0.55
    assert result["latency_ms"] == pytest.approx(1610.0)
    assert result["input_tokens"] == 900
    assert result["output_tokens"] == 123
    assert result["estimated_cloud_cost_usd"] == pytest.approx(0.00015)


def test_summarize_computes_accuracy_latency_and_costs():
    claims = [
        {"claim_id": "C1", "ground_truth_queue": "auto_adjudicate", "ground_truth_missing_docs": "no", "ground_truth_urgency": "standard_30_day"},
        {"claim_id": "C2", "ground_truth_queue": "cob_subrogation", "ground_truth_missing_docs": "no", "ground_truth_urgency": "standard_30_day"},
    ]
    results = [
        {"claim_id": "C1", "routing_queue": "auto_adjudicate", "missing_clinical_docs": "no", "urgency_tier": "standard_30_day", "latency_ms": 100.0, "estimated_cloud_cost_usd": 0.00005, "input_tokens": 500, "output_tokens": 3, "escalated_to_llm": False},
        {"claim_id": "C2", "routing_queue": "auto_adjudicate", "missing_clinical_docs": "no", "urgency_tier": "standard_30_day", "latency_ms": 200.0, "estimated_cloud_cost_usd": 0.00005, "input_tokens": 500, "output_tokens": 3, "escalated_to_llm": True},
    ]
    summary = b.summarize(results, claims, energy_wh=0.05, wall_s=1.2)
    assert summary["count"] == 2
    assert summary["accuracy"] == 0.5  # 1 hit out of 2
    assert summary["docs_accuracy"] == 1.0
    assert summary["urgency_accuracy"] == 1.0
    assert summary["latency_p50_ms"] == 150.0 or summary["latency_p50_ms"] in (100.0, 200.0)
    assert summary["escalation_rate"] == 0.5
    assert summary["cost_usd"] == pytest.approx(0.00010)
    assert summary["annual_cost_usd"] > 0
    assert summary["energy_wh"] == 0.05


def test_simulate_cascade_dynamic_threshold():
    claims = [
        {"claim_id": "C1", "ground_truth_queue": "auto_adjudicate", "ground_truth_missing_docs": "no", "ground_truth_urgency": "standard_30_day"},
        {"claim_id": "C2", "ground_truth_queue": "clinical_nurse_review", "ground_truth_missing_docs": "yes", "ground_truth_urgency": "expedited_72_hr"},
    ]
    s1 = [
        {"claim_id": "C1", "routing_queue": "auto_adjudicate", "confidence": 0.95, "confidence_norm": 0.93, "queue_probabilities": {}, "missing_clinical_docs": "no", "missing_docs_prob": 0.01, "urgency_tier": "standard_30_day", "urgency_probabilities": {}, "urgency_score": 0.0, "latency_ms": 100.0, "input_tokens": 300, "output_tokens": 3, "estimated_cloud_cost_usd": 0.00003},
        {"claim_id": "C2", "routing_queue": "coding_audit_ncci", "confidence": 0.60, "confidence_norm": 0.50, "queue_probabilities": {}, "missing_clinical_docs": "yes", "missing_docs_prob": 0.90, "urgency_tier": "expedited_72_hr", "urgency_probabilities": {}, "urgency_score": 1.0, "latency_ms": 100.0, "input_tokens": 300, "output_tokens": 3, "estimated_cloud_cost_usd": 0.00003},
    ]
    s2 = [
        {"claim_id": "C1", "routing_queue": "auto_adjudicate", "missing_clinical_docs": "no", "urgency_tier": "standard_30_day", "clinical_rationale": "Clean routine.", "latency_ms": 1500.0, "input_tokens": 400, "output_tokens": 100, "estimated_cloud_cost_usd": 0.00009},
        {"claim_id": "C2", "routing_queue": "clinical_nurse_review", "missing_clinical_docs": "yes", "urgency_tier": "expedited_72_hr", "clinical_rationale": "Complex stay.", "latency_ms": 1500.0, "input_tokens": 400, "output_tokens": 100, "estimated_cloud_cost_usd": 0.00009},
    ]

    # At threshold 0.90: C1 stays S1, C2 escalates to S2
    sim90 = b.simulate_cascade(s1, s2, claims, threshold=0.90)
    assert sim90["summary"]["escalation_rate"] == 0.5
    assert sim90["summary"]["accuracy"] == 1.0

    # At threshold 0.50: neither escalates
    sim50 = b.simulate_cascade(s1, s2, claims, threshold=0.50)
    assert sim50["summary"]["escalation_rate"] == 0.0
    assert sim50["summary"]["accuracy"] == 0.5  # C2 is wrong on S1


def test_parse_json_content_raises_on_empty():
    with pytest.raises(RuntimeError, match="System 2 returned empty content"):
        b._parse_json_content("", {"reasoning_content": "thoughts"})


def test_decisions_payload_formatting():
    claim = SEED_CLAIMS[0]
    payload = b._decisions_payload(claim)
    assert payload["model"] == "gpt-6-luna"
    assert "CLAIM ID: CLM-1001" in payload["input"]
    qs = payload["questions"]
    assert len(qs) == 3
    q_dict = {q["name"]: q for q in qs}
    # routing_queue choices have 'value' and 'description'
    assert "value" in q_dict["routing_queue"]["choices"][0]
    assert "description" in q_dict["routing_queue"]["choices"][0]
    # urgency_tier levels have 'label' and 'description'
    assert "label" in q_dict["urgency_tier"]["levels"][0]
    assert "description" in q_dict["urgency_tier"]["levels"][0]


def test_llm_claim_routing_output_schema_forbids_extra():
    schema = b.LLMClaimRoutingOutput.model_json_schema()
    assert schema.get("additionalProperties") is False


def test_probs_parses_score_with_labels_correctly():
    # OpenAI Decisions API score response returns objects with both 'value' (int) and 'label' (str)
    raw_answer = {
        "name": "urgency_tier",
        "probabilities": [
            {"value": 0, "label": "standard_30_day", "probability": 0.85},
            {"value": 1, "label": "expedited_72_hr", "probability": 0.15},
            {"value": 2, "label": "stat_24_hr", "probability": 0.0},
        ],
    }
    labels = ["standard_30_day", "expedited_72_hr", "stat_24_hr"]
    probs = b._probs(raw_answer, labels)
    assert probs["standard_30_day"] == pytest.approx(0.85)
    assert probs["expedited_72_hr"] == pytest.approx(0.15)
    assert probs["stat_24_hr"] == pytest.approx(0.0)

