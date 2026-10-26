"""Unit tests for claims_data.py (dataset validation, batch generation, schema compliance)."""
from claims_data import (
    ROUTING_SCHEMA,
    SEED_CLAIMS,
    VARIANT_CLAIMS,
    ALL_CLAIMS,
    get_batch_claims,
    format_claim_state,
)


def test_seed_and_variant_claims_schema_integrity():
    valid_queues = set(ROUTING_SCHEMA["routing_queue"]["options"])
    valid_missing_docs = {"yes", "no"}
    valid_urgency = set(ROUTING_SCHEMA["urgency_tier"]["options"])

    assert len(SEED_CLAIMS) == 10
    assert len(VARIANT_CLAIMS) == 30
    assert len(ALL_CLAIMS) == 40

    for c in ALL_CLAIMS:
        assert c["ground_truth_queue"] in valid_queues, f"{c['claim_id']} invalid queue"
        assert c["ground_truth_missing_docs"] in valid_missing_docs, f"{c['claim_id']} invalid missing docs"
        assert c["ground_truth_urgency"] in valid_urgency, f"{c['claim_id']} invalid urgency"
        assert c["difficulty"] in ("easy", "hard_ambiguous")
        assert c["billed_amount"] > 0
        assert len(c["icd10_codes"]) >= 1
        assert len(c["cpt_codes"]) >= 1
        assert len(c["clinical_note"]) > 20


def test_batch_generator_exact_80_20_split():
    batch100 = get_batch_claims(100)
    assert len(batch100) == 100
    easy_count = sum(1 for c in batch100 if c["difficulty"] == "easy")
    hard_count = sum(1 for c in batch100 if c["difficulty"] == "hard_ambiguous")
    assert easy_count == 80
    assert hard_count == 20

    batch25 = get_batch_claims(25)
    assert len(batch25) == 25
    assert sum(1 for c in batch25 if c["difficulty"] == "hard_ambiguous") == round(25 * 0.20)


def test_batch_generator_unique_ids_and_determinism():
    b1 = get_batch_claims(100, seed=42)
    b2 = get_batch_claims(100, seed=42)

    ids = [c["claim_id"] for c in b1]
    assert len(ids) == len(set(ids)), "Claim IDs must be unique"
    assert ids[0] == "CLM-2001"
    assert ids[-1] == "CLM-2100"

    # Determinism
    for c1, c2 in zip(b1, b2):
        assert c1["claim_id"] == c2["claim_id"]
        assert c1["billed_amount"] == c2["billed_amount"]
        assert c1["ground_truth_queue"] == c2["ground_truth_queue"]


def test_format_claim_state():
    state = format_claim_state(SEED_CLAIMS[0])
    assert "CLAIM ID: CLM-1001" in state
    assert "PATIENT AGE: 58" in state
    assert "ICD-10 DIAGNOSIS CODES:" in state
    assert "CPT/HCPCS PROCEDURE CODES:" in state
    assert "CLINICAL ATTACHMENT / PROVIDER NOTE:" in state
