"""Unit tests for server.py endpoints using FastAPI TestClient."""
import pytest
from fastapi.testclient import TestClient
from server import app
import backends as b


@pytest.fixture
def client():
    return TestClient(app)


def test_get_claims(client):
    resp = client.get("/api/claims")
    assert resp.status_code == 200
    data = resp.json()
    assert "schema" in data
    assert "seed_claims" in data
    assert len(data["seed_claims"]) == 10
    assert "default_threshold" in data
    assert data["default_threshold"] == 0.90


def test_get_telemetry(client):
    resp = client.get("/api/telemetry")
    assert resp.status_code == 200
    data = resp.json()
    assert "gpu" in data
    assert "llama_server" in data
    assert "cloud" in data


def test_evaluate_single_with_mock(client, monkeypatch):
    async def mock_s1(claim, backend="local"):
        return {
            "engine": "Mock S1",
            "mode": "system1_local",
            "claim_id": claim["claim_id"],
            "routing_queue": "auto_adjudicate",
            "confidence": 0.96,
            "confidence_norm": 0.95,
            "queue_probabilities": {"auto_adjudicate": 0.96},
            "missing_clinical_docs": "no",
            "missing_docs_prob": 0.01,
            "urgency_tier": "standard_30_day",
            "urgency_probabilities": {"standard_30_day": 0.99},
            "urgency_score": 0.01,
            "clinical_rationale": None,
            "latency_ms": 95.0,
            "input_tokens": 400,
            "output_tokens": 3,
            "estimated_cloud_cost_usd": 0.00004,
        }

    async def mock_s2(claim, backend="local"):
        return {
            "engine": "Mock S2",
            "mode": "system2_local",
            "claim_id": claim["claim_id"],
            "routing_queue": "auto_adjudicate",
            "missing_clinical_docs": "no",
            "urgency_tier": "standard_30_day",
            "clinical_rationale": "Routine visit clean claim.",
            "confidence": None,
            "confidence_norm": None,
            "queue_probabilities": None,
            "missing_docs_prob": None,
            "urgency_probabilities": None,
            "urgency_score": None,
            "latency_ms": 1800.0,
            "input_tokens": 500,
            "output_tokens": 110,
            "estimated_cloud_cost_usd": 0.00010,
        }

    monkeypatch.setattr(b, "route_system1", mock_s1)
    monkeypatch.setattr(b, "route_system2", mock_s2)

    payload = {
        "claim_id": "CLM-1001",
        "backend": "local",
        "modes": ["s1", "s2", "cascade"],
        "threshold": 0.90,
    }
    resp = client.post("/api/evaluate-single", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "s1" in data
    assert "s2" in data
    assert "cascade" in data
    assert data["s1"]["routing_queue"] == "auto_adjudicate"
    assert data["cascade"]["escalated_to_llm"] is False


def test_evaluate_single_invalid_claim_400(client):
    resp = client.post("/api/evaluate-single", json={"claim_id": "NON_EXISTENT"})
    assert resp.status_code == 400


def test_simulate_cascade_endpoint(client):
    claims = [{"claim_id": "C1", "ground_truth_queue": "auto_adjudicate", "ground_truth_missing_docs": "no", "ground_truth_urgency": "standard_30_day"}]
    s1 = [{
        "claim_id": "C1", "routing_queue": "auto_adjudicate", "confidence": 0.85,
        "confidence_norm": 0.80, "queue_probabilities": {}, "missing_clinical_docs": "no",
        "missing_docs_prob": 0.05, "urgency_tier": "standard_30_day", "urgency_probabilities": {},
        "urgency_score": 0.0, "latency_ms": 80.0, "input_tokens": 300, "output_tokens": 3,
        "estimated_cloud_cost_usd": 0.00003,
    }]
    s2 = [{
        "claim_id": "C1", "routing_queue": "auto_adjudicate", "missing_clinical_docs": "no",
        "urgency_tier": "standard_30_day", "clinical_rationale": "Adjudicated clean.",
        "latency_ms": 1500.0, "input_tokens": 400, "output_tokens": 100, "estimated_cloud_cost_usd": 0.00009,
    }]

    # At threshold 0.90: confidence 0.85 < 0.90 -> escalates
    r1 = client.post("/api/simulate-cascade", json={"threshold": 0.90, "s1": s1, "s2": s2, "claims": claims})
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["summary"]["escalation_rate"] == 1.0

    # At threshold 0.80: confidence 0.85 >= 0.80 -> stays S1
    r2 = client.post("/api/simulate-cascade", json={"threshold": 0.80, "s1": s1, "s2": s2, "claims": claims})
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["summary"]["escalation_rate"] == 0.0
