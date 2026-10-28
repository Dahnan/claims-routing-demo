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


def test_evaluate_single_cloud_with_mock(client, monkeypatch):
    async def mock_cloud(claim, use_decisions_api=True):
        if use_decisions_api:
            return {
                "engine": "Cloud System 1: OpenAI Decisions API",
                "mode": "system1_cloud",
                "claim_id": claim["claim_id"],
                "routing_queue": "auto_adjudicate",
                "confidence": 0.88,
                "confidence_norm": 0.85,
                "queue_probabilities": {"auto_adjudicate": 0.88},
                "missing_clinical_docs": "no",
                "missing_docs_prob": 0.0,
                "urgency_tier": "standard_30_day",
                "urgency_probabilities": {"standard_30_day": 1.0},
                "urgency_score": 0.0,
                "clinical_rationale": None,
                "latency_ms": 150.0,
                "input_tokens": 789,
                "output_tokens": 0,
                "estimated_cloud_cost_usd": 0.000079,
            }
        return {
            "engine": "Cloud System 2: OpenAI GPT-6 Sol",
            "mode": "system2_cloud",
            "claim_id": claim["claim_id"],
            "routing_queue": "auto_adjudicate",
            "missing_clinical_docs": "no",
            "urgency_tier": "standard_30_day",
            "clinical_rationale": "Cloud rationale verified.",
            "confidence": None,
            "confidence_norm": None,
            "queue_probabilities": None,
            "missing_docs_prob": None,
            "urgency_probabilities": None,
            "urgency_score": None,
            "latency_ms": 1200.0,
            "input_tokens": 750,
            "output_tokens": 80,
            "estimated_cloud_cost_usd": 0.002675,
        }

    monkeypatch.setattr(b, "route_openai_cloud", mock_cloud)

    payload = {
        "claim_id": "CLM-1001",
        "backend": "cloud",
        "modes": ["s1", "s2", "cascade"],
        "threshold": 0.90,
    }
    resp = client.post("/api/evaluate-single", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["backend"] == "cloud"
    assert "Cloud System 1" in data["s1"]["engine"]
    assert "Cloud System 2" in data["s2"]["engine"]


def test_index_html_theme_and_elements(client):
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.text

    # Theme tokens
    assert "--brand-navy" in html
    assert "--brand-blue" in html
    assert "--s2-orange" in html
    assert "--cascade-teal" in html

    # Ensure no UHC or Optum brand names in UI
    assert "uhc" not in html.lower()
    assert "optum" not in html.lower()

    # Scale estimate (1M claims/day)
    assert "1M claims/day" in html
    assert "1,000,000 claims/day" in html
    assert "100k claims/day" not in html
    assert "100,000 claims/day" not in html

    # Essential IDs in S2 -> S1 -> Cascade order
    for elem_id in [
        "s2-winner-queue", "s1-winner-queue",
        "s2-progress-bar", "s1-progress-bar", "cascade-progress-bar",
        "t-s2-acc", "t-s1-acc", "t-casc-acc",
        "t-s2-cost", "t-s1-cost", "t-casc-cost",
        "annual-s2-cost", "annual-s1-cost", "annual-casc-cost", "annual-savings-sub"
    ]:
        assert f'id="{elem_id}"' in html, f"Missing id: {elem_id}"

    # Annual costs must be blank ('--') initially until batch race is run
    assert 'id="annual-s2-cost">--</div>' in html
    assert 'id="annual-s1-cost">--</div>' in html
    assert 'id="annual-casc-cost">--</div>' in html
    assert "$24,600,000" not in html


def test_get_slides(client):
    resp = client.get("/slides")
    assert resp.status_code == 200
    html = resp.text
    assert "System 1 vs. System 2 Claims Routing" in html
    assert "Claim Payload Structure & Inference Mechanics" in html
    assert "slide-1" in html
    assert "slide-2" in html

