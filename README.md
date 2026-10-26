# System 1 vs. System 2 Health Insurance Claims Router

Live interactive architectural bake-off and production reference implementation comparing **System 1 decision models** (single-token calibrated readout) against **System 2 autoregressive LLMs** (structured JSON with clinical rationale) for routing US health insurance claims (EDI 837 metadata + unstructured clinical attachments).

---

## 1. Executive Summary & Core Thesis

* **The Problem:** In high-volume healthcare claims processing (EDI 837 ingestion), health plans traditionally face a trade-off between brittle regex rule heuristics and costly, high-latency generative LLMs that waste hundreds of output tokens explaining routine clean claims.
* **The System 1 Solution:** Single forward-pass decision models (Cygnet readout) evaluate bounded decision questions in a single token pass, returning calibrated multi-class probability distributions across declared routing queues at sub-second latency with zero output token cost.
* **The Production Pattern:** A **Confidence-Gated Cascade** where the System 1 model autonomously auto-adjudicates clear-cut claims (confidence $\ge 0.90$, representing ~80% of volume) and escalates ambiguous clinical edge cases (< 0.90) to the System 2 LLM to generate structured reasoning traces for human clinical nurse reviewers.
* **Model Parity Guarantee:** Both System 1 and System 2 execute on the **exact same model weights** (`gemma-4-12B-it-Q8_0.gguf`) running locally on an NVIDIA RTX 4090. All latency and cost differentials stem strictly from the inference paradigm rather than parameter size.

---

## 2. Architectural Comparison

| Dimension | System 1: Cygnet Readout | System 2: Generative LLM | Confidence Cascade |
| :--- | :--- | :--- | :--- |
| **Model Weights** | `google/gemma-4-12B-it` (Q8_0) | `google/gemma-4-12B-it` (Q8_0) | Shared `gemma-4-12B-it` (Q8_0) |
| **Inference Mode** | 1-token logprob extraction ($T=3.4$) | Autoregressive JSON schema generation | Gated: System 1 $\rightarrow$ System 2 on demand |
| **Output Tokens** | **3 tokens** (1 token / question) | **110 – 140 tokens** | 3 tokens (routine) / ~120 tokens (escalated) |
| **Latency (P50)** | **~500 – 600 ms** (cold) / **~80 ms** (hot) | **~2,800 – 3,200 ms** | **~600 ms** (blended) |
| **Reasoning Artifact** | Calibrated probability distribution | 2-sentence clinical & coding rationale | Probabilities on clean, rationale on escalated |
| **Annual Scale Cost** | ~$146,000 / year (100k claims/day) | ~$2,460,000 / year (100k claims/day) | **~$615,000 / year (75% savings)** |

---

## 3. Decision Rubric & Schema

Every claim payload is evaluated across three simultaneous questions:

1. **`routing_queue` (`choice` - 5 options):**
   * `auto_adjudicate`: Clean, routine claim matching clinical policy.
   * `clinical_nurse_review`: Complex medical necessity, inpatient stay validation, or off-label biologic/oncology regimen.
   * `coding_audit_ncci`: Suspected unbundling, Modifier 25/59 abuse without separate E&M service, or CPT/ICD-10 mismatch.
   * `cob_subrogation`: Trauma, workplace injury, motor vehicle collision, or commercial slip-and-fall indicating third-party liability.
   * `siu_fraud_review`: Phantom billing, impossible daily time-based CPT volume, or severe provider billing anomaly.
2. **`missing_clinical_docs` (`noul` / binary probability):**
   * Calibrated probability of whether the clinical narrative explicitly references an operative report, pathology lab result, or imaging study that is missing from the attachment (`yes` / `no`).
3. **`urgency_tier` (`score` / 3 levels):**
   * `standard_30_day`, `expedited_72_hr`, `stat_24_hr` (reported with discrete argmax and continuous expectation score).

---

## 4. Quickstart & Execution

### 4.1 Prerequisites
* Windows 11 + NVIDIA RTX 4090 (24GB VRAM)
* Conda environment `System1Env` (Python 3.13)
* Weights downloaded at `models/gemma-4-12B-it-Q8_0.gguf`

### 4.2 Start Everything (One-Click)
Run the master PowerShell launch script:
```powershell
.\start_demo_all.ps1
```
This automatically:
1. Validates local weights and Python environment
2. Launches `llama-server.exe` on `http://127.0.0.1:8000` with thinking disabled (`--reasoning off`)
3. Starts the FastAPI server and serves the dashboard on `http://127.0.0.1:8080`
4. Provides connection details for Microsoft Dev Tunnels for corporate VPN presentations

### 4.3 Automated Verification & Smoke Tests
Run unit tests:
```powershell
python -m pytest -v
```

Run end-to-end CLI smoke test on the RTX 4090:
```powershell
python backends.py
```

---

## 5. Repository File Map

```text
d:\Documents\Python\System_1_Demo\
├── config.py                 # Central settings (URLs, threshold, pricing, calibration temp)
├── cygnet_readout.py         # Ported Cygnet single-token readout & calibration math (MIT)
├── backends.py               # Unified 4-backend router (Cygnet, Gemma 4, Cascade, Cloud, NVML)
├── claims_data.py            # Routing rubric, 10 seed claims, 30 variants, 80/20 batch generator
├── server.py                 # FastAPI web application (127.0.0.1:8080)
├── index.html                # Single-page interactive dashboard (Vanilla CSS + Chart.js)
├── start_model_server.ps1    # Launches llama-server with optimal RTX 4090 flags
├── start_demo_all.ps1        # Master launch script
├── tests/
│   ├── test_readout.py       # Ported Cygnet readout math & token aggregation tests
│   ├── test_backends.py      # Cascade logic, token pricing, simulation tests
│   ├── test_claims_data.py   # Schema integrity & 80/20 deterministic batch tests
│   └── test_server.py        # FastAPI endpoint integration tests
└── models/                   # Ignored by git
    └── gemma-4-12B-it-Q8_0.gguf
```

---

## 6. Credits & Provenance

* **Cygnet Readout Approach:** Ported from [`blockbrain-ai/cygnet-recipe`](https://github.com/blockbrain-ai/cygnet-recipe) (MIT License). Readout paradigm originated from [NInfer](https://github.com/igorls/ninfer).
* **Backbone Weights:** Google DeepMind `gemma-4-12B-it` (Apache-2.0 / Gemma Prohibited Use Policy).
