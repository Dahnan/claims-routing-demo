import random
from typing import List, Dict, Any

ROUTING_SCHEMA = {
    "routing_queue": {
        "type": "choice",
        "instructions": (
            "Select the single primary US health insurance adjudication queue "
            "for this claim based on the EDI 837 codes and unstructured clinical notes."
        ),
        "options": {
            "auto_adjudicate": "Clean, routine claim; ICD-10 diagnosis and CPT procedure match standard clinical policy with no red flags.",
            "clinical_nurse_review": "Complex medical necessity, inpatient length-of-stay validation, off-label biologic/oncology drug, or prior-auth discrepancy.",
            "coding_audit_ncci": "Suspected unbundling, Modifier 25 or 59 misuse without separate E&M service, upcoding, or CPT/ICD-10 mismatch.",
            "cob_subrogation": "Trauma, workplace injury, motor vehicle collision, or commercial slip-and-fall indicating potential third-party liability.",
            "siu_fraud_review": "Phantom billing, impossible daily service hours, non-rendered services, or severe provider billing anomaly."
        }
    },
    "missing_clinical_docs": {
        "type": "binary",
        "instructions": (
            "Does the clinical narrative explicitly reference an operative report, pathology lab result, "
            "or imaging study that is missing from the claim attachment?"
        ),
        "options": {
            "yes": "Yes, a required clinical document, lab result, or operative report is referenced as missing or not attached.",
            "no": "No, the documentation appears self-contained or no missing external report is flagged."
        }
    },
    "urgency_tier": {
        "type": "choice",
        "instructions": "Determine the clinical and statutory processing turnaround tier for this claim.",
        "options": {
            "standard_30_day": "Routine post-service outpatient or preventive claim with standard 30-day statutory turnaround.",
            "expedited_72_hr": "High-dollar inpatient, concurrent review, or subacute care requiring 72-hour turnaround.",
            "stat_24_hr": "Acute emergent care, active oncology regimen hold, or urgent appeal requiring 24-hour processing."
        }
    }
}

SEED_CLAIMS: List[Dict[str, Any]] = [
    {
        "claim_id": "CLM-1001",
        "title": "Routine Type 2 Diabetes Follow-Up (Clean)",
        "patient_age": 58,
        "provider_specialty": "Internal Medicine",
        "billed_amount": 185.00,
        "icd10_codes": ["E11.9 - Type 2 diabetes mellitus without complications", "I10 - Essential hypertension"],
        "cpt_codes": ["99213 - Office/outpatient visit est, low MDM", "83036 - Hemoglobin A1c level"],
        "clinical_note": (
            "Established patient presents for 6-month diabetes and blood pressure check. "
            "Home glucose logs stable. Point-of-care HbA1c performed in clinic today is 6.8%. "
            "BP 126/78. Continue Metformin 1000mg BID and Lisinopril 10mg daily. Return in 6 months."
        ),
        "ground_truth_queue": "auto_adjudicate",
        "ground_truth_missing_docs": "no",
        "ground_truth_urgency": "standard_30_day",
        "difficulty": "easy"
    },
    {
        "claim_id": "CLM-1002",
        "title": "Tibial Fracture at Commercial Grocery Store (Subrogation)",
        "patient_age": 42,
        "provider_specialty": "Orthopedic Surgery / Emergency Medicine",
        "billed_amount": 4820.00,
        "icd10_codes": ["S82.101A - Unspecified fracture of upper end of right tibia, initial encounter"],
        "cpt_codes": ["99284 - Emergency dept visit, high MDM", "27752 - Closed treatment of tibial shaft fracture, with manipulation"],
        "clinical_note": (
            "Pt brought to ED after slipping on an unmarked wet floor near the produce aisle at "
            "MetroMart Supermarket at approximately 14:30. Store manager filed an incident report on site. "
            "Right knee/proximal tibia radiographs confirm minimally displaced fracture. Closed reduction and long-leg splint applied."
        ),
        "ground_truth_queue": "cob_subrogation",
        "ground_truth_missing_docs": "no",
        "ground_truth_urgency": "standard_30_day",
        "difficulty": "easy"
    },
    {
        "claim_id": "CLM-1003",
        "title": "Modifier 25 Abuse on Routine Lesion Destruction (NCCI Audit)",
        "patient_age": 64,
        "provider_specialty": "Dermatology",
        "billed_amount": 610.00,
        "icd10_codes": ["L57.0 - Actinic keratosis"],
        "cpt_codes": ["99214-25 - Office visit est, moderate MDM (Modifier 25)", "17000 - Destruction premalignant lesion, first"],
        "clinical_note": (
            "Patient arrived for scheduled liquid nitrogen cryotherapy of single actinic keratosis on left dorsal forearm "
            "identified at previous visit. Lesion treated with 10-second freeze cycle. No new complaints discussed, "
            "no separate physical exam or medication management performed today."
        ),
        "ground_truth_queue": "coding_audit_ncci",
        "ground_truth_missing_docs": "no",
        "ground_truth_urgency": "standard_30_day",
        "difficulty": "easy"
    },
    {
        "claim_id": "CLM-1004",
        "title": "Off-Label Biologic Infusion for Refractory Lupus (Nurse Review)",
        "patient_age": 34,
        "provider_specialty": "Rheumatology",
        "billed_amount": 14500.00,
        "icd10_codes": ["M32.14 - Glomerular disease in systemic lupus erythematosus"],
        "cpt_codes": ["96413 - Chemo/biologic IV infusion, 1 hr", "J9312 - Injection, rituximab, 10 mg (100 units)"],
        "clinical_note": (
            "34yo female with severe Class IV lupus nephritis refractory to mycophenolate and cyclophosphamide. "
            "Initiating high-dose IV Rituximab rescue protocol to prevent progression to end-stage renal disease. "
            "Renal biopsy pathology report from last Tuesday confirms active crescents (note: pathology attachment failed to upload in EDI portal)."
        ),
        "ground_truth_queue": "clinical_nurse_review",
        "ground_truth_missing_docs": "yes",
        "ground_truth_urgency": "expedited_72_hr",
        "difficulty": "easy"
    },
    {
        "claim_id": "CLM-1005",
        "title": "Impossible Time-Based Psychotherapy Volume (SIU Fraud)",
        "patient_age": 29,
        "provider_specialty": "Behavioral Health",
        "billed_amount": 2400.00,
        "icd10_codes": ["F41.1 - Generalized anxiety disorder"],
        "cpt_codes": ["90837 x 8 units - Psychotherapy, 60 minutes with patient", "99215 - Office visit est, high MDM"],
        "clinical_note": (
            "Cloned template note: Patient seen for individual supportive therapy. Mood anxious. "
            "Provider billing batch indicates 34 separate 60-minute individual face-to-face 90837 sessions "
            "rendered by solo NPI on the same calendar date of service, plus duplicate family therapy codes."
        ),
        "ground_truth_queue": "siu_fraud_review",
        "ground_truth_missing_docs": "no",
        "ground_truth_urgency": "standard_30_day",
        "difficulty": "easy"
    },
    {
        "claim_id": "CLM-1006",
        "title": "Annual Preventive Mammogram & Lipid Screening (Clean)",
        "patient_age": 52,
        "provider_specialty": "Obstetrics & Gynecology",
        "billed_amount": 340.00,
        "icd10_codes": ["Z12.31 - Encounter for screening mammogram for malignant neoplasm of breast"],
        "cpt_codes": ["77067 - Screening mammography, bilateral, 2-view", "80061 - Lipid panel"],
        "clinical_note": (
            "Asymptomatic 52yo female here for routine annual preventive screening bilateral digital mammogram "
            "and fasting lipid panel. BI-RADS Category 1 (Negative). Routine screening recommended in 1 year."
        ),
        "ground_truth_queue": "auto_adjudicate",
        "ground_truth_missing_docs": "no",
        "ground_truth_urgency": "standard_30_day",
        "difficulty": "easy"
    },
    {
        "claim_id": "CLM-1007",
        "title": "Rear-End Highway Collision Cervical Whiplash (Subrogation)",
        "patient_age": 39,
        "provider_specialty": "Urgent Care / Radiology",
        "billed_amount": 1950.00,
        "icd10_codes": ["S13.4XXA - Sprain of ligaments of cervical spine, initial", "V43.52XA - Car driver injured in collision with SUV"],
        "cpt_codes": ["72141 - MRI cervical spine without contrast", "99204 - Office visit new, moderate MDM"],
        "clinical_note": (
            "Patient was the restrained driver of a sedan rear-ended at a stoplight on I-95 yesterday evening "
            "by a commercial delivery van. Auto insurance claim #AU-88412 pending. Presenting with acute C5-C6 paraspinal spasm."
        ),
        "ground_truth_queue": "cob_subrogation",
        "ground_truth_missing_docs": "no",
        "ground_truth_urgency": "standard_30_day",
        "difficulty": "easy"
    },
    {
        "claim_id": "CLM-1008",
        "title": "Ambiguous Shoulder Pain: Old Sports Injury vs. Recent Yardwork (Cascade Edge Case)",
        "patient_age": 47,
        "provider_specialty": "Orthopedics",
        "billed_amount": 3250.00,
        "icd10_codes": ["M75.121 - Incomplete rotator cuff tear or rupture of right shoulder, not specified as traumatic"],
        "cpt_codes": ["29827 - Arthroscopy, shoulder, surgical; with rotator cuff repair", "29826 - Decompression subacromial space"],
        "clinical_note": (
            "Pt underwent arthroscopic right rotator cuff repair. History notes chronic right shoulder impingement "
            "dating back to college baseball, though patient also mentioned his shoulder 'popped' last month while helping "
            "a neighbor move heavy patio pavers. Operative report dictations are still pending transcription from the ASC."
        ),
        "ground_truth_queue": "clinical_nurse_review",
        "ground_truth_missing_docs": "yes",
        "ground_truth_urgency": "expedited_72_hr",
        "difficulty": "hard_ambiguous"
    },
    {
        "claim_id": "CLM-1009",
        "title": "Unbundled Comprehensive Metabolic Panel & Individual Electrolytes (NCCI Audit)",
        "patient_age": 71,
        "provider_specialty": "Clinical Pathology Lab",
        "billed_amount": 490.00,
        "icd10_codes": ["N18.30 - Chronic kidney disease, stage 3 unspecified"],
        "cpt_codes": ["80053 - Comprehensive metabolic panel", "84132 - Potassium, serum", "84295 - Sodium, serum", "82565 - Creatinine"],
        "clinical_note": (
            "Outpatient lab draw for CKD Stage 3 monitoring from single venipuncture at 08:15 AM. "
            "Lab billed full 80053 Comprehensive Metabolic Panel alongside separate line items for serum potassium, "
            "sodium, and creatinine from the exact same specimen with Modifier 59 appended."
        ),
        "ground_truth_queue": "coding_audit_ncci",
        "ground_truth_missing_docs": "no",
        "ground_truth_urgency": "standard_30_day",
        "difficulty": "easy"
    },
    {
        "claim_id": "CLM-1010",
        "title": "Ambiguous Inpatient Chest Pain Observation + High-Level Coding (Cascade Edge Case)",
        "patient_age": 61,
        "provider_specialty": "Cardiology / Hospitalist",
        "billed_amount": 18900.00,
        "icd10_codes": ["R07.9 - Chest pain, unspecified", "K21.9 - Gastro-esophageal reflux disease without esophagitis"],
        "cpt_codes": ["99223 - Initial hospital inpatient care, high MDM", "93458-26 - Left heart catheterization"],
        "clinical_note": (
            "Admitted as full inpatient for atypical chest heaviness after spicy meal. Serial troponins negative x3, "
            "EKG normal sinus rhythm. Cardiology proceeded with diagnostic left heart cath (normal coronaries; cath cine-loop summary sheet not attached). "
            "Billed as full acute inpatient admission 99223 rather than observation status alongside separate E&M."
        ),
        "ground_truth_queue": "clinical_nurse_review",
        "ground_truth_missing_docs": "yes",
        "ground_truth_urgency": "expedited_72_hr",
        "difficulty": "hard_ambiguous"
    }
]


def format_claim_state(claim: Dict[str, Any]) -> str:
    """Formats a claim dictionary into a clean text state for System 1 and System 2 evaluation."""
    icd_str = "; ".join(claim["icd10_codes"])
    cpt_str = "; ".join(claim["cpt_codes"])
    return (
        f"CLAIM ID: {claim['claim_id']}\n"
        f"PATIENT AGE: {claim['patient_age']} | SPECIALTY: {claim['provider_specialty']} | BILLED: ${claim['billed_amount']:,.2f}\n"
        f"ICD-10 DIAGNOSIS CODES: {icd_str}\n"
        f"CPT/HCPCS PROCEDURE CODES: {cpt_str}\n"
        f"CLINICAL ATTACHMENT / PROVIDER NOTE:\n{claim['clinical_note']}"
    )


def get_batch_claims(count: int = 100) -> List[Dict[str, Any]]:
    """Generates a deterministic N-claim batch from the seed claims with varied IDs and amounts."""
    batch = []
    rng = random.Random(42)
    for i in range(count):
        base = SEED_CLAIMS[i % len(SEED_CLAIMS)].copy()
        base["claim_id"] = f"CLM-{2001 + i}"
        jitter = rng.uniform(0.92, 1.08)
        base["billed_amount"] = round(base["billed_amount"] * jitter, 2)
        batch.append(base)
    return batch