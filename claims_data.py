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
        "type": "noul",  # yes/no probability readout; 'no' (false) is always shown first
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
        "type": "score",  # ordered levels; also reports the probability-weighted level index
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
            "Admitted as full inpatient for atypical chest heaviness after a large meal. Serial troponins negative x3, EKG normal sinus rhythm. "
            "Hospitalist documented concern for unstable angina given diabetes, prior stent, and recurrent ED visits, but no ischemia was found; "
            "whether a 22-hour stay meets inpatient criteria versus observation status needs clinical validation. Cardiology performed left heart cath "
            "(normal coronaries; cine-loop summary sheet not attached) and billed an E&M visit the same day."
        ),
        "ground_truth_queue": "clinical_nurse_review",
        "ground_truth_missing_docs": "yes",
        "ground_truth_urgency": "expedited_72_hr",
        "difficulty": "hard_ambiguous"
    }
]

VARIANT_CLAIMS: List[Dict[str, Any]] = [
    # --- AUTO_ADJUDICATE (Easy Clean Claims) ---
    {
        "claim_id": "VAR-01",
        "title": "Routine Hypertension & Lipid Management",
        "patient_age": 62,
        "provider_specialty": "Family Practice",
        "billed_amount": 195.00,
        "icd10_codes": ["I10 - Essential (primary) hypertension", "E78.5 - Hyperlipidemia, unspecified"],
        "cpt_codes": ["99213 - Office/outpatient visit est, low MDM", "80061 - Lipid panel"],
        "clinical_note": "Annual follow-up for uncomplicated essential hypertension and hyperlipidemia. Patient asymptomatic, compliant with atorvastatin. BP 124/76 mmHg. Routine labs drawn and within target.",
        "ground_truth_queue": "auto_adjudicate",
        "ground_truth_missing_docs": "no",
        "ground_truth_urgency": "standard_30_day",
        "difficulty": "easy",
    },
    {
        "claim_id": "VAR-02",
        "title": "Preventive Screening Colonoscopy (Clean)",
        "patient_age": 51,
        "provider_specialty": "Gastroenterology",
        "billed_amount": 1650.00,
        "icd10_codes": ["Z12.11 - Encounter for screening for malignant neoplasm of colon"],
        "cpt_codes": ["45378 - Diagnostic screening colonoscopy"],
        "clinical_note": "Average-risk 51yo male presenting for routine colorectal cancer screening. Cecum reached with good bowel prep. Completely normal exam with no polyps or lesions identified. Return in 10 years.",
        "ground_truth_queue": "auto_adjudicate",
        "ground_truth_missing_docs": "no",
        "ground_truth_urgency": "standard_30_day",
        "difficulty": "easy",
    },
    {
        "claim_id": "VAR-03",
        "title": "Well-Woman Annual Exam with Pap Smear",
        "patient_age": 36,
        "provider_specialty": "Obstetrics & Gynecology",
        "billed_amount": 260.00,
        "icd10_codes": ["Z01.419 - Encounter for gynecological exam without abnormal findings"],
        "cpt_codes": ["99395 - Periodic preventive medicine est, 18-39 yrs", "88142 - Cytopathology, cervical or vaginal"],
        "clinical_note": "Annual routine preventive gynecological exam for healthy 36yo female. Normal pelvic exam. Cervical cytology collected. Patient counseled on diet and wellness.",
        "ground_truth_queue": "auto_adjudicate",
        "ground_truth_missing_docs": "no",
        "ground_truth_urgency": "standard_30_day",
        "difficulty": "easy",
    },
    {
        "claim_id": "VAR-04",
        "title": "Pediatric Well-Child Check & Routine Vaccinations",
        "patient_age": 4,
        "provider_specialty": "Pediatrics",
        "billed_amount": 310.00,
        "icd10_codes": ["Z00.129 - Encounter for routine child health check without abnormal findings"],
        "cpt_codes": ["99382 - Initial preventive medicine, 1-4 yrs", "90707 - MMR vaccine", "90716 - Varicella vaccine"],
        "clinical_note": "4-year-old child here for routine pre-kindergarten health supervision. Growth percentiles normal, developmental milestones achieved. Age-appropriate vaccinations administered after consent.",
        "ground_truth_queue": "auto_adjudicate",
        "ground_truth_missing_docs": "no",
        "ground_truth_urgency": "standard_30_day",
        "difficulty": "easy",
    },
    {
        "claim_id": "VAR-05",
        "title": "Routine Contact Dermatitis Follow-Up",
        "patient_age": 28,
        "provider_specialty": "Dermatology",
        "billed_amount": 140.00,
        "icd10_codes": ["L23.9 - Allergic contact dermatitis, unspecified cause"],
        "cpt_codes": ["99212 - Office/outpatient visit est, straightforward"],
        "clinical_note": "Follow-up for mild poison ivy contact dermatitis on left calf. Rash completely resolved with short course of topical hydrocortisone. No further treatment needed.",
        "ground_truth_queue": "auto_adjudicate",
        "ground_truth_missing_docs": "no",
        "ground_truth_urgency": "standard_30_day",
        "difficulty": "easy",
    },

    # --- COB_SUBROGATION (Trauma, Accidents, Third-Party Liability) ---
    {
        "claim_id": "VAR-06",
        "title": "Rideshare Passenger Multi-Vehicle Collision",
        "patient_age": 31,
        "provider_specialty": "Emergency Medicine / Radiology",
        "billed_amount": 5400.00,
        "icd10_codes": ["V49.9XXA - Car occupant injured in collision, unspecified", "S22.31XA - Fracture of one rib, right side, initial"],
        "cpt_codes": ["99285 - Emergency dept visit, high MDM", "71046 - Chest X-ray 2 views", "74177 - CT abdomen and pelvis with contrast"],
        "clinical_note": "Restrained rear-seat passenger in an Uber rideshare that was T-boned by a delivery truck at an intersection. Police report #2026-99411 filed. Isolated nondisplaced 6th rib fracture. Discharged home with analgesia.",
        "ground_truth_queue": "cob_subrogation",
        "ground_truth_missing_docs": "no",
        "ground_truth_urgency": "standard_30_day",
        "difficulty": "easy",
    },
    {
        "claim_id": "VAR-07",
        "title": "Warehouse Forklift Crush Injury (Workers Comp)",
        "patient_age": 45,
        "provider_specialty": "Orthopedic Surgery",
        "billed_amount": 7800.00,
        "icd10_codes": ["S92.301A - Fracture of unspecified metatarsal bone, right foot", "W23.1XXA - Caught between moving parts of machinery"],
        "cpt_codes": ["28475 - Closed treatment of metatarsal fracture, with manipulation", "99284 - Emergency dept visit, moderate/high MDM"],
        "clinical_note": "Distribution center logistics worker suffered foot injury when a motorized forklift backed into his steel-toed boot. First Report of Injury submitted to employer worker compensation carrier claim #WC-55214.",
        "ground_truth_queue": "cob_subrogation",
        "ground_truth_missing_docs": "no",
        "ground_truth_urgency": "standard_30_day",
        "difficulty": "easy",
    },
    {
        "claim_id": "VAR-08",
        "title": "Slip & Fall on Black Ice at Office Complex",
        "patient_age": 54,
        "provider_specialty": "Orthopedics / Urgent Care",
        "billed_amount": 3100.00,
        "icd10_codes": ["S52.501A - Unspecified fracture of lower end of right radius, initial", "W00.0XXA - Fall on same level from slip on ice"],
        "cpt_codes": ["25600 - Closed treatment of distal radial fracture", "73110 - Radiologic examination, wrist, 3 views"],
        "clinical_note": "Patient slipped on untreated black ice on the walkway directly outside corporate office building at 08:00 AM. Commercial property management safety team notified; incident report #PM-410 filed on scene. Colles fracture splinted.",
        "ground_truth_queue": "cob_subrogation",
        "ground_truth_missing_docs": "no",
        "ground_truth_urgency": "standard_30_day",
        "difficulty": "easy",
    },
    {
        "claim_id": "VAR-09",
        "title": "Crosswalk Pedestrian Struck by Commercial Van",
        "patient_age": 23,
        "provider_specialty": "Trauma Surgery",
        "billed_amount": 16400.00,
        "icd10_codes": ["V03.10XA - Pedestrian on foot injured in collision with car/van", "S82.201A - Fracture of shaft of right tibia, initial encounter"],
        "cpt_codes": ["27758 - Open treatment of tibial shaft fracture with intramedullary implant", "99291 - Critical care, first 30-74 minutes"],
        "clinical_note": "College student legally crossing in marked crosswalk with walk signal when struck by a commercial catering van making a left turn. Police collision report active with primary commercial auto liability insurance.",
        "ground_truth_queue": "cob_subrogation",
        "ground_truth_missing_docs": "no",
        "ground_truth_urgency": "standard_30_day",
        "difficulty": "easy",
    },
    {
        "claim_id": "VAR-10",
        "title": "Restaurant Staircase Fall with Handrail Failure",
        "patient_age": 60,
        "provider_specialty": "Emergency Medicine",
        "billed_amount": 4200.00,
        "icd10_codes": ["S42.001A - Fracture of unspecified part of right clavicle, initial", "W10.9XXA - Fall on and from stairs and steps"],
        "cpt_codes": ["99284 - Emergency dept visit, high MDM", "73000 - Radiologic examination, clavicle", "23500 - Closed treatment clavicle fracture"],
        "clinical_note": "Patron was descending dining area staircase at Bistro 9 when wooden handrail gave way from wall anchor, causing fall down 5 steps. Restaurant manager took photos and created formal casualty report.",
        "ground_truth_queue": "cob_subrogation",
        "ground_truth_missing_docs": "no",
        "ground_truth_urgency": "standard_30_day",
        "difficulty": "easy",
    },

    # --- CODING_AUDIT_NCCI (Unbundling, Modifier Misuse, Upcoding) ---
    {
        "claim_id": "VAR-11",
        "title": "Unbundled Colonoscopy Biopsy & Snare Polypectomy",
        "patient_age": 67,
        "provider_specialty": "Gastroenterology",
        "billed_amount": 2850.00,
        "icd10_codes": ["D12.6 - Benign neoplasm of colon, unspecified"],
        "cpt_codes": ["45385 - Colonoscopy with removal of tumor by snare technique", "45380-59 - Colonoscopy with biopsy, single or multiple (Modifier 59)"],
        "clinical_note": "Colonoscopy performed. Single 12mm ascending colon polyp removed entirely by snare cautery. Provider billed separate cold biopsy 45380 with Modifier 59 appended, but op note confirms no separate lesion was biopsied.",
        "ground_truth_queue": "coding_audit_ncci",
        "ground_truth_missing_docs": "no",
        "ground_truth_urgency": "standard_30_day",
        "difficulty": "easy",
    },
    {
        "claim_id": "VAR-12",
        "title": "Modifier 25 on Preventive Visit with Routine Lab Draw",
        "patient_age": 49,
        "provider_specialty": "Internal Medicine",
        "billed_amount": 520.00,
        "icd10_codes": ["Z00.00 - Encounter for general adult medical exam without abnormal findings"],
        "cpt_codes": ["99396 - Periodic preventive medicine est, 40-64 yrs", "99214-25 - Office visit est, moderate MDM (Modifier 25)", "36415 - Routine venipuncture"],
        "clinical_note": "Annual preventive wellness checkup. Patient had routine screening labs drawn. Billed full 99396 preventive code plus a separate 99214 with Modifier 25 without any distinct acute medical problem or complaint addressed.",
        "ground_truth_queue": "coding_audit_ncci",
        "ground_truth_missing_docs": "no",
        "ground_truth_urgency": "standard_30_day",
        "difficulty": "easy",
    },
    {
        "claim_id": "VAR-13",
        "title": "Unbundled Bilateral Knee Injections Without Modifiers",
        "patient_age": 70,
        "provider_specialty": "Physical Medicine & Rehabilitation",
        "billed_amount": 890.00,
        "icd10_codes": ["M17.0 - Bilateral primary osteoarthritis of knee"],
        "cpt_codes": ["20610 x 2 units - Arthrocentesis, aspiration and/or injection, major joint"],
        "clinical_note": "Patient with bilateral knee degenerative joint disease received intra-articular corticosteroid injections. Provider billed 2 units of 20610 without anatomical RT/LT or 50 bilateral modifiers, triggering NCCI mutually exclusive edit.",
        "ground_truth_queue": "coding_audit_ncci",
        "ground_truth_missing_docs": "no",
        "ground_truth_urgency": "standard_30_day",
        "difficulty": "easy",
    },
    {
        "claim_id": "VAR-14",
        "title": "Unbundled Lysis of Adhesions with Cholecystectomy",
        "patient_age": 53,
        "provider_specialty": "General Surgery",
        "billed_amount": 4700.00,
        "icd10_codes": ["K80.20 - Calculus of gallbladder without cholecystitis without obstruction", "K66.0 - Peritoneal adhesions"],
        "cpt_codes": ["47562 - Laparoscopic cholecystectomy", "44180-59 - Laparoscopic enterolysis, lysis of adhesions (Modifier 59)"],
        "clinical_note": "Elective laparoscopic cholecystectomy. Minimal filmy adhesions taken down in the immediate operative field to access Calot's triangle. Incidental adhesiolysis billed separately under 44180 with Modifier 59.",
        "ground_truth_queue": "coding_audit_ncci",
        "ground_truth_missing_docs": "no",
        "ground_truth_urgency": "standard_30_day",
        "difficulty": "easy",
    },
    {
        "claim_id": "VAR-15",
        "title": "Unbundled Trigger Point Injection & PT Modalities",
        "patient_age": 41,
        "provider_specialty": "Pain Management",
        "billed_amount": 680.00,
        "icd10_codes": ["M79.18 - Myalgia, other site"],
        "cpt_codes": ["20552 - Injection, single or multiple trigger points, 1-2 muscle groups", "97010 - Application of hot/cold packs", "97140-59 - Manual therapy techniques (Modifier 59)"],
        "clinical_note": "Trigger point injection of bilateral trapezius muscles followed immediately by hot packs and 15 minutes of myofascial release on the identical anatomical region. Billed unbundled physical medicine modalities.",
        "ground_truth_queue": "coding_audit_ncci",
        "ground_truth_missing_docs": "no",
        "ground_truth_urgency": "standard_30_day",
        "difficulty": "easy",
    },

    # --- CLINICAL_NURSE_REVIEW (Medical Necessity, High-Cost Drugs, Level of Care) ---
    {
        "claim_id": "VAR-16",
        "title": "Off-Label Pembrolizumab for Rare Thymic Carcinoma",
        "patient_age": 59,
        "provider_specialty": "Medical Oncology",
        "billed_amount": 22400.00,
        "icd10_codes": ["C37 - Malignant neoplasm of thymus"],
        "cpt_codes": ["96413 - Chemotherapy IV infusion, 1 hour", "J9271 - Injection, pembrolizumab, 1 mg (200 units)"],
        "clinical_note": "Stage IV thymic carcinoma with progression following first-line platinum doublet chemotherapy. Requesting off-label compassionate authorization for Keytruda every 3 weeks based on positive PD-L1 expression (pathology report pending attachment).",
        "ground_truth_queue": "clinical_nurse_review",
        "ground_truth_missing_docs": "yes",
        "ground_truth_urgency": "expedited_72_hr",
        "difficulty": "easy",
    },
    {
        "claim_id": "VAR-17",
        "title": "Inpatient Length of Stay Extension for Severe Aspiration Pneumonia",
        "patient_age": 82,
        "provider_specialty": "Pulmonology / Critical Care",
        "billed_amount": 34500.00,
        "icd10_codes": ["J69.0 - Pneumonitis due to inhalation of food and vomit", "J96.01 - Acute respiratory failure with hypoxia"],
        "cpt_codes": ["99233 x 4 - Subsequent hospital care, high MDM", "94660 - CPAP/BiPAP ventilation management"],
        "clinical_note": "82yo resident of nursing home admitted 9 days ago with aspiration pneumonia and respiratory failure. Hospital requesting authorization for days 7 through 10 of inpatient stay due to high-flow nasal cannula weaning failure and delirium.",
        "ground_truth_queue": "clinical_nurse_review",
        "ground_truth_missing_docs": "no",
        "ground_truth_urgency": "expedited_72_hr",
        "difficulty": "easy",
    },
    {
        "claim_id": "VAR-18",
        "title": "CAR-T Cell Therapy Commercial Authorization Request",
        "patient_age": 48,
        "provider_specialty": "Hematology / Cellular Therapy",
        "billed_amount": 385000.00,
        "icd10_codes": ["C83.30 - Diffuse large B-cell lymphoma, unspecified site"],
        "cpt_codes": ["0540T - Chimeric antigen receptor T-cell administration", "Q2041 - Axicabtagene ciloleucel, up to 200 million autologous anti-CD19 CAR T cells"],
        "clinical_note": "Refractory DLBCL failing two prior lines of chemo-immunotherapy including R-CHOP and salvage ICE. Multi-disciplinary cellular therapy board approved CAR-T Yescarta regimen. Bone marrow biopsy cytogenetics report from outside hospital not yet received.",
        "ground_truth_queue": "clinical_nurse_review",
        "ground_truth_missing_docs": "yes",
        "ground_truth_urgency": "stat_24_hr",
        "difficulty": "easy",
    },
    {
        "claim_id": "VAR-19",
        "title": "Inpatient Rehabilitation Facility Admission Post-Stroke",
        "patient_age": 68,
        "provider_specialty": "Physical Medicine & Rehabilitation",
        "billed_amount": 26800.00,
        "icd10_codes": ["I63.411 - Cerebral infarction due to embolism of right middle cerebral artery", "G81.91 - Hemiplegia, unspecified affecting right dominant side"],
        "cpt_codes": ["99223 - Initial hospital inpatient care, high MDM"],
        "clinical_note": "Patient with acute ischemic MCA stroke completing acute inpatient stabilization. Requesting transfer to acute inpatient rehab facility (IRF). Patient can tolerate 3 hours of therapy daily across PT, OT, and ST.",
        "ground_truth_queue": "clinical_nurse_review",
        "ground_truth_missing_docs": "no",
        "ground_truth_urgency": "expedited_72_hr",
        "difficulty": "easy",
    },
    {
        "claim_id": "VAR-20",
        "title": "Second-Line Biologic Ustekinumab for Refractory Crohn's",
        "patient_age": 33,
        "provider_specialty": "Gastroenterology",
        "billed_amount": 21900.00,
        "icd10_codes": ["K50.012 - Crohn's disease of small intestine with intestinal obstruction"],
        "cpt_codes": ["96413 - Chemotherapy/biologic IV infusion, 1 hr", "J3357 - Ustekinumab, for intravenous injection, 1 mg (390 units)"],
        "clinical_note": "Severe ileal Crohn's with primary loss of response to adalimumab and secondary failure to infliximab. Ordering weight-based Stelara 390 mg IV induction dose. Fecal calprotectin level elevated at 820 ug/g (lab sheet not attached in portal).",
        "ground_truth_queue": "clinical_nurse_review",
        "ground_truth_missing_docs": "yes",
        "ground_truth_urgency": "expedited_72_hr",
        "difficulty": "easy",
    },

    # --- SIU_FRAUD_REVIEW (Phantom Billing, Impossible Hours, Excessive Services) ---
    {
        "claim_id": "VAR-21",
        "title": "Mass Podiatric Nail Debridement in Single Day Batch",
        "patient_age": 79,
        "provider_specialty": "Podiatry",
        "billed_amount": 14200.00,
        "icd10_codes": ["B35.1 - Tinea unguium"],
        "cpt_codes": ["11721 x 80 units - Debridement of nails, any method, 6 or more", "99214 x 80 units - Office visit est, moderate MDM"],
        "clinical_note": "Batch billing submission for 80 individual patients at Sunnyvale Assisted Living facility on single date of service by solo traveling podiatrist. Cumulative billed clinical time exceeds 48 hours in one 24-hour day.",
        "ground_truth_queue": "siu_fraud_review",
        "ground_truth_missing_docs": "no",
        "ground_truth_urgency": "standard_30_day",
        "difficulty": "easy",
    },
    {
        "claim_id": "VAR-22",
        "title": "DME Billing for Power Wheelchair Post-Mortem",
        "patient_age": 88,
        "provider_specialty": "Durable Medical Equipment",
        "billed_amount": 12500.00,
        "icd10_codes": ["G82.20 - Paraplegia, unspecified"],
        "cpt_codes": ["K0823 - Power wheelchair, group 2 standard, captain's chair", "E0950 - Wheelchair accessory, tray"],
        "clinical_note": "DME supplier billed customized motorized wheelchair with date of service 3 weeks subsequent to beneficiary recorded date of death in national Social Security Death Master File. Prescribing physician signature is forged digital stamp.",
        "ground_truth_queue": "siu_fraud_review",
        "ground_truth_missing_docs": "no",
        "ground_truth_urgency": "standard_30_day",
        "difficulty": "easy",
    },
    {
        "claim_id": "VAR-23",
        "title": "Impossible Telehealth Volume 60 Visits in 4 Hours",
        "patient_age": 44,
        "provider_specialty": "General Practice / Telemedicine",
        "billed_amount": 9600.00,
        "icd10_codes": ["J06.9 - Acute upper respiratory infection, unspecified"],
        "cpt_codes": ["99215 - Office visit est, high level MDM, 40-54 mins"],
        "clinical_note": "Telehealth vendor claims audit reveals solo practitioner billed 60 separate Level 5 (99215) comprehensive evaluation visits between 1:00 PM and 5:00 PM on the same afternoon. System connection logs show average duration of 45 seconds per encounter.",
        "ground_truth_queue": "siu_fraud_review",
        "ground_truth_missing_docs": "no",
        "ground_truth_urgency": "standard_30_day",
        "difficulty": "easy",
    },
    {
        "claim_id": "VAR-24",
        "title": "Quad-Limb EMG & Nerve Conduction for Simple Headache",
        "patient_age": 37,
        "provider_specialty": "Neurology Clinic",
        "billed_amount": 6200.00,
        "icd10_codes": ["G44.209 - Tension-type headache, unspecified"],
        "cpt_codes": ["95913 - Nerve conduction studies, 13 or more studies", "95886 x 4 - Needle electromyography, each extremity"],
        "clinical_note": "Routine tension headache patient subjected to full bilateral upper and lower extremity needle electromyography and 14 separate nerve conduction tests. Pattern identified across 100% of clinic new patients regardless of primary complaint.",
        "ground_truth_queue": "siu_fraud_review",
        "ground_truth_missing_docs": "no",
        "ground_truth_urgency": "standard_30_day",
        "difficulty": "easy",
    },

    # --- HARD / AMBIGUOUS EDGE CASES (Deliberately Produce Split Confidences) ---
    {
        "claim_id": "VAR-25",
        "title": "Ambiguous Subrogation vs Auto-Adjudicate: Curb Fall Near Store",
        "patient_age": 46,
        "provider_specialty": "Urgent Care",
        "billed_amount": 1820.00,
        "icd10_codes": ["S93.401A - Sprain of unspecified ligament of right ankle, initial encounter"],
        "cpt_codes": ["99203 - Office visit new, low MDM", "73600 - Radiologic examination, ankle, 2 views"],
        "clinical_note": "Patient twisted right ankle while walking along outdoor retail plaza sidewalk. Provider note states: 'Patient slipped near curb transition outside grocery store, but states it may just be clumsy footing after jogging; no incident report or business notification made.' Radiographs negative for fracture.",
        "ground_truth_queue": "cob_subrogation",
        "ground_truth_missing_docs": "no",
        "ground_truth_urgency": "standard_30_day",
        "difficulty": "hard_ambiguous",
    },
    {
        "claim_id": "VAR-26",
        "title": "Ambiguous Coding vs Nurse Review: Emergency Revision Surgery",
        "patient_age": 63,
        "provider_specialty": "Colorectal Surgery",
        "billed_amount": 16400.00,
        "icd10_codes": ["T81.31XA - Disruption of external operation wound, not elsewhere classified"],
        "cpt_codes": ["49002 - Reopening of recent laparotomy", "99291 - Critical care, first 74 minutes"],
        "clinical_note": "Patient returned to operating room on post-op day 4 following low anterior resection due to sudden fascial dehiscence. Surgeon billed emergency re-exploration alongside critical care code; operative report dictation is pending transcription and medical necessity of surgical staging is borderline.",
        "ground_truth_queue": "clinical_nurse_review",
        "ground_truth_missing_docs": "yes",
        "ground_truth_urgency": "expedited_72_hr",
        "difficulty": "hard_ambiguous",
    },
    {
        "claim_id": "VAR-27",
        "title": "Ambiguous Nurse Review vs Auto-Adjudicate: Knee Arthroscopy on Cusp",
        "patient_age": 64,
        "provider_specialty": "Orthopedic Surgery",
        "billed_amount": 4900.00,
        "icd10_codes": ["M23.22 - Derangement of meniscus due to old tear or injury, right knee", "M17.11 - Unilateral primary osteoarthritis, right knee"],
        "cpt_codes": ["29881 - Arthroscopy, knee, surgical; with meniscectomy"],
        "clinical_note": "64yo male underwent right partial medial meniscectomy. Mild background tri-compartmental osteoarthritis noted. Chart documents 4 months of physical therapy and NSAIDs without relief, but formal intra-operative photographic prints are missing from attachment. Falls on borderline of medical policy criteria.",
        "ground_truth_queue": "clinical_nurse_review",
        "ground_truth_missing_docs": "yes",
        "ground_truth_urgency": "standard_30_day",
        "difficulty": "hard_ambiguous",
    },
    {
        "claim_id": "VAR-28",
        "title": "Ambiguous SIU vs Coding Audit: High-Frequency Allergy Injections",
        "patient_age": 35,
        "provider_specialty": "Allergy & Immunology",
        "billed_amount": 2900.00,
        "icd10_codes": ["J30.1 - Allergic rhinitis due to pollen"],
        "cpt_codes": ["95165 x 30 units - Antigen therapy preparation", "95117 x 3 units - Multiple allergen injections"],
        "clinical_note": "Solo allergist billing maximum units of antigen extract preparation along with 3 weekly injection encounters across 4 family members simultaneously. Documentation template appears auto-populated, raising questions between aggressive code maximization versus abusive billing patterns.",
        "ground_truth_queue": "coding_audit_ncci",
        "ground_truth_missing_docs": "no",
        "ground_truth_urgency": "standard_30_day",
        "difficulty": "hard_ambiguous",
    },
    {
        "claim_id": "VAR-29",
        "title": "Ambiguous Nurse Review vs Subrogation: Degenerative Disc with Work Flare",
        "patient_age": 50,
        "provider_specialty": "Spine Surgery",
        "billed_amount": 12800.00,
        "icd10_codes": ["M51.16 - Intervertebral disc disorders with radiculopathy, lumbar region"],
        "cpt_codes": ["63030 - Lumbar laminotomy with decompression of nerve root", "72148 - MRI lumbar spine without contrast"],
        "clinical_note": "Patient presents with acute L5 radiculopathy following heavy lifting incident at employment warehouse. However, chart confirms extensive 10-year pre-existing history of chronic degenerative lumbar disc disease, and workers comp carrier issued formal initial controversion. Operative report pending hospital release.",
        "ground_truth_queue": "clinical_nurse_review",
        "ground_truth_missing_docs": "yes",
        "ground_truth_urgency": "expedited_72_hr",
        "difficulty": "hard_ambiguous",
    },
    {
        "claim_id": "VAR-30",
        "title": "Ambiguous Inpatient Observation vs Admission for Bradycardia",
        "patient_age": 75,
        "provider_specialty": "Cardiology / Hospitalist",
        "billed_amount": 9200.00,
        "icd10_codes": ["R00.1 - Bradycardia, unspecified", "I44.1 - Atrioventricular block, second degree"],
        "cpt_codes": ["99222 - Initial hospital inpatient care, moderate MDM", "93010 - Electrocardiogram routine report"],
        "clinical_note": "75yo male admitted for asymptomatic Mobitz I AV block discovered on routine clinic EKG. 26-hour monitoring stay without hemodynamic instability or syncope. Billed as full acute inpatient stay rather than outpatient observation with cardiac telemetry.",
        "ground_truth_queue": "coding_audit_ncci",
        "ground_truth_missing_docs": "no",
        "ground_truth_urgency": "standard_30_day",
        "difficulty": "hard_ambiguous",
    },
]

ALL_CLAIMS: List[Dict[str, Any]] = SEED_CLAIMS + VARIANT_CLAIMS
EASY_CLAIMS: List[Dict[str, Any]] = [c for c in ALL_CLAIMS if c.get("difficulty") == "easy"]
HARD_CLAIMS: List[Dict[str, Any]] = [c for c in ALL_CLAIMS if c.get("difficulty") == "hard_ambiguous"]


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


def get_batch_claims(count: int = 100, seed: int = 42) -> List[Dict[str, Any]]:
    """Generates a deterministic N-claim batch from the seed and variant pool with an exact 80/20 easy/hard split.

    Applies deterministic amount jitter (0.92 - 1.08) and sequential claim IDs (CLM-2001, CLM-2002, ...).
    """
    rng = random.Random(seed)
    n_hard = round(count * 0.20)
    n_easy = count - n_hard

    sampled_easy = [EASY_CLAIMS[i % len(EASY_CLAIMS)].copy() for i in range(n_easy)]
    sampled_hard = [HARD_CLAIMS[i % len(HARD_CLAIMS)].copy() for i in range(n_hard)]

    combined = sampled_easy + sampled_hard
    rng.shuffle(combined)

    batch = []
    for i, base in enumerate(combined):
        item = base.copy()
        item["claim_id"] = f"CLM-{2001 + i}"
        jitter = rng.uniform(0.92, 1.08)
        item["billed_amount"] = round(item["billed_amount"] * jitter, 2)
        batch.append(item)
    return batch