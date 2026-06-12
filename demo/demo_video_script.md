# SIA Demo Video Script (~3 minutes)

## 0:00–0:20 — Intro
"This is the Support Integrity Auditor. It detects tickets whose assigned
priority contradicts what the ticket actually says — with no human mismatch
labels. It bootstraps its own supervision from three independent signals,
trains a classifier on those pseudo-labels, and emits an evidence dossier for
every flag."

## 0:20–1:00 — Pseudo-label strategy (slide / README diagram)
- Signal 1: weighted urgency lexicon with negation scopes + escalation phrases.
- Signal 2: embedding clustering; clusters scored against severity anchors.
- Signal 3: resolution-time inverse-percentile (SLA behavior as weak proxy).
- Weighted fusion (0.45 / 0.35 / 0.20) -> inferred severity 0–3;
  mismatch when |inferred − assigned| ≥ 2.
- Show ablation table: fusion of all three beats every subset.

## 1:00–1:45 — Hidden Crisis walkthrough (Streamlit, Batch tab)
- Upload customer_support_tickets.csv.
- Open dossier TKT-100003: dashboard outage assigned **Low**.
- Point at evidence: keyword "not loading" (w=2.0), semantic cluster → High
  anchor, 41h resolution. Δ=+2 → Hidden Crisis. Grounding audit: PASS.

## 1:45–2:15 — False Alarm walkthrough
- Dossier TKT-100000: "Where is your headquarters located?" assigned **High**.
- Evidence: only interrogative keywords (w≤0.15), cluster → Low anchor.
  Δ=−2 → False Alarm.

## 2:15–2:50 — Live adversarial input (Single ticket tab)
- Type: "I noticed an entry in my sign-in history from a country I have
  never visited." Priority: Low.
- No urgency keywords at all → still flagged: Hidden Crisis, inferred
  Critical. Explain: semantic + lexicon generalization, not keyword matching.
- Bonus: type the negation trap "There is no fraud or suspicious activity
  here, I just need an invoice copy" with priority Critical → False Alarm
  (negation scope disarms the fraud keywords).

## 2:50–3:00 — Close
"83% accuracy was the bar; the system verifies at 99.3% accuracy, 0.975
macro-F1, with 10/10 on the adversarial set and zero hallucinated evidence."
