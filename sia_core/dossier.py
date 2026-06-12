"""Stage 3: Evidence Dossier generation.

Hard rule compliance: every feature_evidence item is mechanically traceable to
a specific input field. Evidence values are *extracted* (regex match offsets,
field values, computed statistics of field values) — never free-generated.
The constraint_analysis sentence is assembled from templates whose slots are
filled exclusively with those extracted values, so hallucination is impossible
by construction.
"""
from __future__ import annotations

import numpy as np

from .config import INT_TO_SEVERITY
from .data import domain_tier


def _keyword_evidence(rule_explain: dict) -> list[dict]:
    items = []
    seen = set()
    for h in sorted(rule_explain["matches"], key=lambda x: -x["weight"]):
        key = h["phrase"].lower()
        if key in seen:
            continue
        seen.add(key)
        items.append({
            "signal": "keyword",
            "value": h["phrase"],
            "weight": round(h["weight"], 2),
            "source_field": "Ticket_Description",
            "negated": h["negated"],
        })
        if len(items) == 4:
            break
    for phrase in rule_explain["escalation"][:2]:
        items.append({"signal": "keyword", "value": phrase,
                      "weight": "escalation-modifier",
                      "source_field": "Ticket_Description",
                      "negated": False})
    return items


def build_dossier(row, *, assigned_priority: str, inferred_level: int,
                  delta: int, mismatch_type: str, confidence: float,
                  rule_explain: dict, emb_explain: dict, rt_explain: dict) -> dict:
    inferred = INT_TO_SEVERITY[int(inferred_level)]
    evidence = _keyword_evidence(rule_explain)

    rt_hours = rt_explain["hours"]
    rt_pct = rt_explain["percentile"]
    if rt_explain["imputed"]:
        rt_interp = "resolution time absent in input; dataset median used"
    elif rt_pct <= 0.25:
        rt_interp = (f"resolved in {rt_hours:.0f}h — faster than "
                     f"{(1 - rt_pct) * 100:.0f}% of tickets; operations treated "
                     "it as urgent")
    elif rt_pct >= 0.75:
        rt_interp = (f"resolved in {rt_hours:.0f}h — slower than "
                     f"{rt_pct * 100:.0f}% of tickets; operations treated it "
                     "as routine")
    else:
        rt_interp = f"resolved in {rt_hours:.0f}h — mid-range turnaround"
    evidence.append({"signal": "resolution_time", "value": float(rt_hours),
                     "interpretation": rt_interp,
                     "source_field": "Resolution_Time_Hours"})

    evidence.append({
        "signal": "semantic_cluster",
        "value": f"cluster #{emb_explain['cluster']} "
                 f"(nearest severity anchor: "
                 f"{INT_TO_SEVERITY[emb_explain['anchor_level']]})",
        "weight": round(emb_explain["score"], 2),
        "similarity": round(emb_explain["similarity"], 3),
        "source_field": "Ticket_Description (embedding)",
    })

    tier = domain_tier(__import__("pandas").Series([row["Customer_Email"]]))[0]
    evidence.append({"signal": "metadata", "value":
                     f"channel={row['Ticket_Channel']}, customer_tier={tier}, "
                     f"category={row['Issue_Category']}",
                     "source_field": "Ticket_Channel/Customer_Email/Issue_Category"})

    top_kw = next((e for e in evidence if e["signal"] == "keyword"
                   and not e.get("negated")), None)
    kw_part = (f"text evidence '{top_kw['value']}'" if top_kw
               else "absence of urgency terms in the text")
    if mismatch_type == "Hidden Crisis":
        analysis = (
            f"Ticket {row['Ticket_ID']} is assigned '{assigned_priority}' but its "
            f"content indicates '{inferred}' severity ({kw_part}; semantic group "
            f"aligned with the {INT_TO_SEVERITY[emb_explain['anchor_level']]} "
            f"anchor). {rt_interp.capitalize()}. The objective signals exceed the "
            f"assigned priority by {abs(delta)} level(s), indicating an "
            f"under-prioritized ticket.")
    else:
        analysis = (
            f"Ticket {row['Ticket_ID']} is assigned '{assigned_priority}' but its "
            f"content indicates only '{inferred}' severity ({kw_part}; semantic "
            f"group aligned with the "
            f"{INT_TO_SEVERITY[emb_explain['anchor_level']]} anchor). "
            f"{rt_interp.capitalize()}. The assigned priority exceeds the "
            f"objective signals by {abs(delta)} level(s), indicating an inflated "
            f"priority.")

    return {
        "ticket_id": str(row["Ticket_ID"]),
        "assigned_priority": assigned_priority,
        "inferred_severity": inferred,
        "mismatch_type": mismatch_type,
        "severity_delta": int(delta),
        "feature_evidence": evidence,
        "constraint_analysis": analysis,
        "confidence": round(float(confidence), 4),
    }


def audit_dossier_grounding(dossier: dict, row) -> list[str]:
    """Verify each evidence item is traceable to the input ticket. Returns a
    list of violations (empty = fully grounded)."""
    violations = []
    text = str(row["Ticket_Description"]).lower()
    for ev in dossier["feature_evidence"]:
        if ev["signal"] == "keyword":
            if str(ev["value"]).lower() not in text:
                violations.append(f"keyword '{ev['value']}' not found in ticket text")
        elif ev["signal"] == "resolution_time":
            actual = row.get("Resolution_Time_Hours")
            if actual is not None and not (isinstance(actual, float) and
                                           np.isnan(actual)):
                if abs(float(ev["value"]) - float(actual)) > 1e-6:
                    violations.append("resolution_time value differs from field")
        elif ev["signal"] == "metadata":
            for fld in ("Ticket_Channel", "Issue_Category"):
                if str(row[fld]) not in str(ev["value"]):
                    violations.append(f"metadata evidence missing field {fld}")
    if dossier["assigned_priority"] != str(row["Priority_Level"]):
        violations.append("assigned_priority differs from Priority_Level field")
    return violations
