"""Support Integrity Auditor - Streamlit web app.

Run:  streamlit run app.py
Deploy: push repo to GitHub -> share.streamlit.io -> point at app.py.

Features (per problem statement):
  - single-ticket form input or batch CSV upload
  - binary judgment + full Evidence Dossier per ticket
  - Priority Mismatch Dashboard (flag distribution, mismatch types,
    top contributing signals)
  - severity delta heatmap across ticket categories and channels
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predict import dossiers_for, predict_frame  # noqa: E402
from sia_core.data import load_tickets  # noqa: E402

ARTIFACTS = os.environ.get("SIA_ARTIFACTS", "artifacts")

st.set_page_config(page_title="Support Integrity Auditor", page_icon="🔎",
                   layout="wide")
st.title("🔎 Support Integrity Auditor (SIA)")
st.caption("Self-supervised detection of priority mismatches in support "
           "tickets — evidence-grounded, hallucination-free dossiers.")


@st.cache_resource
def _check_artifacts():
    need = ["run_config.json", "signals.json"]
    missing = [f for f in need if not os.path.exists(os.path.join(ARTIFACTS, f))]
    return missing


missing = _check_artifacts()
if missing:
    st.error(f"Missing artifacts in '{ARTIFACTS}/': {missing}. "
             "Run train_pipeline.py first.")
    st.stop()

tab_single, tab_batch, tab_dash = st.tabs(
    ["Single ticket", "Batch CSV", "Mismatch Dashboard"])

REQUIRED = ["Ticket_ID", "Ticket_Subject", "Ticket_Description",
            "Issue_Category", "Priority_Level", "Ticket_Channel",
            "Customer_Email"]


def render_dossier(d: dict):
    badge = "🚨 Hidden Crisis" if d["mismatch_type"] == "Hidden Crisis" \
        else "📉 False Alarm"
    st.markdown(
        f"**{badge}** — assigned **{d['assigned_priority']}**, inferred "
        f"**{d['inferred_severity']}** (Δ {d['severity_delta']:+d}), "
        f"confidence {d['confidence']:.2%}")
    audit = d.get("grounding_audit", "PASS")
    st.markdown("Grounding audit: "
                + ("✅ PASS" if audit == "PASS" else f"❌ {audit}"))
    st.json(d)


def run_batch(df: pd.DataFrame):
    out, ctx = predict_frame(df, ARTIFACTS)
    docs = dossiers_for(df, out, ctx, "flagged")
    return out, ctx, docs


# ---------------------------------------------------------------- single
with tab_single:
    with st.form("single"):
        c1, c2 = st.columns(2)
        with c1:
            subject = st.text_input("Ticket subject", "Quick question")
            description = st.text_area(
                "Ticket description",
                "Hi Support, I noticed an entry in my sign-in history from a "
                "country I have never visited.")
            category = st.selectbox("Issue category", [
                "Technical", "Billing", "Account", "General Inquiry", "Fraud"])
        with c2:
            priority = st.selectbox("Assigned priority (to audit)",
                                    ["Low", "Medium", "High", "Critical"])
            channel = st.selectbox("Channel", ["Chat", "Email", "Web Form"])
            email = st.text_input("Customer email", "user@example.com")
            rt = st.number_input("Resolution time (hours, 0 = unknown)",
                                 min_value=0.0, value=0.0, step=1.0)
        go = st.form_submit_button("Audit ticket", type="primary")
    if go:
        row = pd.DataFrame([{
            "Ticket_ID": "WEB-0001", "Customer_Name": "Web User",
            "Customer_Email": email, "Ticket_Subject": subject,
            "Ticket_Description": description, "Issue_Category": category,
            "Priority_Level": priority, "Ticket_Channel": channel,
            "Submission_Date": "", "Assigned_Agent": "", "Satisfaction_Score": 3,
            "Resolution_Time_Hours": rt if rt > 0 else np.nan}])
        out, ctx, docs = run_batch(row)
        j = out["judgment"].iloc[0]
        if j == "Mismatched":
            st.error(f"Judgment: **{j}**")
            render_dossier(docs[0])
        else:
            st.success(
                f"Judgment: **Consistent** — inferred severity "
                f"{out['inferred_severity'].iloc[0]} matches assigned "
                f"{priority} (confidence {out['confidence'].iloc[0]:.2%}).")

# ----------------------------------------------------------------- batch
with tab_batch:
    up = st.file_uploader("Upload ticket CSV", type="csv")
    st.caption("Required columns: " + ", ".join(REQUIRED)
               + " (+ optional Resolution_Time_Hours)")
    if up is not None:
        df = pd.read_csv(up)
        miss = [c for c in REQUIRED if c not in df.columns]
        if miss:
            st.error(f"Missing columns: {miss}")
        else:
            if "Resolution_Time_Hours" not in df.columns:
                df["Resolution_Time_Hours"] = np.nan
            df["Ticket_Subject"] = df["Ticket_Subject"].fillna("").astype(str)
            df["Ticket_Description"] = (
                df["Ticket_Description"].fillna("").astype(str))
            out, ctx, docs = run_batch(df)
            st.session_state["batch"] = (df, out, docs)
    if "batch" in st.session_state:
        df, out, docs = st.session_state["batch"]
        n_flag = int((out["judgment"] == "Mismatched").sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tickets", len(out))
        c2.metric("Flagged", n_flag)
        c3.metric("Hidden Crises",
                  int((out["mismatch_type"] == "Hidden Crisis").sum()))
        c4.metric("False Alarms",
                  int((out["mismatch_type"] == "False Alarm").sum()))
        st.dataframe(out, use_container_width=True)
        st.download_button("Download predictions.csv",
                           out.to_csv(index=False), "predictions.csv")
        st.download_button("Download dossiers.json",
                           json.dumps(docs, indent=2), "dossiers.json")
        if docs:
            pick = st.selectbox("Inspect dossier",
                                [d["ticket_id"] for d in docs])
            render_dossier(next(d for d in docs if d["ticket_id"] == pick))

# ------------------------------------------------------------- dashboard
with tab_dash:
    if "batch" not in st.session_state:
        st.info("Upload a CSV in the Batch tab first — the dashboard "
                "visualizes the latest batch run.")
    else:
        df, out, docs = st.session_state["batch"]
        flagged = out[out["judgment"] == "Mismatched"]
        st.subheader("Priority Mismatch Dashboard")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Flagged tickets by assigned priority**")
            st.bar_chart(flagged["Priority_Level"].value_counts())
            st.markdown("**Mismatch types**")
            st.bar_chart(flagged["mismatch_type"].value_counts())
        with c2:
            st.markdown("**Top contributing signals (flagged tickets)**")
            sig_counts = {}
            for d in docs:
                for ev in d["feature_evidence"]:
                    if ev["signal"] == "keyword" and not ev.get("negated") \
                            and isinstance(ev.get("weight"), (int, float)) \
                            and ev["weight"] >= 1.0:
                        sig_counts["keyword: " + str(ev["value"]).lower()] = \
                            sig_counts.get(
                                "keyword: " + str(ev["value"]).lower(), 0) + 1
            top = pd.Series(sig_counts).sort_values(ascending=False).head(12)
            st.bar_chart(top)
        st.markdown("**Severity delta heatmap — category × channel**")
        merged = df[["Issue_Category", "Ticket_Channel"]].join(
            out[["severity_delta"]])
        pivot = merged.pivot_table(index="Issue_Category",
                                   columns="Ticket_Channel",
                                   values="severity_delta", aggfunc="mean")
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 3.5))
            im = ax.imshow(pivot.to_numpy(), cmap="RdBu_r", vmin=-1.5,
                           vmax=1.5, aspect="auto")
            ax.set_xticks(range(len(pivot.columns)), pivot.columns)
            ax.set_yticks(range(len(pivot.index)), pivot.index)
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    ax.text(j, i, f"{pivot.iloc[i, j]:+.2f}",
                            ha="center", va="center", fontsize=9)
            ax.set_title("Mean severity delta (inferred − assigned)")
            fig.colorbar(im, shrink=0.8)
            st.pyplot(fig)
        except Exception:
            st.dataframe(pivot.style.background_gradient(
                cmap="RdBu_r", vmin=-1.5, vmax=1.5).format("{:+.2f}"))
