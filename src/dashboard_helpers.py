"""
dashboard_helpers.py
Helper utilities for Streamlit dashboard pages.
Made by: Krishna Mathur | AS25DXB018 | MAIB September
"""
import streamlit as st
import pandas as pd
import numpy as np


# ------------------------------------------------------------------
# KPI card row
# ------------------------------------------------------------------
def display_kpi_cards(metrics, resources_df, patients_df):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Patients", len(patients_df))
        beds = _res_val(resources_df, "Beds", "available_count")
        st.metric("Available Beds", beds)
    with c2:
        docs = _res_val(resources_df, "Doctors", "available_count")
        st.metric("Available Doctors", docs)
        nurses = _res_val(resources_df, "Nurses", "available_count")
        st.metric("Available Nurses", nurses)
    with c3:
        st.metric("Avg Patient Urgency", metrics.get("avg_urgency", "—"))
        st.metric("Treated (Concurrent)", metrics.get("treated_count", "—"))
    with c4:
        avg_wait = metrics.get("avg_wait_time", "—")
        label_wait = f"{avg_wait} min" if isinstance(avg_wait, (int, float)) else avg_wait
        st.metric("Avg Wait Time", label_wait)
        fi = metrics.get("fairness_index", "—")
        util = metrics.get("resource_utilization", "—")
        util_label = f"{util:.0%}" if isinstance(util, float) else util
        st.metric("Resource Utilisation", util_label)
        st.metric("Fairness Index (Jain's)", fi)


def _res_val(df, name, col):
    try:
        return int(df.loc[df["resource_name"] == name, col].values[0])
    except Exception:
        return "—"


# ------------------------------------------------------------------
# RDMU concept mapping table
# ------------------------------------------------------------------
def display_concept_table():
    data = {
        "RDMU Concept": [
            "Uncertainty",
            "State",
            "Action",
            "Reward",
            "Policy",
            "Reinforcement Learning",
            "Constraint-Based Decision",
            "Fairness",
            "Sensitivity Analysis",
            "Human-in-the-Loop",
        ],
        "Meaning": [
            "Unknown future patient arrivals and severity distributions",
            "Snapshot of the system: urgency, wait time, capacity pressure, fairness gap",
            "Allocation decision for each patient: Full Care / Stabilise / Queue / Transfer",
            "Scalar feedback signal telling the agent how good its decision was",
            "A mapping from every state to the best action, learned by the RL agent",
            "Q-learning updates the policy from simulated hospital experience",
            "Hard resource limits (beds, doctors, nurses) must not be exceeded",
            "Equal treatment opportunity across demographic and social groups",
            "Varying parameters to observe how outcomes change (fairness vs efficiency)",
            "Clinician reviews AI recommendations and can override them",
        ],
        "Where in This App": [
            "Stochastic patient arrivals (Poisson process) in Simulation page",
            "RL Policy Engine — state space visualisation and Q-table",
            "Q-table action selection and Optimiser allocation decisions",
            "Reward curve in RL Policy Engine page",
            "Learned Policy Table and Heatmap in RL page",
            "RL Policy Engine — Q-learning training loop",
            "Optimisation Baseline — resource feasibility checks",
            "Fairness Index metric; Sensitivity Analysis page",
            "Sensitivity Analysis page — parameter sliders",
            "Resource Allocation Simulator — manual resource controls",
        ],
        "Example from Project": [
            "Arrivals follow Poisson(λ); severity drawn from Normal distribution",
            "State = (Urgency=5, Wait=35 min, Capacity=75%, Fairness=Underserved)",
            "Action 0 = Allocate Full Care to a Level-5 critical patient",
            "Reward = +70 for treating critical patient before unsafe wait threshold",
            "Policy: {Urgency=5, Wait>30 min} → Always Allocate Full Care",
            "Bellman update: Q(s,a) ← Q(s,a) + α[r + γ max Q(s') - Q(s,a)]",
            "Max 30 beds; 5 reserved as emergency buffer for urgency ≥ 4",
            "Jain's Fairness Index ≥ 0.90 target across Group A / B / C",
            "Fairness weight swept 0.1 → 0.9 shows Fairness ↑ as Efficiency ↓",
            "Clinician overrides Transfer → Full Care for a borderline patient",
        ],
    }
    return pd.DataFrame(data)


# ------------------------------------------------------------------
# Sensitivity analysis runner
# ------------------------------------------------------------------
def run_sensitivity_analysis(patients_df, resources_df, n_points=20):
    """Sweep fairness weight and compute metrics for each point."""
    from src.optimizer import ConstraintOptimizer
    from src.metrics import compute_metrics

    results = []
    fw_vals = np.linspace(0.05, 0.90, n_points)
    subset = patients_df.head(120)

    for fw in fw_vals:
        ew = max(0.05, round(1.0 - fw, 2))
        uw = round(min(0.70, fw * 0.4 + 0.35), 2)

        opt = ConstraintOptimizer(fairness_weight=fw, urgency_weight=uw, efficiency_weight=ew)
        alloc_df, _ = opt.allocate(subset, resources_df)
        m = compute_metrics(alloc_df, resources_df=resources_df)

        results.append({
            "fairness_weight": round(fw, 2),
            "efficiency_weight": round(ew, 2),
            "urgency_weight": round(uw, 2),
            "fairness_index": m.get("fairness_index", 0),
            "resource_utilization": m.get("resource_utilization", 0),
            "avg_wait_time": m.get("avg_wait_time", 0),
            "treated_count": m.get("treated_count", 0),
            "patient_outcome_score": m.get("patient_outcome_score", 0),
        })

    return pd.DataFrame(results)


# ------------------------------------------------------------------
# Colour-coded status badge
# ------------------------------------------------------------------
STATUS_COLOR = {
    "Treated": "🟢",
    "Stabilized": "🟡",
    "Queued": "🟠",
    "Transferred": "🔴",
}


def badge(status):
    return STATUS_COLOR.get(status, "⚪") + " " + status
