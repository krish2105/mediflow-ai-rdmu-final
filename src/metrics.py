"""
metrics.py
Performance metrics for allocation evaluation.
RDMU Concepts: Fairness (Jain's Index), Efficiency, Urgency Handling, Policy Evaluation.
Made by: Krishna Mathur | AS25DXB018 | MAIB September
"""
import pandas as pd
import numpy as np


def _jains_fairness(values):
    """Jain's Fairness Index = (sum xi)^2 / (n * sum xi^2)."""
    arr = np.array(values, dtype=float)
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    return float((arr.sum() ** 2) / (n * (arr ** 2).sum()))


def compute_metrics(allocation_df, simulation_df=None, resources_df=None):
    """
    Compute a metrics dict from allocation results and optional simulation timeline.

    Parameters
    ----------
    allocation_df : pd.DataFrame  — batch allocation output (status, urgency_level, …)
    simulation_df : pd.DataFrame  — time-step simulation timeline (optional)
    resources_df  : pd.DataFrame  — hospital resource table (optional, improves utilisation KPI)

    Returns a flat dict of scalar metrics.
    """
    m = {}

    if allocation_df is None or len(allocation_df) == 0:
        return m

    treated_mask    = allocation_df["status"] == "Treated"
    stabilized_mask = allocation_df["status"] == "Stabilized"
    queued_mask     = allocation_df["status"].isin(["Queued"])
    transferred_mask = allocation_df["status"] == "Transferred"

    m["total_patients"]    = len(allocation_df)
    m["treated_count"]     = int(treated_mask.sum())
    m["stabilized_count"]  = int(stabilized_mask.sum())
    m["queued_count"]      = int(queued_mask.sum())
    m["transferred_count"] = int(transferred_mask.sum())

    m["avg_wait_time"] = round(float(allocation_df["wait_time"].mean()), 1)
    m["max_wait_time"] = round(float(allocation_df["wait_time"].max()), 1)
    m["avg_urgency"]   = round(float(allocation_df["urgency_level"].mean()), 2)

    # ── Jain's Fairness Index across demographic groups ───────────────────
    if "fairness_group" in allocation_df.columns:
        # Avoid groupby.apply DeprecationWarning — use vectorised aggregation
        group_total   = allocation_df.groupby("fairness_group").size()
        group_treated = (
            allocation_df[treated_mask]
            .groupby("fairness_group")
            .size()
            .reindex(group_total.index, fill_value=0)
        )
        group_rates = group_treated / group_total.clip(lower=1)
        m["fairness_index"] = round(_jains_fairness(group_rates.values), 3)
    else:
        m["fairness_index"] = 0.0

    # ── Urgency handling: % of critical (≥4) patients treated ────────────
    crit = allocation_df[allocation_df["urgency_level"] >= 4]
    if len(crit) > 0:
        m["urgency_handling_score"] = round(
            float((crit["status"] == "Treated").sum() / len(crit)), 3
        )
    else:
        m["urgency_handling_score"] = 1.0

    # ── Patient outcome score: urgency-weighted status quality ────────────
    def _outcome(row):
        if row["status"] == "Treated":
            return row["urgency_level"] * 10.0
        elif row["status"] == "Stabilized":
            return row["urgency_level"] * 5.0
        elif row["status"] == "Queued":
            return max(0.0, row["urgency_level"] * 3.0 - row["wait_time"] * 0.1)
        return 0.0

    m["patient_outcome_score"] = round(
        float(allocation_df.apply(_outcome, axis=1).mean()), 2
    )

    # ── Resource utilisation: doctors used / doctors available ───────────
    # If resources_df provided, show true occupancy; else use treatment rate.
    if resources_df is not None:
        try:
            total_docs = int(
                resources_df.loc[
                    resources_df["resource_name"] == "Doctors", "available_count"
                ].values[0]
            )
            total_beds = int(
                resources_df.loc[
                    resources_df["resource_name"] == "Beds", "available_count"
                ].values[0]
            )
            # Doctors are the binding bottleneck (each treated patient uses 1)
            docs_used = min(m["treated_count"], total_docs)
            beds_used = min(m["treated_count"], total_beds)
            m["resource_utilization"] = round(
                (docs_used / max(total_docs, 1) + beds_used / max(total_beds, 1)) / 2.0,
                3,
            )
        except Exception:
            m["resource_utilization"] = round(
                m["treated_count"] / max(m["total_patients"], 1), 3
            )
    else:
        m["resource_utilization"] = round(
            m["treated_count"] / max(m["total_patients"], 1), 3
        )

    # ── Capacity violations: critical patients NOT given full treatment ───
    m["capacity_violation_count"] = int(
        (
            (allocation_df["urgency_level"] >= 4)
            & (allocation_df["status"] != "Treated")
        ).sum()
    )

    # ── Optional simulation-level metrics ─────────────────────────────────
    if simulation_df is not None and len(simulation_df) > 0:
        m["avg_bed_utilization"]    = round(float(simulation_df["bed_utilization"].mean()), 3)
        m["avg_doctor_utilization"] = round(float(simulation_df["doctor_utilization"].mean()), 3)
        m["peak_queue_length"]      = int(simulation_df["queue_length"].max())
        m["avg_sim_wait"]           = round(float(simulation_df["avg_wait_time"].mean()), 2)

    return m
