"""
app.py  —  MediFlow AI: Dynamic Hospital Emergency Resource Allocation System
           using Reinforcement Learning
Course  : Reasoning and Decision Making under Uncertainty — MAIB DSC 103
Student : Krishna Mathur | AS25DXB018 | MAIB September
Topic   : Topic 1 — Dynamic Resource Allocation using Reinforcement Learning
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediFlow AI — Hospital Resource Allocation",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Ensure directories ─────────────────────────────────────────────────────────
for d in ["data", "outputs", "slides"]:
    Path(d).mkdir(parents=True, exist_ok=True)

# ── Lazy imports from src ──────────────────────────────────────────────────────
from src.data_generator import load_data, save_data
from src.simulation_engine import run_simulation
from src.rl_agent import QLearningAgent
from src.optimizer import ConstraintOptimizer
from src.metrics import compute_metrics
from src.visualizations import (
    plot_urgency_distribution, plot_fairness_distribution, plot_resource_bar,
    plot_reward_curve, plot_policy_heatmap, plot_allocation_timeline,
    plot_policy_comparison_bar, plot_sensitivity, plot_metrics_heatmap,
    plot_action_distribution,
)
from src.dashboard_helpers import (
    display_kpi_cards, display_concept_table, run_sensitivity_analysis,
)

# ── Link placeholders (replace after deployment) ───────────────────────────────
COLAB_LINK = "[PASTE_ACCESSIBLE_COLAB_LINK_HERE]"
GITHUB_LINK = "[PASTE_GITHUB_REPO_LINK_HERE]"
STREAMLIT_LINK = "[PASTE_STREAMLIT_APP_LINK_HERE]"

# ── Session state initialisation ──────────────────────────────────────────────
def _init_state():
    if "data_loaded" not in st.session_state:
        st.session_state.patients_df, st.session_state.resources_df = load_data("data")
        st.session_state.data_loaded = True
    for key in ["rl_agent", "rl_rewards", "rl_alloc_df",
                "opt_alloc_df", "opt_res_summary",
                "sim_df", "sens_df"]:
        if key not in st.session_state:
            st.session_state[key] = None

_init_state()

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🏥 MediFlow AI")
st.sidebar.markdown("**Dynamic Hospital Resource Allocation**")
st.sidebar.markdown("---")

PAGES = [
    "1 · Executive Dashboard",
    "2 · Data Preview",
    "3 · Resource Allocation Simulator",
    "4 · RL Policy Engine",
    "5 · Optimisation Baseline",
    "6 · Policy Comparison",
    "7 · Sensitivity Analysis",
    "8 · Metrics and Performance",
    "9 · RDMU Concept Mapping",
    "10 · Limitations and Future Scope",
]
page = st.sidebar.radio("Navigate", PAGES)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Student:** Krishna Mathur")
st.sidebar.markdown(f"**ID:** AS25DXB018 | MAIB Sep")
st.sidebar.markdown(f"**Course:** MAIB DSC 103")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _show_links():
    st.markdown(
        f"**Google Colab:** `{COLAB_LINK}`  |  "
        f"**GitHub:** `{GITHUB_LINK}`  |  "
        f"**Streamlit App:** `{STREAMLIT_LINK}`"
    )

# ─── Page 1: Executive Dashboard ──────────────────────────────────────────────
def page_executive_dashboard():
    st.title("🏥 MediFlow AI — Executive Dashboard")
    st.markdown(
        "**Course:** Reasoning and Decision Making under Uncertainty — MAIB DSC 103  \n"
        "**Made by:** Krishna Mathur | AS25DXB018 | MAIB September  \n"
        "**Topic:** Topic 1 — Dynamic Resource Allocation using Reinforcement Learning"
    )
    _show_links()
    st.markdown("---")

    patients_df = st.session_state.patients_df
    resources_df = st.session_state.resources_df

    # Quick allocation for KPIs (use optimizer with defaults)
    opt = ConstraintOptimizer()
    alloc_df, _ = opt.allocate(patients_df.head(200), resources_df)
    metrics = compute_metrics(alloc_df, resources_df=resources_df)

    # KPI cards
    st.subheader("📊 Key Performance Indicators")
    display_kpi_cards(metrics, resources_df, patients_df)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Urgency Distribution")
        fig = plot_urgency_distribution(patients_df)
        st.pyplot(fig); plt.close(fig)
    with col2:
        st.subheader("Resource Availability")
        fig = plot_resource_bar(resources_df)
        st.pyplot(fig); plt.close(fig)

    st.markdown("---")
    st.subheader("🔑 Key RDMU Concepts Used")
    cols = st.columns(4)
    concepts = [
        ("Uncertainty", "Patient arrivals and severity are stochastic — modelled with Poisson and Normal distributions."),
        ("State–Action–Reward", "Q-learning defines State (urgency, wait, capacity, fairness), Action (4 allocation decisions), and Reward (clinical outcome signal)."),
        ("Reinforcement Learning", "A tabular Q-learning agent learns optimal allocation policies through simulated hospital experience."),
        ("Constraint Optimisation", "Hard resource limits (beds, doctors, nurses, equipment) and emergency buffers ensure physically feasible allocations."),
    ]
    for col, (title, desc) in zip(cols, concepts):
        with col:
            st.info(f"**{title}**\n\n{desc}")

    st.markdown("---")
    st.subheader("📚 Project Links")
    st.markdown(f"- **Google Colab Notebook:** `{COLAB_LINK}`")
    st.markdown(f"- **GitHub Repository:** `{GITHUB_LINK}`")
    st.markdown(f"- **Streamlit Deployed App:** `{STREAMLIT_LINK}`")
    st.caption("Replace placeholders above after deployment. See README for instructions.")


# ─── Page 2: Data Preview ─────────────────────────────────────────────────────
def page_data_preview():
    st.title("📁 Data Preview")
    patients_df = st.session_state.patients_df
    resources_df = st.session_state.resources_df

    if st.button("🔄 Regenerate Synthetic Hospital Data"):
        patients_df, resources_df = save_data("data")
        st.session_state.patients_df = patients_df
        st.session_state.resources_df = resources_df
        st.success("Data regenerated and saved to data/")

    st.subheader("Patients Dataset")
    st.markdown(f"Shape: **{patients_df.shape[0]} rows × {patients_df.shape[1]} columns**")
    st.dataframe(patients_df.head(10), use_container_width=True)
    st.markdown(f"**Columns:** {', '.join(patients_df.columns.tolist())}")

    st.subheader("Resources Dataset")
    st.dataframe(resources_df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Urgency Distribution")
        fig = plot_urgency_distribution(patients_df)
        st.pyplot(fig); plt.close(fig)
    with col2:
        st.subheader("Fairness Group Distribution")
        fig = plot_fairness_distribution(patients_df)
        st.pyplot(fig); plt.close(fig)

    st.subheader("Descriptive Statistics — Patients")
    st.dataframe(patients_df.describe().round(2), use_container_width=True)


# ─── Page 3: Resource Allocation Simulator ───────────────────────────────────
def page_simulator():
    st.title("⚙️ Resource Allocation Simulator")
    patients_df = st.session_state.patients_df
    resources_df = st.session_state.resources_df.copy()

    st.sidebar.markdown("### Simulator Controls")
    arrival_rate = st.sidebar.slider("Patient Arrival Rate (λ per step)", 0.5, 5.0, 1.5, 0.1)
    n_steps = st.sidebar.slider("Simulation Steps", 20, 200, 80, 10)
    bed_adj = st.sidebar.slider("Beds Available", 5, 50, 30, 1)
    doc_adj = st.sidebar.slider("Doctors Available", 3, 20, 10, 1)
    nurse_adj = st.sidebar.slider("Nurses Available", 5, 40, 20, 1)
    equip_adj = st.sidebar.slider("Equipment Available", 3, 30, 15, 1)

    resources_df.loc[resources_df["resource_name"] == "Beds", "available_count"] = bed_adj
    resources_df.loc[resources_df["resource_name"] == "Doctors", "available_count"] = doc_adj
    resources_df.loc[resources_df["resource_name"] == "Nurses", "available_count"] = nurse_adj
    resources_df.loc[resources_df["resource_name"] == "Equipment", "available_count"] = equip_adj

    if st.button("▶ Run Simulation"):
        with st.spinner("Running hospital simulation…"):
            sim_df = run_simulation(patients_df, resources_df, arrival_rate, n_steps)
            st.session_state.sim_df = sim_df

    sim_df = st.session_state.sim_df
    if sim_df is not None:
        st.subheader("Simulation Timeline")
        st.dataframe(sim_df.tail(20), use_container_width=True)

        fig = plot_allocation_timeline(sim_df)
        st.pyplot(fig); plt.close(fig)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Peak Queue", int(sim_df["queue_length"].max()))
        col2.metric("Total Treated", int(sim_df["cumulative_treated"].max()))
        col3.metric("Avg Bed Util.", f"{sim_df['bed_utilization'].mean():.1%}")
        col4.metric("Avg Wait (steps)", f"{sim_df['avg_wait_time'].mean():.1f}")

        st.success(
            f"Simulation complete. Under λ={arrival_rate} arrivals/step with "
            f"{bed_adj} beds and {doc_adj} doctors, the system processed "
            f"**{int(sim_df['cumulative_treated'].max())} patients** over {n_steps} steps."
        )
    else:
        st.info("Set parameters in the sidebar and click **Run Simulation** to begin.")


# ─── Page 4: RL Policy Engine ─────────────────────────────────────────────────
def page_rl_engine():
    st.title("🤖 RL Policy Engine — Q-Learning")
    patients_df = st.session_state.patients_df
    resources_df = st.session_state.resources_df

    st.markdown("""
    The **Q-learning agent** learns an optimal allocation policy through repeated simulated interactions
    with the hospital environment. Each patient episode updates the Q-table via the Bellman equation:

    > **Q(s,a) ← Q(s,a) + α [ r + γ · max Q(s') − Q(s,a) ]**
    """)

    with st.expander("ℹ️ State / Action / Reward Design"):
        st.markdown("""
        **State** = (urgency_bucket, wait_time_bucket, capacity_pressure_bucket, fairness_pressure_bucket)
        - Urgency: 5 levels · Wait: 4 buckets (<15, 15-30, 30-60, >60 min)
        - Capacity: 3 levels (Low/Medium/High) · Fairness: 2 levels (Normal/Underserved)
        - **Total states:** 5 × 4 × 3 × 2 = **120 states**

        **Actions** (4):  Allocate Full Care | Stabilize and Queue | Queue for Later | Transfer / Refer

        **Reward** signals:
        - +10×urgency for treating; +20 bonus for critical (urgency ≥ 4)
        - −30 for queuing a critical patient; −2 per minute past safe wait limit
        - +5 fairness bonus for treating underserved groups
        - −15 for over-allocating at high capacity
        """)

    col1, col2 = st.columns(2)
    with col1:
        alpha = st.slider("Learning Rate α", 0.01, 0.5, 0.10, 0.01)
        gamma = st.slider("Discount Factor γ", 0.5, 0.99, 0.90, 0.01)
    with col2:
        epsilon = st.slider("Exploration ε (initial)", 0.05, 0.9, 0.30, 0.05)
        n_eps = st.slider("Training Episodes", 100, 2000, 500, 100)

    if st.button("🚀 Train Q-Learning Agent"):
        agent = QLearningAgent(alpha=alpha, gamma=gamma, epsilon=epsilon, n_episodes=n_eps)
        progress = st.progress(0)
        with st.spinner("Training Q-learning agent…"):
            # Chunked training with progress
            chunk = max(1, n_eps // 20)
            rewards_all = []
            tmp_agent = QLearningAgent(alpha=alpha, gamma=gamma, epsilon=epsilon, n_episodes=chunk)
            for i in range(20):
                rh = tmp_agent.train(patients_df, resources_df)
                rewards_all.extend(rh)
                progress.progress((i+1)/20)
            agent.q_table = tmp_agent.q_table
            agent.reward_history = rewards_all

        st.session_state.rl_agent = agent
        st.session_state.rl_rewards = rewards_all

        # Allocate using trained policy
        rl_alloc = agent.allocate_patients(patients_df.head(200), resources_df)
        st.session_state.rl_alloc_df = rl_alloc
        st.success(f"Training complete — {n_eps} total episodes. Q-table shape: {agent.q_table.shape}")

    agent = st.session_state.rl_agent
    rewards = st.session_state.rl_rewards

    if agent is not None:
        st.subheader("📈 Reward Curve")
        fig = plot_reward_curve(rewards)
        st.pyplot(fig); plt.close(fig)

        st.subheader("🗺️ Learned Policy Heatmap")
        policy = agent.get_policy()
        fig = plot_policy_heatmap(policy)
        st.pyplot(fig); plt.close(fig)

        st.subheader("📋 Policy Table (first 30 rows)")
        policy_df = agent.get_policy_df()
        st.dataframe(policy_df.head(30), use_container_width=True)

        if st.session_state.rl_alloc_df is not None:
            st.subheader("📊 RL Allocation Action Distribution")
            fig = plot_action_distribution(st.session_state.rl_alloc_df, "RL Strategy — Action Distribution")
            st.pyplot(fig); plt.close(fig)

        st.subheader("🔍 Single-Patient Recommendation")
        c1, c2, c3, c4 = st.columns(4)
        urg = c1.slider("Urgency Level", 1, 5, 3)
        wt = c2.slider("Wait Time (min)", 0, 120, 25)
        cp = c3.slider("Capacity Pressure", 0.0, 1.0, 0.5, 0.05)
        fp = c4.slider("Fairness Pressure", 0.0, 1.0, 0.3, 0.05)
        rec, qvals = agent.recommend(urg, wt, cp, fp)
        st.success(f"**Recommended Action:** {rec}")
        st.markdown(f"Q-Values: `{np.round(qvals, 2).tolist()}`")
    else:
        st.info("Set hyperparameters above and click **Train Q-Learning Agent**.")


# ─── Page 5: Optimisation Baseline ───────────────────────────────────────────
def page_optimizer():
    st.title("📐 Constraint-Optimisation Baseline")
    patients_df = st.session_state.patients_df
    resources_df = st.session_state.resources_df

    st.markdown("""
    The **Constraint-Optimisation Baseline** uses deterministic feasibility rules — not a formal
    LP/IP solver — to allocate patients. It serves as the comparison benchmark for the RL agent.

    **Rules:**
    1. Compute a weighted priority score for each patient (urgency + fairness + wait time − resource burden).
    2. Sort patients by priority score (descending).
    3. Allocate only if beds, doctors, nurses, and equipment are available.
    4. Reserve emergency buffer for critical patients (urgency ≥ 4).
    5. Track treated, queued, and transferred patients.
    """)

    col1, col2, col3 = st.columns(3)
    fw = col1.slider("Fairness Weight", 0.05, 0.90, 0.30, 0.05)
    uw = col2.slider("Urgency Weight", 0.10, 0.80, 0.50, 0.05)
    ew = col3.slider("Efficiency Weight", 0.05, 0.60, 0.20, 0.05)

    n_patients_opt = st.slider("Number of Patients to Allocate", 50, 500, 200, 50)

    if st.button("▶ Run Optimisation Baseline"):
        opt = ConstraintOptimizer(fw, uw, ew)
        with st.spinner("Running constraint-optimisation…"):
            alloc_df, res_summary = opt.allocate(patients_df.head(n_patients_opt), resources_df)
        st.session_state.opt_alloc_df = alloc_df
        st.session_state.opt_res_summary = res_summary
        st.session_state.opt_resources_df = resources_df
        st.success("Optimisation complete.")

    alloc_df = st.session_state.opt_alloc_df
    res_summary = st.session_state.opt_res_summary

    if alloc_df is not None:
        st.subheader("Feasible Allocation Table")
        st.dataframe(alloc_df, use_container_width=True)

        st.subheader("Resource Constraint Summary")
        col1, col2 = st.columns(2)
        with col1:
            used_df = pd.DataFrame({
                "Resource": list(res_summary["used"].keys()),
                "Used": list(res_summary["used"].values()),
                "Available": list(res_summary["available"].values()),
                "Utilisation": list(res_summary["utilization"].values()),
            })
            st.dataframe(used_df, use_container_width=True)
        with col2:
            st.metric("Capacity Violations (Critical Not Treated)", res_summary["capacity_violations"])
            m = compute_metrics(alloc_df, resources_df=resources_df)
            st.metric("Treated (Concurrent Slots)", m.get("treated_count", "—"))
            st.metric("Queued / Stabilised", m.get("queued_count", 0) + m.get("stabilized_count", 0))
            st.metric("Transferred", m.get("transferred_count", "—"))

        fig = plot_action_distribution(alloc_df, "Optimisation Baseline — Action Distribution")
        st.pyplot(fig); plt.close(fig)
    else:
        st.info("Click **Run Optimisation Baseline** above.")


# ─── Page 6: Policy Comparison ────────────────────────────────────────────────
def page_policy_comparison():
    st.title("🔬 Policy Comparison: RL vs Optimisation Baseline")

    rl_alloc = st.session_state.rl_alloc_df
    opt_alloc = st.session_state.opt_alloc_df
    resources_df = st.session_state.resources_df

    if rl_alloc is None or opt_alloc is None:
        st.warning("Please run both the **RL Policy Engine** (Page 4) and the **Optimisation Baseline** (Page 5) first.")
        return

    rl_m = compute_metrics(rl_alloc, resources_df=resources_df)
    opt_m = compute_metrics(opt_alloc, resources_df=resources_df)

    st.subheader("📊 Side-by-Side Metrics")
    compare_keys = [
        ("Total Patients", "total_patients"),
        ("Treated Count", "treated_count"),
        ("Queued Count", "queued_count"),
        ("Transferred Count", "transferred_count"),
        ("Avg Wait Time", "avg_wait_time"),
        ("Fairness Index", "fairness_index"),
        ("Urgency Handling Score", "urgency_handling_score"),
        ("Patient Outcome Score", "patient_outcome_score"),
        ("Resource Utilisation", "resource_utilization"),
        ("Capacity Violations", "capacity_violation_count"),
    ]
    comp_df = pd.DataFrame([
        {"Metric": label, "RL Strategy": rl_m.get(key, "—"), "Optimisation Baseline": opt_m.get(key, "—")}
        for label, key in compare_keys
    ])
    st.dataframe(comp_df, use_container_width=True)

    st.subheader("📈 Comparison Bar Chart")
    fig = plot_policy_comparison_bar(rl_m, opt_m)
    st.pyplot(fig); plt.close(fig)

    st.subheader("💼 Business Interpretation")
    fi_rl = rl_m.get("fairness_index", 0)
    fi_opt = opt_m.get("fairness_index", 0)
    uh_rl = rl_m.get("urgency_handling_score", 0)
    uh_opt = opt_m.get("urgency_handling_score", 0)

    st.markdown(f"""
    | Dimension | RL Strategy | Optimisation Baseline | Recommendation |
    |---|---|---|---|
    | Fairness | {fi_rl:.3f} | {fi_opt:.3f} | {'RL' if fi_rl > fi_opt else 'Optimiser'} preferred |
    | Urgency Handling | {uh_rl:.3f} | {uh_opt:.3f} | {'RL' if uh_rl > uh_opt else 'Optimiser'} preferred |
    | Throughput | {rl_m.get('treated_count',0)} treated | {opt_m.get('treated_count',0)} treated | {'RL' if rl_m.get('treated_count',0) > opt_m.get('treated_count',0) else 'Optimiser'} preferred |
    """)

    st.info(
        "**Key insight:** The RL agent learns from experience and adapts to resource pressure. "
        "The Optimisation Baseline is deterministic and always feasible but may miss fairness dynamics. "
        "In practice, both would be used together — the optimiser ensures feasibility while RL improves long-run outcomes."
    )


# ─── Page 7: Sensitivity Analysis ────────────────────────────────────────────
def page_sensitivity():
    st.title("🎚️ Sensitivity Analysis — Fairness vs Efficiency")
    patients_df = st.session_state.patients_df
    resources_df = st.session_state.resources_df

    st.markdown("""
    Sensitivity Analysis reveals how outcomes change as we adjust the **fairness weight**
    from 0.05 to 0.90 (with efficiency weight = 1 − fairness weight). This exposes the
    inherent trade-off between treating the most patients efficiently and treating all groups equitably.
    """)

    n_points = st.slider("Number of Analysis Points", 5, 40, 20, 5)

    if st.button("🔍 Generate Sensitivity Analysis"):
        with st.spinner("Running sensitivity sweep…"):
            sens_df = run_sensitivity_analysis(patients_df, resources_df, n_points)
        st.session_state.sens_df = sens_df
        sens_df.to_csv("data/sensitivity_results.csv", index=False)
        st.success("Sensitivity analysis complete. Results saved to data/sensitivity_results.csv")

    sens_df = st.session_state.sens_df
    if sens_df is not None:
        st.subheader("Sensitivity Results Table")
        st.dataframe(sens_df.round(3), use_container_width=True)

        st.subheader("Fairness vs Efficiency Trade-off Charts")
        fig = plot_sensitivity(sens_df)
        st.pyplot(fig); plt.close(fig)

        st.subheader("Outcome Score vs Fairness Weight")
        fig2, ax = plt.subplots(figsize=(10, 4))
        ax.plot(sens_df["fairness_weight"], sens_df["patient_outcome_score"],
                color="#2563EB", linewidth=2.2, marker="o", markersize=5)
        ax.set_xlabel("Fairness Weight"); ax.set_ylabel("Patient Outcome Score")
        ax.set_title("Patient Outcome Score vs Fairness Weight", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig2); plt.close(fig2)

        # Download
        csv = sens_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download Sensitivity Results CSV", csv,
                           "sensitivity_results.csv", "text/csv")
    else:
        st.info("Click **Generate Sensitivity Analysis** above.")


# ─── Page 8: Metrics and Performance ─────────────────────────────────────────
def page_metrics():
    st.title("📊 Metrics and Performance")

    opt_alloc = st.session_state.opt_alloc_df
    rl_alloc = st.session_state.rl_alloc_df
    sim_df = st.session_state.sim_df
    resources_df = st.session_state.resources_df

    alloc_df = opt_alloc if opt_alloc is not None else rl_alloc
    if alloc_df is None:
        st.warning("Run the Optimisation Baseline or RL Policy Engine first to see metrics.")
        return

    metrics = compute_metrics(alloc_df, sim_df, resources_df=resources_df)

    st.subheader("📋 Full Metrics Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Patients", metrics.get("total_patients", "—"))
    col2.metric("Treated", metrics.get("treated_count", "—"))
    col3.metric("Queued", metrics.get("queued_count", "—"))
    col4.metric("Transferred", metrics.get("transferred_count", "—"))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Wait Time (min)", metrics.get("avg_wait_time", "—"))
    col2.metric("Fairness Index (Jain's)", metrics.get("fairness_index", "—"))
    col3.metric("Urgency Handling Score", metrics.get("urgency_handling_score", "—"))
    col4.metric("Patient Outcome Score", metrics.get("patient_outcome_score", "—"))

    col1, col2 = st.columns(2)
    util = metrics.get("resource_utilization", 0)
    col1.metric("Resource Utilisation (Doctors+Beds)", f"{util:.1%}" if isinstance(util, float) else util)
    col2.metric("Critical Patients Not Treated", metrics.get("capacity_violation_count", "—"))

    st.markdown("---")
    st.subheader("🌡️ Performance Metrics Heatmap")
    fig = plot_metrics_heatmap(metrics)
    st.pyplot(fig); plt.close(fig)

    st.subheader("📥 Export Allocation Report")
    alloc_df["outcome_label"] = alloc_df["status"]
    csv = alloc_df.to_csv(index=False).encode("utf-8")
    alloc_df.to_csv("outputs/allocation_report.csv", index=False)
    st.download_button("⬇ Download Allocation Report CSV", csv,
                       "allocation_report.csv", "text/csv")
    st.caption("Report also saved to outputs/allocation_report.csv")


# ─── Page 9: RDMU Concept Mapping ────────────────────────────────────────────
def page_concept_mapping():
    st.title("📚 RDMU Concept Mapping")
    st.markdown("""
    This page documents how the key **Reasoning and Decision Making under Uncertainty (RDMU)**
    concepts are applied in MediFlow AI. The minimum required 4 concepts are highlighted below,
    along with 6 additional concepts for completeness.
    """)

    concept_df = display_concept_table()
    st.dataframe(concept_df, use_container_width=True, height=420)

    st.markdown("---")
    st.subheader("🎯 Core 4 RDMU Concepts (Minimum Required)")
    core = {
        "1. Uncertainty": (
            "Patient arrivals, severity levels, and treatment outcomes are inherently uncertain. "
            "The system models this using Poisson arrival processes and stochastic simulation."
        ),
        "2. State / Action / Reward (MDP Framework)": (
            "The allocation problem is formulated as a Markov Decision Process. State encodes "
            "urgency, wait time, capacity pressure, and fairness gap. Actions are 4 allocation "
            "decisions. Reward guides the agent toward efficient and fair outcomes."
        ),
        "3. Reinforcement Learning (Q-Learning)": (
            "A tabular Q-learning agent updates its policy via Bellman equation: "
            "Q(s,a) ← Q(s,a) + α[r + γ·max Q(s') − Q(s,a)]. The agent explores using "
            "ε-greedy selection and converges to a near-optimal policy over 500+ episodes."
        ),
        "4. Constraint-Based Decision Making": (
            "Hard resource constraints (beds, doctors, nurses, equipment) and emergency buffers "
            "are enforced during allocation. The optimiser never violates available capacity, "
            "ensuring physically feasible decisions at all times."
        ),
    }
    for concept, explanation in core.items():
        with st.expander(f"✅ {concept}"):
            st.markdown(explanation)

    st.markdown("---")
    st.subheader("Additional RDMU Concepts in This Project")
    extra = {
        "5. Fairness": "Jain's Fairness Index measures equal treatment across Group A / B / C. The optimizer's fairness weight directly controls the fairness–efficiency trade-off.",
        "6. Sensitivity Analysis": "The Sensitivity Analysis page sweeps fairness weight from 0.05 to 0.90 and records how key metrics respond — a standard uncertainty analysis technique.",
        "7. Policy Comparison": "RL-driven vs. constraint-driven strategies are compared on 10 metrics, demonstrating the difference between adaptive and rule-based decision-making.",
        "8. Human-in-the-Loop": "The simulator allows clinicians to manually override resource settings and re-run allocation, maintaining human agency over AI recommendations.",
    }
    for k, v in extra.items():
        st.markdown(f"**{k}:** {v}")


# ─── Page 10: Limitations and Future Scope ────────────────────────────────────
def page_limitations():
    st.title("⚠️ Limitations and Future Scope")

    st.subheader("Current Limitations")
    limitations = [
        "**Synthetic Data Only:** The system uses artificially generated patient records. Real hospital data would differ significantly in distribution and complexity.",
        "**Not Production-Ready:** This prototype has not undergone clinical validation, security hardening, or regulatory review.",
        "**Tabular Q-Learning:** The Q-table approach does not scale well to very large or continuous state spaces; deep RL (DQN, PPO) would be needed for real-world deployment.",
        "**Deterministic Optimiser:** The baseline uses heuristic feasibility rules, not a formal LP/IP solver (e.g., PuLP or scipy.optimize), due to the exam environment.",
        "**No Real-Time Data:** The system does not integrate with live hospital information systems (HIS, EHR, bed management).",
        "**Privacy & Security:** No patient data privacy controls, authentication, or audit logging are implemented.",
        "**Fairness Definition:** Only Jain's Fairness Index is used; other fairness criteria (equity, equal opportunity, disparate impact) are not measured.",
        "**No Clinical Validation:** Reward function weightings are illustrative; clinical experts would need to calibrate these values.",
    ]
    for lim in limitations:
        st.markdown(f"- {lim}")

    st.markdown("---")
    st.subheader("Future Scope")
    future = [
        "**Real Hospital Integration:** Connect to live HIS/EHR APIs for real-time patient and resource data.",
        "**Deep Reinforcement Learning:** Replace tabular Q-learning with DQN or Proximal Policy Optimisation (PPO) for continuous state spaces.",
        "**Formal LP/IP Solver:** Integrate PuLP or OR-Tools for mathematically optimal constraint satisfaction.",
        "**Multi-Objective Optimisation:** Pareto-front analysis balancing efficiency, fairness, cost, and patient safety simultaneously.",
        "**Explainable AI (XAI):** SHAP values or LIME to explain individual allocation decisions to clinicians.",
        "**Dashboard Authentication:** Role-based access control so only authorised staff can view or override allocations.",
        "**Audit Logging:** Full decision trail for regulatory compliance and quality assurance.",
        "**Ethics Review & Fairness Testing:** External audit of demographic fairness, disparate impact testing, and bias detection.",
        "**Federated Learning:** Train across multiple hospitals without sharing patient data.",
    ]
    for item in future:
        st.markdown(f"- {item}")

    st.markdown("---")
    st.subheader("Academic Disclaimer")
    st.info(
        "This software prototype was developed as a final examination submission for "
        "MAIB DSC 103 — Reasoning and Decision Making under Uncertainty, "
        "SP Jain School of Global Management, MAIB September 2025 cohort. "
        "It is intended for educational demonstration only and is not suitable for "
        "clinical or operational use without substantial further development, validation, "
        "and regulatory approval."
    )
    _show_links()


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════
if page == PAGES[0]:
    page_executive_dashboard()
elif page == PAGES[1]:
    page_data_preview()
elif page == PAGES[2]:
    page_simulator()
elif page == PAGES[3]:
    page_rl_engine()
elif page == PAGES[4]:
    page_optimizer()
elif page == PAGES[5]:
    page_policy_comparison()
elif page == PAGES[6]:
    page_sensitivity()
elif page == PAGES[7]:
    page_metrics()
elif page == PAGES[8]:
    page_concept_mapping()
elif page == PAGES[9]:
    page_limitations()
