"""
visualizations.py
All Matplotlib chart functions. No seaborn used.
Returns Figure objects for use with st.pyplot().
Made by: Krishna Mathur | AS25DXB018 | MAIB September
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


def _get_cmap(name, n=None):
    """Compat wrapper: works on matplotlib ≥3.5, uses new API ≥3.7."""
    cmap = matplotlib.colormaps.get_cmap(name)
    return cmap.resampled(n) if n is not None else cmap

_COLORS = ["#2563EB", "#16A34A", "#D97706", "#DC2626", "#7C3AED", "#0891B2"]
_URGENCY_COLORS = ["#22C55E", "#84CC16", "#EAB308", "#F97316", "#EF4444"]


def _fig(w=10, h=5):
    return plt.subplots(figsize=(w, h))


# -----------------------------------------------------------------------
def plot_urgency_distribution(patients_df):
    """Pie chart of patient urgency levels."""
    fig, ax = _fig(7, 5)
    counts = patients_df["urgency_level"].value_counts().sort_index()
    labels = [f"Level {i}" for i in counts.index]
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=labels,
        colors=_URGENCY_COLORS[:len(counts)],
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5}
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax.set_title("Urgency Level Distribution", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig


def plot_fairness_distribution(df):
    """Pie chart of fairness group distribution."""
    fig, ax = _fig(7, 5)
    if "fairness_group" not in df.columns:
        ax.text(0.5, 0.5, "No fairness_group column", ha="center")
        return fig
    counts = df["fairness_group"].value_counts()
    ax.pie(counts.values, labels=counts.index,
           colors=_COLORS[:len(counts)],
           autopct="%1.1f%%", startangle=90,
           wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    ax.set_title("Fairness Group Distribution", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig


def plot_resource_bar(resources_df):
    """Bar chart of available vs max capacity for each resource."""
    fig, ax = _fig(9, 5)
    x = np.arange(len(resources_df))
    w = 0.35
    b1 = ax.bar(x - w/2, resources_df["available_count"], w, label="Available", color="#2563EB", alpha=0.88)
    b2 = ax.bar(x + w/2, resources_df["max_capacity"], w, label="Max Capacity", color="#94A3B8", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(resources_df["resource_name"], fontsize=10)
    ax.set_ylabel("Count")
    ax.set_title("Hospital Resource Availability", fontsize=13, fontweight="bold")
    ax.legend()
    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(int(bar.get_height())), ha="center", fontsize=9, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def plot_reward_curve(reward_history):
    """RL training reward curve with moving average."""
    fig, ax = _fig(11, 5)
    eps = range(len(reward_history))
    ax.plot(eps, reward_history, color="#94A3B8", linewidth=0.9, alpha=0.6, label="Episode Reward")
    window = max(1, len(reward_history) // 20)
    ma = pd.Series(reward_history).rolling(window=window, min_periods=1).mean()
    ax.plot(eps, ma, color="#DC2626", linewidth=2.2, label=f"Moving Avg (w={window})")
    ax.set_xlabel("Training Episode", fontsize=10)
    ax.set_ylabel("Average Reward", fontsize=10)
    ax.set_title("Q-Learning Training Reward Curve", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    return fig


def plot_policy_heatmap(policy_dict):
    """
    2D heatmap: urgency × wait-time, colour = action index.
    Slice: capacity_pressure=Low, fairness=Not Underserved.
    """
    n_u, n_w = 5, 4
    action_map = {
        "Allocate Full Care": 0,
        "Stabilize and Queue": 1,
        "Queue for Later": 2,
        "Transfer / Refer": 3,
    }
    short = ["Full Care", "Stabilise", "Queue", "Transfer"]
    matrix = np.zeros((n_u, n_w))
    for (u, w, c, f), action in policy_dict.items():
        if c == 0 and f == 0:
            matrix[u, w] = action_map.get(action, 0)

    cmap = _get_cmap("RdYlGn_r", 4)
    fig, ax = _fig(11, 6)
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=3)
    ax.set_xticks(range(n_w))
    ax.set_xticklabels(["<15 min", "15-30 min", "30-60 min", ">60 min"], fontsize=9)
    ax.set_yticks(range(n_u))
    ax.set_yticklabels([f"Urgency {i+1}" for i in range(n_u)], fontsize=9)
    ax.set_xlabel("Patient Wait Time Bucket")
    ax.set_ylabel("Patient Urgency Level")
    ax.set_title("Learned RL Policy  (Low Capacity, No Fairness Pressure)",
                 fontsize=12, fontweight="bold")
    for i in range(n_u):
        for j in range(n_w):
            ax.text(j, i, short[int(matrix[i, j])],
                    ha="center", va="center", color="white", fontweight="bold", fontsize=8)
    patches = [mpatches.Patch(color=cmap(k/3), label=a)
               for k, a in enumerate(["Allocate Full Care", "Stabilize & Queue",
                                       "Queue for Later", "Transfer / Refer"])]
    ax.legend(handles=patches, loc="upper right", bbox_to_anchor=(1.38, 1), fontsize=8)
    plt.tight_layout()
    return fig


def plot_allocation_timeline(sim_df):
    """Queue length and utilisation curves over simulation time."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))

    axes[0].plot(sim_df["step"], sim_df["queue_length"], color="#DC2626", linewidth=2, label="Queue Length")
    axes[0].fill_between(sim_df["step"], sim_df["queue_length"], alpha=0.15, color="#DC2626")
    axes[0].set_ylabel("Patients in Queue")
    axes[0].set_title("Patient Queue Length Over Simulation Time", fontsize=12, fontweight="bold")
    axes[0].legend(); axes[0].grid(True, alpha=0.25)

    axes[1].plot(sim_df["step"], sim_df["bed_utilization"], color="#2563EB", linewidth=2, label="Bed Util.")
    axes[1].plot(sim_df["step"], sim_df["doctor_utilization"], color="#16A34A", linewidth=2, label="Doctor Util.")
    axes[1].plot(sim_df["step"], sim_df["nurse_utilization"], color="#D97706", linewidth=2, label="Nurse Util.")
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("Utilisation Rate (0–1)")
    axes[1].set_title("Resource Utilisation Over Time", fontsize=12, fontweight="bold")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(); axes[1].grid(True, alpha=0.25)

    plt.tight_layout()
    return fig


def plot_policy_comparison_bar(rl_metrics, opt_metrics):
    """Side-by-side bar chart comparing RL vs Optimizer on key metrics."""
    keys = ["resource_utilization", "fairness_index", "urgency_handling_score", "patient_outcome_score"]
    labels = ["Resource Utilisation", "Fairness Index", "Urgency Handling", "Outcome Score (norm)"]

    def _norm(v, k):
        if k == "patient_outcome_score":
            return min(v / 50.0, 1.0)
        return float(v)

    rl_vals = [_norm(rl_metrics.get(k, 0), k) for k in keys]
    opt_vals = [_norm(opt_metrics.get(k, 0), k) for k in keys]

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = _fig(12, 6)
    b1 = ax.bar(x - w/2, rl_vals, w, label="RL Strategy", color="#2563EB", alpha=0.88)
    b2 = ax.bar(x + w/2, opt_vals, w, label="Optimisation Baseline", color="#16A34A", alpha=0.88)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=12, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score (0 – 1)")
    ax.set_title("Policy Comparison: RL Strategy vs Constraint-Optimisation Baseline",
                 fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    return fig


def plot_sensitivity(sensitivity_df):
    """Fairness weight vs Fairness Index scatter + Wait Time vs Utilisation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sc1 = axes[0].scatter(
        sensitivity_df["fairness_weight"],
        sensitivity_df["fairness_index"],
        c=sensitivity_df["efficiency_weight"], cmap="viridis", s=70, alpha=0.8, edgecolors="none"
    )
    axes[0].set_xlabel("Fairness Weight"); axes[0].set_ylabel("Fairness Index")
    axes[0].set_title("Fairness Weight vs Fairness Index", fontsize=11, fontweight="bold")
    axes[0].grid(True, alpha=0.25)
    plt.colorbar(sc1, ax=axes[0], label="Efficiency Weight")

    sc2 = axes[1].scatter(
        sensitivity_df["avg_wait_time"],
        sensitivity_df["resource_utilization"],
        c=sensitivity_df["fairness_weight"], cmap="RdYlGn", s=70, alpha=0.8, edgecolors="none"
    )
    axes[1].set_xlabel("Average Wait Time (steps)"); axes[1].set_ylabel("Resource Utilisation")
    axes[1].set_title("Wait Time vs Resource Utilisation", fontsize=11, fontweight="bold")
    axes[1].grid(True, alpha=0.25)
    plt.colorbar(sc2, ax=axes[1], label="Fairness Weight")

    plt.tight_layout()
    return fig


def plot_metrics_heatmap(metrics):
    """3×2 heatmap of key performance scores using imshow."""
    names = [
        ["Treatment Rate", "Avg Wait (inv.)", "Resource Util."],
        ["Fairness Index", "Urgency Score", "Outcome Score"],
    ]
    raw_vals = [
        [
            metrics.get("treated_count", 0) / max(metrics.get("total_patients", 1), 1),
            1.0 - min(metrics.get("avg_wait_time", 0) / 120.0, 1.0),
            metrics.get("resource_utilization", 0),
        ],
        [
            metrics.get("fairness_index", 0),
            metrics.get("urgency_handling_score", 0),
            min(metrics.get("patient_outcome_score", 0) / 50.0, 1.0),
        ],
    ]
    data = np.array(raw_vals)

    fig, ax = _fig(10, 4)
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(["Metric A", "Metric B", "Metric C"], fontsize=9)
    ax.set_yticks(range(2)); ax.set_yticklabels(["Treatment", "Quality"], fontsize=9)
    for i in range(2):
        for j in range(3):
            ax.text(j, i, f"{data[i,j]:.2f}\n{names[i][j]}",
                    ha="center", va="center", color="black", fontsize=9, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Score (0=Low  1=High)", fraction=0.03)
    ax.set_title("Performance Metrics Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_action_distribution(allocation_df, title="Action Distribution"):
    """Bar chart of how many patients received each action."""
    counts = allocation_df["action"].value_counts()
    fig, ax = _fig(9, 4)
    bars = ax.bar(counts.index, counts.values,
                  color=_COLORS[:len(counts)], alpha=0.88, edgecolor="white")
    ax.set_xlabel("Allocation Action"); ax.set_ylabel("Patient Count")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(int(bar.get_height())), ha="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    return fig
