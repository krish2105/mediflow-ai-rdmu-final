"""
create_notebook.py
Generates MediFlowAI_Notebook.ipynb programmatically using nbformat.
Run: python create_notebook.py
Made by: Krishna Mathur | AS25DXB018 | MAIB September
"""
import nbformat as nbf
from pathlib import Path

NB_PATH = "MediFlowAI_Notebook.ipynb"

COLAB_LINK = "[PASTE_ACCESSIBLE_COLAB_LINK_HERE]"
GITHUB_LINK = "[PASTE_GITHUB_REPO_LINK_HERE]"
STREAMLIT_LINK = "[PASTE_STREAMLIT_APP_LINK_HERE]"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src)


cells = []

# ══════════════════════════════════════════════════════════════════════════
# CELL 1 — Title & Student Details
# ══════════════════════════════════════════════════════════════════════════
cells.append(md(f"""\
# MediFlow AI
## Dynamic Hospital Emergency Resource Allocation using Reinforcement Learning

---

| Field | Value |
|---|---|
| **Student Name** | Krishna Mathur |
| **Student ID** | AS25DXB018 |
| **Program** | MAIB September |
| **Course** | Reasoning and Decision Making under Uncertainty — MAIB DSC 103 |
| **Topic** | Topic 1 — Dynamic Resource Allocation using Reinforcement Learning |
| **Google Colab** | `{COLAB_LINK}` |
| **GitHub** | `{GITHUB_LINK}` |
| **Streamlit App** | `{STREAMLIT_LINK}` |

> **How to share on Google Colab:**
> 1. Upload this file to [colab.research.google.com](https://colab.research.google.com)
> 2. Runtime → Run all
> 3. Share → Anyone with the link → Viewer → Copy link
> 4. Replace `{COLAB_LINK}` in README, app.py, and create_ppt.py
"""))

# ══════════════════════════════════════════════════════════════════════════
# CELL 2 — Requirement Understanding
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## 1. Requirement Understanding

**Exam requirement:** Build a decision-making software prototype demonstrating
at least **4 key RDMU concepts**.

**What this prototype does:**
- Simulates a hospital emergency department with stochastic patient arrivals
- Trains a Q-learning agent to learn optimal resource allocation policies
- Implements a constraint-optimisation baseline for policy comparison
- Visualises fairness vs efficiency trade-offs via sensitivity analysis
- Provides an interactive Streamlit dashboard (10 pages) for clinical exploration

**Minimum 4 RDMU Concepts demonstrated:**
1. **Uncertainty** — Stochastic patient arrivals (Poisson), random severity (Normal)
2. **State / Action / Reward (MDP)** — Full Markov Decision Process formulation
3. **Reinforcement Learning (Q-Learning)** — Tabular Q-learning with Bellman update
4. **Constraint-Based Decision Making** — Hard resource limits + emergency buffers
"""))

# ══════════════════════════════════════════════════════════════════════════
# CELL 3 — Application Flow Chart
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## 2. Application Flow Chart

```
Patient Data (500 synthetic records, seed=42)
         │
         ▼
Resource Capacity (Beds=30, Doctors=10, Nurses=20, Equipment=15)
         │
         ├───────────────────────────────────┐
         ▼                                   ▼
┌────────────────────┐          ┌─────────────────────────────┐
│  RL Policy Engine  │          │  Constraint-Optimisation    │
│  (Q-Learning)      │          │  Baseline (Heuristic Rules) │
│  State → Q-table   │          │  Priority Score + Feasible  │
│  → Best Action     │          │  Allocation                 │
└────────────────────┘          └─────────────────────────────┘
         │                                   │
         └──────────────┬────────────────────┘
                        ▼
               Policy Comparison
      (Fairness Index, Utilisation, Urgency Score)
                        │
                        ▼
              Dashboard Metrics & Charts
                        │
                        ▼
            Human Review / Override (Streamlit)
```
"""))

# ══════════════════════════════════════════════════════════════════════════
# CELL 4 — Prerequisites
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## 3. Prerequisites

- Python 3.8+
- Google Colab or local Jupyter environment
- Libraries: `pandas numpy matplotlib scikit-learn streamlit nbformat python-pptx`
"""))

# ══════════════════════════════════════════════════════════════════════════
# CELL 5 — Install
# ══════════════════════════════════════════════════════════════════════════
cells.append(code("""\
# Install all required libraries (safe to re-run)
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install",
    "pandas", "numpy", "matplotlib", "scikit-learn",
    "streamlit", "nbformat", "python-pptx", "-q"])
print("All libraries ready.")
"""))

# ══════════════════════════════════════════════════════════════════════════
# CELL 6 — Imports
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("## 4. Import Libraries"))

cells.append(code("""\
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


def _get_cmap(name, n=None):
    cmap = matplotlib.colormaps.get_cmap(name)
    return cmap.resampled(n) if n is not None else cmap

np.random.seed(42)
print("Libraries imported.")
print(f"NumPy {np.__version__}  |  Pandas {pd.__version__}")
"""))

# ══════════════════════════════════════════════════════════════════════════
# CELL 7 — Data Generation (markdown)
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## 5. Generate Synthetic Hospital Data

### Why synthetic data?
Real patient data requires HIPAA/GDPR compliance, ethical approval, and
de-identification. Synthetic data allows full reproducibility in an exam context.

### Data Schema
| File | Rows | Key columns |
|---|---|---|
| `patients.csv` | 500 | patient_id, urgency_level (1-5), severity_score, fairness_group (A/B/C), required_bed/doctor/nurse/equipment |
| `resources.csv` | 4 | resource_name, available_count, emergency_buffer, unit_cost, max_capacity |

**RDMU Concept 1 — Uncertainty:** Arrivals follow `Poisson(λ)`;
severity drawn from `Normal(urgency×15, 5)` — stochastic and unpredictable.
"""))

# ══════════════════════════════════════════════════════════════════════════
# CELL 8 — Data Generation (code)
# ══════════════════════════════════════════════════════════════════════════
cells.append(code("""\
def generate_patients(n=500, seed=42):
    np.random.seed(seed)
    urgency = np.random.choice(
        [1, 2, 3, 4, 5], size=n,
        p=[0.10, 0.20, 0.30, 0.25, 0.15]
    )
    arrival = np.cumsum(np.random.exponential(5, n)).astype(int)
    severity = np.clip(
        urgency * 15 + np.random.normal(0, 5, n), 10, 100
    ).astype(int)
    max_wait = np.array([
        max(10, 120 - u * 20 + np.random.randint(-5, 5))
        for u in urgency
    ])
    return pd.DataFrame({
        "patient_id":            [f"P{str(i+1).zfill(4)}" for i in range(n)],
        "arrival_time":          arrival,
        "urgency_level":         urgency,
        "severity_score":        severity,
        "age_group":             np.random.choice(
                                     ["Child", "Adult", "Senior"], n,
                                     p=[0.15, 0.55, 0.30]),
        "required_bed":          np.random.choice([1, 0], n, p=[0.85, 0.15]),
        "required_doctor":       np.ones(n, dtype=int),
        "required_nurse":        np.random.choice([1, 2], n, p=[0.60, 0.40]),
        "required_equipment":    np.random.choice([0, 1, 2], n, p=[0.40, 0.40, 0.20]),
        "max_safe_wait_time":    max_wait,
        "fairness_group":        np.random.choice(
                                     ["Group_A", "Group_B", "Group_C"], n,
                                     p=[0.40, 0.35, 0.25]),
        "expected_treatment_time": np.random.randint(20, 120, n),
    })


def generate_resources():
    return pd.DataFrame({
        "resource_name":    ["Beds", "Doctors", "Nurses", "Equipment"],
        "available_count":  [30, 10, 20, 15],
        "emergency_buffer": [5,  2,  3,  2],
        "unit_cost":        [500, 1200, 400, 800],
        "max_capacity":     [35, 12, 25, 18],
    })


Path("data").mkdir(exist_ok=True)
patients_df  = generate_patients(500)
resources_df = generate_resources()
patients_df.to_csv("data/patients.csv",  index=False)
resources_df.to_csv("data/resources.csv", index=False)

print(f"patients.csv  : {patients_df.shape}  rows × cols")
print(f"resources.csv : {resources_df.shape}  rows × cols")
print("\\nUrgency distribution:")
print(patients_df["urgency_level"].value_counts().sort_index())
print("\\nFairness groups:")
print(patients_df["fairness_group"].value_counts())
patients_df.head()
"""))

# ══════════════════════════════════════════════════════════════════════════
# CELL 9 — Data Visualisation
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("## 6. Visualise the Data"))

cells.append(code("""\
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Urgency pie
counts = patients_df["urgency_level"].value_counts().sort_index()
axes[0].pie(
    counts.values,
    labels=[f"Level {i}" for i in counts.index],
    colors=["#22C55E", "#84CC16", "#EAB308", "#F97316", "#EF4444"],
    autopct="%1.1f%%", startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
)
axes[0].set_title("Urgency Level Distribution", fontweight="bold")

# Fairness pie
fg_counts = patients_df["fairness_group"].value_counts()
axes[1].pie(
    fg_counts.values, labels=fg_counts.index,
    colors=["#2563EB", "#16A34A", "#D97706"],
    autopct="%1.1f%%", startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
)
axes[1].set_title("Fairness Group Distribution", fontweight="bold")

# Resource bar
x = range(len(resources_df))
axes[2].bar(
    [xi - 0.18 for xi in x], resources_df["available_count"],
    0.35, label="Available", color="#2563EB", alpha=0.88,
)
axes[2].bar(
    [xi + 0.18 for xi in x], resources_df["max_capacity"],
    0.35, label="Max Capacity", color="#94A3B8", alpha=0.7,
)
axes[2].set_xticks(list(x))
axes[2].set_xticklabels(resources_df["resource_name"])
axes[2].set_title("Resource Availability vs Capacity", fontweight="bold")
axes[2].legend(); axes[2].grid(True, axis="y", alpha=0.3)

plt.suptitle("MediFlow AI — Hospital Data Overview", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("data_overview.png", dpi=120, bbox_inches="tight")
plt.show()
print("Chart saved to data_overview.png")
"""))

# ══════════════════════════════════════════════════════════════════════════
# CELL 10 — RL Theory (markdown)
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## 7. Reinforcement Learning — Q-Learning Agent

**RDMU Concept 2: State / Action / Reward (MDP)**
**RDMU Concept 3: Reinforcement Learning**

### Bellman Update Equation
> **Q(s, a) ← Q(s, a) + α [ r + γ · max_a' Q(s', a') − Q(s, a) ]**

| Symbol | Meaning | Value |
|---|---|---|
| α (alpha) | Learning rate | 0.10 |
| γ (gamma) | Discount factor | 0.90 |
| ε (epsilon) | Initial exploration rate | 0.30 → 0.01 |
| Episodes | Training episodes | 500 |

### State Space (120 total states)
`State = (urgency_bucket × wait_bucket × capacity_bucket × fairness_bucket)`
= 5 × 4 × 3 × 2 = **120 states**

### Action Space (4 actions)
| # | Action | Description |
|---|---|---|
| 0 | Allocate Full Care | Assign bed + doctor + nurse immediately |
| 1 | Stabilize and Queue | Basic stabilisation, queue for full care |
| 2 | Queue for Later | Non-critical, wait for resources to free |
| 3 | Transfer / Refer | Send to another facility |

### Reward Function
- `+urgency × 10` for Full Care; `+20` bonus for critical (urgency ≥ 4)
- `−30` for queuing a critical patient; `−2 × overtime` past safe wait limit
- `+5` fairness bonus for treating underserved demographic groups
- `−15` for over-allocating at high capacity pressure (> 80%)
"""))

# ══════════════════════════════════════════════════════════════════════════
# CELL 11 — RL Agent class
# ══════════════════════════════════════════════════════════════════════════
cells.append(code("""\
ACTION_NAMES = [
    "Allocate Full Care",
    "Stabilize and Queue",
    "Queue for Later",
    "Transfer / Refer",
]


class QLearningAgent:
    \"\"\"Tabular Q-learning agent for hospital resource allocation.\"\"\"

    N_URGENCY  = 5   # urgency levels 1-5
    N_WAIT     = 4   # wait buckets: <15, 15-30, 30-60, >60 min
    N_CAPACITY = 3   # capacity pressure: low/medium/high
    N_FAIRNESS = 2   # fairness pressure: normal / underserved
    N_ACTIONS  = 4

    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.3, n_episodes=500):
        self.alpha     = alpha
        self.gamma     = gamma
        self.epsilon   = epsilon
        self.n_episodes = n_episodes
        self.q_table   = np.zeros((5, 4, 3, 2, 4))
        self.reward_history: list = []

    def discretize_state(self, urgency, wait_time, cap_pressure, fair_pressure):
        u = int(np.clip(int(urgency) - 1, 0, 4))
        w = (0 if wait_time < 15 else
             1 if wait_time < 30 else
             2 if wait_time < 60 else 3)
        c = (0 if cap_pressure < 0.5 else
             1 if cap_pressure < 0.8 else 2)
        f = 1 if fair_pressure > 0.5 else 0
        return (u, w, c, f)

    def compute_reward(self, urgency, wait_time, action, cap_pressure, fair_pressure):
        r = 0.0
        if action == 0:                              # Full Care
            r += urgency * 10.0 + (20.0 if urgency >= 4 else 0.0)
            if cap_pressure > 0.8:
                r -= 15.0                            # over-allocation penalty
        elif action == 1:                            # Stabilize
            r += urgency * 4.0
        elif action == 2:                            # Queue
            r -= (30.0 if urgency >= 4 else 0.0) + wait_time * 0.5
        else:                                        # Transfer
            r -= (20.0 if urgency >= 4 else 0.0) + 10.0
        if fair_pressure > 0.5 and action in (0, 1):
            r += 5.0                                 # fairness bonus
        safe_limit = max(10, 100 - urgency * 20)
        if wait_time > safe_limit:
            r -= (wait_time - safe_limit) * 2.0     # unsafe wait penalty
        return r

    def _bellman_update(self, state, action, reward, next_state):
        old_q    = self.q_table[state][action]
        best_next = float(np.max(self.q_table[next_state]))
        self.q_table[state][action] = (
            old_q + self.alpha * (reward + self.gamma * best_next - old_q)
        )

    def train(self, patients_df, resources_df):
        \"\"\"Train Q-learning agent. Returns episode reward history.\"\"\"
        self.reward_history = []
        n_patients = len(patients_df)
        total_beds = int(
            resources_df[resources_df["resource_name"] == "Beds"][
                "available_count"
            ].values[0]
        )
        eps = self.epsilon

        for _ in range(self.n_episodes):
            ep_reward  = 0.0
            curr_beds  = total_beds
            idx_order  = np.random.permutation(n_patients)[:50]

            for idx in idx_order:
                patient   = patients_df.iloc[idx]
                urgency   = patient["urgency_level"]
                wait_time = float(np.random.uniform(0, 90))
                cap       = 1.0 - curr_beds / max(total_beds, 1)
                fair      = float(np.random.uniform(0, 1))

                state  = self.discretize_state(urgency, wait_time, cap, fair)
                action = (
                    np.random.randint(self.N_ACTIONS)
                    if np.random.random() < eps
                    else int(np.argmax(self.q_table[state]))
                )
                reward = self.compute_reward(urgency, wait_time, action, cap, fair)

                if action == 0 and curr_beds > 0:
                    curr_beds = max(0, curr_beds - int(patient.get("required_bed", 1)))

                next_wait  = max(0.0, wait_time - 5.0)
                next_cap   = 1.0 - curr_beds / max(total_beds, 1)
                next_state = self.discretize_state(urgency, next_wait, next_cap, fair)

                self._bellman_update(state, action, reward, next_state)
                ep_reward += reward

            eps = max(0.01, eps * 0.995)
            self.reward_history.append(ep_reward / 50.0)

        self.epsilon = eps
        return self.reward_history

    def allocate_patients(self, patients_df, resources_df):
        \"\"\"Apply learned Q-policy to a patient batch. Returns allocation DataFrame.\"\"\"
        res     = resources_df.set_index("resource_name")
        current = {
            "Beds":      int(res.loc["Beds",      "available_count"]),
            "Doctors":   int(res.loc["Doctors",   "available_count"]),
            "Nurses":    int(res.loc["Nurses",    "available_count"]),
            "Equipment": int(res.loc["Equipment", "available_count"]),
        }
        totals  = dict(current)
        group_counts  = patients_df["fairness_group"].value_counts().to_dict()
        group_treated = {g: 0 for g in group_counts}
        allocation    = []

        for i, (_, patient) in enumerate(patients_df.iterrows()):
            urgency   = patient["urgency_level"]
            # Wait time in minutes: 0 → 90 across the batch
            wait_time = float(min(i * 0.5, 90.0))
            cap       = 1.0 - current["Beds"] / max(totals["Beds"], 1)
            group     = patient["fairness_group"]
            g_rate    = group_treated.get(group, 0) / max(group_counts.get(group, 1), 1)
            avg_rate  = sum(group_treated.values()) / max(sum(group_counts.values()), 1)
            fair      = max(0.0, avg_rate - g_rate)

            state     = self.discretize_state(urgency, wait_time, cap, fair)
            action    = ACTION_NAMES[int(np.argmax(self.q_table[state]))]

            need_bed = int(patient.get("required_bed",   1))
            need_doc = int(patient.get("required_doctor", 1))
            need_nur = int(patient.get("required_nurse",  1))
            need_equ = int(patient.get("required_equipment", 0))

            if action == "Allocate Full Care":
                if (current["Beds"]      >= need_bed and
                        current["Doctors"]   >= need_doc and
                        current["Nurses"]    >= need_nur and
                        current["Equipment"] >= need_equ):
                    current["Beds"]      -= need_bed
                    current["Doctors"]   -= need_doc
                    current["Nurses"]    -= need_nur
                    current["Equipment"] -= need_equ
                    status = "Treated"
                    group_treated[group] = group_treated.get(group, 0) + 1
                else:
                    action = "Stabilize and Queue"
                    status = "Stabilized"
            elif action == "Stabilize and Queue":
                status = "Stabilized"
            elif action == "Queue for Later":
                status = "Queued"
            else:
                status = "Transferred"

            allocation.append({
                "patient_id":    patient["patient_id"],
                "urgency_level": urgency,
                "fairness_group": group,
                "wait_time":     wait_time,
                "action":        action,
                "status":        status,
                "priority_score": round(urgency / 5.0 - cap * 0.1, 3),
            })

        return pd.DataFrame(allocation)


print("QLearningAgent class defined. Training 500 episodes…")
agent = QLearningAgent(alpha=0.10, gamma=0.90, epsilon=0.30, n_episodes=500)
reward_history = agent.train(patients_df, resources_df)
final_avg = float(np.mean(reward_history[-50:]))
print(f"Training complete.  Final avg reward (last 50 eps): {final_avg:.2f}")
print(f"Q-table shape: {agent.q_table.shape}  (total states × actions)")
"""))

# ══════════════════════════════════════════════════════════════════════════
# CELL 12 — Reward Curve Plot
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("### 7a. Training Reward Curve"))

cells.append(code("""\
fig, ax = plt.subplots(figsize=(11, 4))
eps_x = range(len(reward_history))
ax.plot(eps_x, reward_history, color="#94A3B8", linewidth=0.9,
        alpha=0.6, label="Episode Reward")
window = max(1, len(reward_history) // 20)
ma = pd.Series(reward_history).rolling(window=window, min_periods=1).mean()
ax.plot(eps_x, ma, color="#DC2626", linewidth=2.2,
        label=f"Moving Avg (w={window})")
ax.set_xlabel("Training Episode"); ax.set_ylabel("Avg Reward per Patient")
ax.set_title("Q-Learning Training Reward Curve", fontsize=13, fontweight="bold")
ax.legend(); ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig("reward_curve.png", dpi=120, bbox_inches="tight")
plt.show()
print(f"Reward: ep1={reward_history[0]:.1f}  last50_avg={final_avg:.1f}")
"""))

# ══════════════════════════════════════════════════════════════════════════
# CELL 13 — Policy Heatmap
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("### 7b. Learned Policy Heatmap (Urgency × Wait Time)"))

cells.append(code("""\
short_names = ["Full Care", "Stabilise", "Queue", "Transfer"]
n_u, n_w    = 5, 4
policy_matrix = np.zeros((n_u, n_w))
for u in range(n_u):
    for w in range(n_w):
        policy_matrix[u, w] = int(np.argmax(agent.q_table[(u, w, 0, 0)]))

cmap = matplotlib.colormaps.get_cmap("RdYlGn_r").resampled(4)
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(policy_matrix, cmap=cmap, aspect="auto", vmin=0, vmax=3)
ax.set_xticks(range(n_w))
ax.set_xticklabels(["<15 min", "15-30 min", "30-60 min", ">60 min"])
ax.set_yticks(range(n_u))
ax.set_yticklabels([f"Urgency {i+1}" for i in range(n_u)])
ax.set_title(
    "Learned RL Policy (Low Capacity, No Fairness Pressure)",
    fontsize=12, fontweight="bold",
)
for i in range(n_u):
    for j in range(n_w):
        ax.text(j, i, short_names[int(policy_matrix[i, j])],
                ha="center", va="center",
                color="white", fontweight="bold", fontsize=9)
legend_patches = [
    mpatches.Patch(color=cmap(k / 3.0), label=s)
    for k, s in enumerate(short_names)
]
ax.legend(handles=legend_patches, loc="upper right",
          bbox_to_anchor=(1.38, 1), fontsize=8)
plt.tight_layout()
plt.savefig("policy_heatmap.png", dpi=120, bbox_inches="tight")
plt.show()
"""))

# ══════════════════════════════════════════════════════════════════════════
# CELL 14 — RL Allocation
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("### 7c. Apply RL Policy to 200 Patients"))

cells.append(code("""\
rl_alloc_df = agent.allocate_patients(patients_df.head(200), resources_df)
print("RL allocation status counts:")
print(rl_alloc_df["status"].value_counts())
print("\\nSample allocation:")
rl_alloc_df.head(8)
"""))

# ══════════════════════════════════════════════════════════════════════════
# CELL 15 — Constraint Optimizer (markdown)
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## 8. Constraint-Optimisation Baseline

**RDMU Concept 4: Constraint-Based Decision Making**

The optimiser allocates patients using **deterministic feasibility rules**
(not a formal LP/IP solver). It is a greedy weighted-priority queue with
hard resource constraints.

### Priority Score Formula
```
Score = urgency_weight × (urgency/5)
      + fairness_weight × fairness_bonus
      + efficiency_weight × (wait_time/120)
      − 0.10 × resource_burden
```

### Hard Constraints
- Available Beds: 30 (5 reserved as emergency buffer for urgency ≥ 4)
- Available Doctors: 10 (2 reserved for critical patients)
- Available Nurses: 20 | Equipment: 15

### Emergency Buffer
Urgency ≥ 4 patients bypass the buffer and can use all available resources.
"""))

# ══════════════════════════════════════════════════════════════════════════
# CELL 16 — Optimizer code
# ══════════════════════════════════════════════════════════════════════════
cells.append(code("""\
def _priority_score(patient, wait_time, group_counts, group_treated,
                    fw=0.30, uw=0.50, ew=0.20):
    urgency_norm = patient["urgency_level"] / 5.0
    wait_norm    = min(wait_time / 120.0, 1.0)
    group        = patient["fairness_group"]
    g_total      = max(group_counts.get(group, 1), 1)
    g_treated    = group_treated.get(group, 0)
    fair_bonus   = 1.0 - (g_treated / g_total)
    burden = (
        int(patient.get("required_bed",       1)) +
        int(patient.get("required_doctor",    1)) +
        int(patient.get("required_nurse",     1)) +
        int(patient.get("required_equipment", 0))
    ) / 6.0
    return uw * urgency_norm + fw * fair_bonus + ew * wait_norm - 0.10 * burden


def run_optimizer(patients_df, resources_df, fw=0.30, uw=0.50, ew=0.20):
    \"\"\"Allocate patients using deterministic feasibility rules.
    Returns (allocation_df, used_dict, available_dict).
    \"\"\"
    res       = resources_df.set_index("resource_name")
    available = {n: int(res.loc[n, "available_count"])
                 for n in ["Beds", "Doctors", "Nurses", "Equipment"]}
    buffer    = {n: int(res.loc[n, "emergency_buffer"])
                 for n in ["Beds", "Doctors", "Nurses", "Equipment"]}
    group_counts  = patients_df["fairness_group"].value_counts().to_dict()
    group_treated = {g: 0 for g in group_counts}

    scored = []
    for i, (_, p) in enumerate(patients_df.iterrows()):
        # Wait time in minutes: 0 → 90 across the batch (realistic ED queue)
        wt    = min(float(i) * 0.5, 90.0)
        score = _priority_score(p, wt, group_counts, group_treated, fw, uw, ew)
        scored.append({**p.to_dict(), "_score": score, "_wait": wt})

    scored.sort(key=lambda x: -x["_score"])
    used       = {k: 0 for k in available}
    allocation = []

    for p in scored:
        urgency   = p["urgency_level"]
        group     = p["fairness_group"]
        buf_mult  = 0 if urgency >= 4 else 1
        eff_avail = {k: available[k] - buf_mult * buffer[k] for k in available}
        need      = {
            "Beds":      int(p.get("required_bed",       1)),
            "Doctors":   int(p.get("required_doctor",    1)),
            "Nurses":    int(p.get("required_nurse",     1)),
            "Equipment": int(p.get("required_equipment", 0)),
        }
        can = all(used[k] + need[k] <= eff_avail[k] for k in need)

        if can:
            action = "Allocate Full Care"
            status = "Treated"
            for k in need:
                used[k] += need[k]
            group_treated[group] = group_treated.get(group, 0) + 1
        elif urgency >= 4:
            if used["Doctors"] < available["Doctors"]:
                action, status = "Stabilize and Queue", "Stabilized"
                used["Doctors"] += 1
            else:
                action, status = "Transfer / Refer", "Transferred"
        elif urgency >= 3:
            action, status = "Stabilize and Queue", "Queued"
        else:
            action, status = "Queue for Later", "Queued"

        allocation.append({
            "patient_id":    p["patient_id"],
            "urgency_level": urgency,
            "fairness_group": group,
            "priority_score": round(p["_score"], 3),
            "wait_time":     p["_wait"],
            "action":        action,
            "status":        status,
        })

    return pd.DataFrame(allocation), used, available


opt_alloc_df, used_resources, available_resources = run_optimizer(
    patients_df.head(200), resources_df
)
print("Optimisation status counts:")
print(opt_alloc_df["status"].value_counts())
print("\\nResources used:")
for k in used_resources:
    util = used_resources[k] / max(available_resources[k], 1)
    print(f"  {k:12s}: {used_resources[k]:3d} / {available_resources[k]}  "
          f"({util:.0%} utilised)")
opt_alloc_df.head(8)
"""))

# ══════════════════════════════════════════════════════════════════════════
# CELL 17 — Metrics helper
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("## 9. Metrics Computation — All KPIs"))

cells.append(code("""\
def jains_fairness_index(values):
    \"\"\"Jain's Fairness Index = (sum xi)^2 / (n * sum xi^2).\"\"\"
    arr = np.array(values, dtype=float)
    n   = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    return float((arr.sum() ** 2) / (n * (arr ** 2).sum()))


def compute_all_metrics(alloc_df, resources_df=None):
    \"\"\"
    Compute full KPI dict from batch allocation.
    Pass resources_df for accurate resource utilisation (doctors/beds).
    \"\"\"
    m = {
        "total_patients":   len(alloc_df),
        "treated_count":    int((alloc_df["status"] == "Treated").sum()),
        "stabilized_count": int((alloc_df["status"] == "Stabilized").sum()),
        "queued_count":     int(alloc_df["status"].isin(["Queued"]).sum()),
        "transferred_count": int((alloc_df["status"] == "Transferred").sum()),
        "avg_wait_time":    round(float(alloc_df["wait_time"].mean()), 1),
        "max_wait_time":    round(float(alloc_df["wait_time"].max()),  1),
        "avg_urgency":      round(float(alloc_df["urgency_level"].mean()), 2),
    }

    # Jain's Fairness Index across demographic groups
    if "fairness_group" in alloc_df.columns:
        treated_mask  = alloc_df["status"] == "Treated"
        grp_total     = alloc_df.groupby("fairness_group").size()
        grp_treated   = (alloc_df[treated_mask]
                         .groupby("fairness_group").size()
                         .reindex(grp_total.index, fill_value=0))
        group_rates   = grp_treated / grp_total.clip(lower=1)
        m["fairness_index"] = round(jains_fairness_index(group_rates.values), 3)
    else:
        m["fairness_index"] = 0.0

    # Urgency handling: % of critical (≥4) patients that were treated
    crit = alloc_df[alloc_df["urgency_level"] >= 4]
    m["urgency_handling_score"] = round(
        float((crit["status"] == "Treated").sum() / max(len(crit), 1)), 3
    ) if len(crit) > 0 else 1.0

    # Patient outcome score: urgency-weighted quality
    def _outcome(row):
        if row["status"] == "Treated":
            return row["urgency_level"] * 10.0
        if row["status"] == "Stabilized":
            return row["urgency_level"] * 5.0
        if row["status"] == "Queued":
            return max(0.0, row["urgency_level"] * 3.0 - row["wait_time"] * 0.1)
        return 0.0

    m["patient_outcome_score"] = round(
        float(alloc_df.apply(_outcome, axis=1).mean()), 2
    )

    # Resource utilisation: doctors used / doctors available
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

    # Capacity violations: critical patients NOT given full treatment
    m["capacity_violation_count"] = int(
        (
            (alloc_df["urgency_level"] >= 4) &
            (alloc_df["status"] != "Treated")
        ).sum()
    )

    return m


rl_m  = compute_all_metrics(rl_alloc_df,  resources_df)
opt_m = compute_all_metrics(opt_alloc_df, resources_df)

kpi_labels = [
    ("total_patients",       "Total Patients"),
    ("treated_count",        "Treated (Concurrent Slots)"),
    ("stabilized_count",     "Stabilized"),
    ("queued_count",         "Queued"),
    ("transferred_count",    "Transferred"),
    ("avg_wait_time",        "Avg Wait Time (min)"),
    ("fairness_index",       "Fairness Index (Jain's)"),
    ("urgency_handling_score","Urgency Handling Score"),
    ("patient_outcome_score","Patient Outcome Score"),
    ("resource_utilization", "Resource Utilisation (Doctors+Beds)"),
    ("capacity_violation_count","Critical Patients Not Treated"),
]

compare_df = pd.DataFrame([
    {
        "KPI":                  label,
        "RL Strategy":          rl_m.get(key, "—"),
        "Optimisation Baseline": opt_m.get(key, "—"),
    }
    for key, label in kpi_labels
])
print(compare_df.to_string(index=False))
"""))

# ══════════════════════════════════════════════════════════════════════════
# CELL 18 — Policy Comparison Chart
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("## 10. Policy Comparison Chart"))

cells.append(code("""\
plot_keys   = ["resource_utilization", "fairness_index",
               "urgency_handling_score", "patient_outcome_score"]
plot_labels = ["Resource Util.", "Fairness Index",
               "Urgency Score", "Outcome (norm.)"]


def _normalize(val, key):
    \"\"\"Normalize outcome score to 0-1 range for plotting.\"\"\"
    if key == "patient_outcome_score":
        return min(float(val) / 50.0, 1.0)
    return float(val)


rl_vals  = [_normalize(rl_m.get(k,  0), k) for k in plot_keys]
opt_vals = [_normalize(opt_m.get(k, 0), k) for k in plot_keys]

x = np.arange(len(plot_labels))
w = 0.36

fig, ax = plt.subplots(figsize=(10, 5))
bars_rl  = ax.bar(x - w / 2, rl_vals,  w, label="RL Strategy",          color="#2563EB", alpha=0.88)
bars_opt = ax.bar(x + w / 2, opt_vals, w, label="Optimisation Baseline", color="#16A34A", alpha=0.88)
ax.set_xticks(x); ax.set_xticklabels(plot_labels, fontsize=10)
ax.set_ylim(0, 1.2); ax.set_ylabel("Score (0 – 1)")
ax.set_title("Policy Comparison: RL Strategy vs Constraint-Optimisation Baseline",
             fontsize=12, fontweight="bold")
ax.legend(); ax.grid(True, axis="y", alpha=0.25)
for bar in list(bars_rl) + list(bars_opt):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.2f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
plt.tight_layout()
plt.savefig("policy_comparison.png", dpi=120, bbox_inches="tight")
plt.show()
print("Key insight: RL adapts to fairness pressure; Optimiser is deterministic but always feasible.")
"""))

# ══════════════════════════════════════════════════════════════════════════
# CELL 19 — Sensitivity Analysis (markdown)
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## 11. Sensitivity Analysis — Fairness vs Efficiency

**RDMU Concept: Sensitivity Analysis**

We sweep the **fairness weight** from 0.05 → 0.90 and observe how
key metrics respond. This reveals the inherent trade-off:
- Higher fairness weight → more equal treatment across demographic groups
- Lower fairness weight → higher throughput and resource efficiency

The **Jain's Fairness Index** target is ≥ 0.90 for equitable care.
"""))

# ══════════════════════════════════════════════════════════════════════════
# CELL 20 — Sensitivity Analysis (code)
# ══════════════════════════════════════════════════════════════════════════
cells.append(code("""\
sensitivity_results = []

for fw in np.linspace(0.05, 0.90, 18):
    ew = round(max(0.05, 1.0 - fw), 2)
    uw = round(min(0.70, fw * 0.4 + 0.35), 2)
    alloc_s, _, _ = run_optimizer(
        patients_df.head(120), resources_df, fw, uw, ew
    )
    ms = compute_all_metrics(alloc_s, resources_df)
    sensitivity_results.append({
        "fairness_weight":    round(fw, 2),
        "efficiency_weight":  round(ew, 2),
        "urgency_weight":     round(uw, 2),
        "fairness_index":     ms["fairness_index"],
        "resource_utilization": ms["resource_utilization"],
        "avg_wait_time":      ms["avg_wait_time"],
        "treated_count":      ms["treated_count"],
        "patient_outcome_score": ms["patient_outcome_score"],
    })

sens_df = pd.DataFrame(sensitivity_results)
sens_df.to_csv("data/sensitivity_results.csv", index=False)
print(f"Sensitivity sweep complete: {len(sens_df)} data points")
sens_df[["fairness_weight", "fairness_index", "resource_utilization",
         "avg_wait_time", "treated_count"]].round(3)
"""))

# ══════════════════════════════════════════════════════════════════════════
# CELL 21 — Sensitivity Charts
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("### 11a. Sensitivity Trade-off Charts"))

cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Chart 1: Fairness Weight vs Fairness Index
sc1 = axes[0].scatter(
    sens_df["fairness_weight"], sens_df["fairness_index"],
    c=sens_df["efficiency_weight"], cmap="viridis",
    s=80, alpha=0.85, edgecolors="none",
)
axes[0].set_xlabel("Fairness Weight");  axes[0].set_ylabel("Fairness Index (Jain's)")
axes[0].set_title("Fairness Weight → Fairness Index", fontweight="bold")
axes[0].grid(True, alpha=0.25)
plt.colorbar(sc1, ax=axes[0], label="Efficiency Weight")

# Chart 2: Wait Time vs Resource Utilisation
sc2 = axes[1].scatter(
    sens_df["avg_wait_time"], sens_df["resource_utilization"],
    c=sens_df["fairness_weight"], cmap="RdYlGn",
    s=80, alpha=0.85, edgecolors="none",
)
axes[1].set_xlabel("Avg Wait Time (min)")
axes[1].set_ylabel("Resource Utilisation")
axes[1].set_title("Wait Time vs Resource Utilisation", fontweight="bold")
axes[1].grid(True, alpha=0.25)
plt.colorbar(sc2, ax=axes[1], label="Fairness Weight")

plt.suptitle("Sensitivity Analysis: Fairness vs Efficiency Trade-off",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("sensitivity_analysis.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved to sensitivity_analysis.png and data/sensitivity_results.csv")
"""))

# ══════════════════════════════════════════════════════════════════════════
# CELL 22 — Metrics Heatmap
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("## 12. Performance Metrics Heatmap (imshow)"))

cells.append(code("""\
# Combined metrics heatmap for optimizer results
heatmap_names = [
    ["Treatment Rate", "Wait Score (inv.)", "Resource Util."],
    ["Fairness Index", "Urgency Score",     "Outcome Score"],
]
raw_vals = [
    [
        opt_m["treated_count"] / max(opt_m["total_patients"], 1),
        1.0 - min(opt_m["avg_wait_time"] / 90.0, 1.0),
        opt_m["resource_utilization"],
    ],
    [
        opt_m["fairness_index"],
        opt_m["urgency_handling_score"],
        min(opt_m["patient_outcome_score"] / 50.0, 1.0),
    ],
]
data = np.array(raw_vals)

fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(3)); ax.set_xticklabels(["A", "B", "C"], fontsize=9)
ax.set_yticks(range(2));
ax.set_yticklabels(["Treatment Efficiency", "Care Quality"], fontsize=9)
for i in range(2):
    for j in range(3):
        ax.text(j, i,
                f"{data[i, j]:.2f}\\n{heatmap_names[i][j]}",
                ha="center", va="center",
                color="black", fontsize=9, fontweight="bold")
plt.colorbar(im, ax=ax, label="Score (0=Low  1=High)", fraction=0.03)
ax.set_title("Constraint-Optimiser — Performance Metrics Heatmap",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("metrics_heatmap.png", dpi=120, bbox_inches="tight")
plt.show()
"""))

# ══════════════════════════════════════════════════════════════════════════
# CELL 23 — RDMU Concept Mapping Table
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("## 13. RDMU Concept Mapping"))

cells.append(code("""\
rdmu_table = pd.DataFrame({
    "RDMU Concept": [
        "Uncertainty",
        "State (MDP)",
        "Action (MDP)",
        "Reward (MDP)",
        "Policy",
        "Reinforcement Learning",
        "Constraint-Based Decision",
        "Fairness",
        "Sensitivity Analysis",
        "Human-in-the-Loop",
    ],
    "Meaning": [
        "Unknown future patient arrivals and severity",
        "Snapshot: urgency, wait time, capacity, fairness",
        "Allocation decision per patient (4 choices)",
        "Scalar feedback signal for agent quality",
        "Mapping of every state to its best action",
        "Q-learning updates policy from experience",
        "Hard resource limits must never be exceeded",
        "Equal treatment opportunity across all groups",
        "Vary parameters to observe outcome changes",
        "Clinician reviews and can override AI decision",
    ],
    "Where in App": [
        "Simulator: Poisson arrival process",
        "RL Engine: state space & Q-table",
        "Q-table action selection & allocator",
        "RL Engine: reward curve chart",
        "RL Engine: learned policy heatmap & table",
        "RL Engine: Bellman update training loop",
        "Optimiser: resource feasibility checks",
        "Fairness Index KPI; Sensitivity Analysis",
        "Sensitivity Analysis page sliders",
        "Simulator: manual resource override controls",
    ],
    "Example from Project": [
        "Arrivals ~ Poisson(λ); severity ~ Normal(urgency×15, 5)",
        "State=(Urgency=5, Wait=30min, Cap=75%, Underserved)",
        "Action 0 = Allocate Full Care to Level-5 patient",
        "Reward = +70 for treating critical before safe-wait exceeded",
        "{Urgency=5, Wait>30min} → Always Allocate Full Care",
        "Q(s,a) ← Q(s,a) + α[r + γ max Q(s') − Q(s,a)]",
        "30 beds max; 5 reserved as emergency buffer (urgency ≥ 4)",
        "Jain's Fairness Index ≥ 0.90 target across Group A/B/C",
        "fairness_weight 0.05→0.90: Fairness ↑ as Efficiency ↓",
        "Clinician overrides Transfer → Full Care for borderline patient",
    ],
})
print(rdmu_table.to_string(index=False))
rdmu_table
"""))

# ══════════════════════════════════════════════════════════════════════════
# CELL 24 — Streamlit Dashboard Summary
# ══════════════════════════════════════════════════════════════════════════
cells.append(md(f"""\
## 14. Streamlit Dashboard Summary

The Streamlit dashboard (`app.py`) provides **10 interactive pages**.

| Page | Description |
|---|---|
| 1. Executive Dashboard | KPI cards, urgency chart, resource bar chart, project links |
| 2. Data Preview | patients.csv & resources.csv preview, regenerate button |
| 3. Resource Allocation Simulator | Poisson time-step simulation with slider controls |
| 4. RL Policy Engine | Q-learning training, reward curve, policy heatmap |
| 5. Optimisation Baseline | Constraint-based allocation with weight controls |
| 6. Policy Comparison | RL vs Optimiser side-by-side metrics and bar chart |
| 7. Sensitivity Analysis | Fairness weight sweep trade-off charts |
| 8. Metrics and Performance | Heatmap, all KPIs, export CSV |
| 9. RDMU Concept Mapping | All 10 RDMU concepts mapped to app features |
| 10. Limitations and Future Scope | Academic disclaimer and roadmap |

### Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Deployment Links
- Google Colab: `{COLAB_LINK}`
- GitHub Repository: `{GITHUB_LINK}`
- Streamlit App: `{STREAMLIT_LINK}`
"""))

# ══════════════════════════════════════════════════════════════════════════
# CELL 25 — Limitations
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## 15. Limitations and Future Scope

### Current Limitations
- **Synthetic data only** — not validated against real clinical outcomes
- **Tabular Q-learning** — does not scale to continuous state spaces
- **Heuristic optimiser** — no formal LP/IP solver (PuLP, OR-Tools)
- **Not production-ready** — no authentication, audit logging, or regulatory review
- **Fairness metric** — only Jain's Index; other criteria (equal opportunity, disparate impact) not assessed

### Future Scope
- Deep RL (DQN, PPO) for continuous state spaces
- Real hospital data integration via HL7 FHIR APIs
- Formal LP/IP optimisation using PuLP or OR-Tools
- Multi-objective Pareto analysis (efficiency, fairness, cost, safety)
- Explainable AI (SHAP values) for clinical transparency
- Federated learning across hospital networks

---
*This prototype is for educational purposes only and is not suitable for clinical use.*
"""))

# ══════════════════════════════════════════════════════════════════════════
# CELL 26 — Final Summary Print
# ══════════════════════════════════════════════════════════════════════════
cells.append(code("""\
print("=" * 65)
print("MediFlow AI — Notebook Execution Summary")
print("=" * 65)
print(f"Patients generated        : {len(patients_df)}")
print(f"RL training episodes      : {agent.n_episodes}")
print(f"RL final avg reward       : {final_avg:.2f}")
print()
print("--- RL Strategy ---")
print(f"  Treated (concurrent)    : {rl_m['treated_count']} / {rl_m['total_patients']}")
print(f"  Stabilized              : {rl_m['stabilized_count']}")
print(f"  Avg Wait Time           : {rl_m['avg_wait_time']} min")
print(f"  Fairness Index (Jain's) : {rl_m['fairness_index']}")
print(f"  Resource Utilisation    : {rl_m['resource_utilization']:.0%}")
print(f"  Critical Not Treated    : {rl_m['capacity_violation_count']}")
print()
print("--- Constraint-Optimisation Baseline ---")
print(f"  Treated (concurrent)    : {opt_m['treated_count']} / {opt_m['total_patients']}")
print(f"  Queued                  : {opt_m['queued_count']}")
print(f"  Avg Wait Time           : {opt_m['avg_wait_time']} min")
print(f"  Fairness Index (Jain's) : {opt_m['fairness_index']}")
print(f"  Resource Utilisation    : {opt_m['resource_utilization']:.0%}")
print(f"  Critical Not Treated    : {opt_m['capacity_violation_count']}")
print()
print("Sensitivity results saved : data/sensitivity_results.csv")
print("Charts saved              : reward_curve.png, policy_heatmap.png,")
print("                            policy_comparison.png, sensitivity_analysis.png")
print("=" * 65)
print("Notebook execution complete. Made by Krishna Mathur | AS25DXB018")
"""))

# ══════════════════════════════════════════════════════════════════════════
# Build and save notebook
# ══════════════════════════════════════════════════════════════════════════
nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language":     "python",
    "name":         "python3",
}
nb.metadata["language_info"] = {
    "name":    "python",
    "version": "3.8.0",
}

with open(NB_PATH, "w", encoding="utf-8") as fh:
    nbf.write(nb, fh)

print(f"Notebook written to {NB_PATH}")
print(f"Total cells: {len(cells)}")
code_cells = sum(1 for c in cells if c["cell_type"] == "code")
md_cells   = sum(1 for c in cells if c["cell_type"] == "markdown")
print(f"  Code cells: {code_cells}")
print(f"  Markdown cells: {md_cells}")
