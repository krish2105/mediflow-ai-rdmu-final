# MediFlow AI — PPT Slide Content
## MAIB DSC 103 Final Exam — Krishna Mathur | AS25DXB018

---

### SLIDE 1: Title Slide
**Title:** MediFlow AI: Dynamic Hospital Emergency Resource Allocation using Reinforcement Learning

**Subtitle / Details:**
- Course: Reasoning and Decision Making under Uncertainty — MAIB DSC 103
- Made by: Krishna Mathur | AS25DXB018 | MAIB September
- Exam Topic: Topic 1 — Dynamic Resource Allocation using Reinforcement Learning
- Submission Date: June 2026

**Links (replace placeholders):**
- Google Colab: [PASTE_ACCESSIBLE_COLAB_LINK_HERE]
- GitHub Repo: [PASTE_GITHUB_REPO_LINK_HERE]
- Streamlit App: [PASTE_STREAMLIT_APP_LINK_HERE]

**Footer:** Made by Krishna Mathur | AS25DXB018 | MAIB September

**Speaker Notes:**
This presentation covers the design, development, and evaluation of MediFlow AI,
a hospital emergency resource allocation simulator built using Reinforcement Learning
and constraint-based decision making for the MAIB DSC 103 final examination.

---

### SLIDE 2: Requirement Understanding
**Title:** Requirement Understanding

**Bullets:**
- Topic: Dynamic Resource Allocation System using Reinforcement Learning
- Build a working prototype that allocates hospital resources to patients under uncertainty
- Minimum 4 RDMU key concepts applied: Uncertainty, State-Action-Reward, RL/Q-Learning, Constraint Optimisation
- Deliverables: Google Colab notebook, GitHub repository, Streamlit deployed app, 10-slide PPT
- The system must compare RL-driven vs. rule-based allocation strategies
- Sensitivity analysis: fairness vs. efficiency trade-off must be visualised

**Speaker Notes:**
The exam asks us to pick one of five topics and build a fully working prototype.
I selected Topic 1 and built MediFlow AI to simulate hospital resource allocation
using reinforcement learning, with a constraint-based baseline for comparison.

---

### SLIDE 3: Business Problem
**Title:** Business Problem — Why Resource Allocation Matters

**Bullets:**
- Emergency departments face unpredictable patient arrivals every day (Uncertainty)
- Limited resources: beds, doctors, nurses, and equipment cannot serve all patients at once
- Wrong allocation decisions increase patient waiting time and raise health risk
- Fairness must be maintained across all demographic groups — not just efficiency
- Current manual triage is slow, inconsistent, and does not optimise across multiple constraints
- AI-driven allocation can learn adaptive policies and improve outcomes over time

**Suggested Visual:** A simple diagram showing patients (varying urgency levels) arriving at an ED with limited beds and staff — some patients in a queue.

**Speaker Notes:**
Emergency departments worldwide face resource scarcity daily. A patient arriving at urgency level 5
may wait too long if the allocation system is purely manual or rule-based. MediFlow AI uses RL to
learn policies that balance urgency, fairness, and resource limits simultaneously.

---

### SLIDE 4: Application Flow Chart
**Title:** Application Flow Chart

**Flow Diagram:**
```
Patient Data Input (500 synthetic patients)
          ↓
Resource Capacity Input (Beds=30, Doctors=10, Nurses=20, Equipment=15)
          ↓
    ┌─────────────────┐         ┌────────────────────────┐
    │  RL Policy      │         │  Constraint-Optimisation│
    │  Engine         │         │  Baseline               │
    │  (Q-Learning)   │         │  (Feasibility Rules)    │
    └─────────────────┘         └────────────────────────┘
          ↓                                ↓
          └──────────── Policy Comparison ─────────────┘
                             ↓
                    Dashboard Metrics
              (Fairness Index, Utilisation, etc.)
                             ↓
                  Human Review Decision
              (Clinician overrides via Streamlit)
```

**Bullets:**
- Data flows from patient input through two parallel decision engines
- RL agent uses Q-table to choose actions; Optimiser uses priority score + feasibility
- Both produce an allocation table; metrics are computed and compared
- Clinician can override any decision via the Streamlit dashboard

**Speaker Notes:**
The system has two parallel allocation engines. The RL agent learns from simulated
experience while the optimiser applies deterministic feasibility rules. Both results
are compared on the dashboard so clinicians can make informed override decisions.

---

### SLIDE 5: Prerequisites for the Application
**Title:** Prerequisites for the Application

**Bullets:**
- Python 3.8 or later
- Google Colab (cloud) or local Python environment with Jupyter
- Key libraries: pandas, numpy, matplotlib, streamlit, nbformat, python-pptx
- Basic understanding of Reinforcement Learning and MDP framework
- GitHub account for repository hosting and version control
- Streamlit Community Cloud account for free app deployment
- No paid APIs, no external databases, no internet connection after package install

**Installation Command:**
```
pip install streamlit pandas numpy scikit-learn matplotlib nbformat python-pptx
```

**Speaker Notes:**
The entire application runs on freely available tools. There are no secret API keys,
no paid services, and no external dataset downloads. Everything runs on synthetic data
generated at runtime, making it fully reproducible anywhere.

---

### SLIDE 6: Description of Data Used
**Title:** Description of Data Used

**Bullets:**
- All data is synthetically generated using NumPy random functions (seed=42 for reproducibility)
- patients.csv — 500 records with columns: patient_id, arrival_time, urgency_level (1–5), severity_score, age_group, required_bed, required_doctor, required_nurse, required_equipment, max_safe_wait_time, fairness_group, expected_treatment_time
- resources.csv — 4 rows representing Beds (30), Doctors (10), Nurses (20), Equipment (15) with emergency buffer and cost
- Patient arrivals follow a Poisson process (stochastic); urgency follows a weighted discrete distribution
- Fairness groups (A, B, C) simulate demographic diversity in the patient population
- Why synthetic: Real patient data requires HIPAA/GDPR compliance and ethical approval — not feasible for an exam prototype

**Speaker Notes:**
Synthetic data allows full reproducibility without privacy concerns. The distributions
are calibrated to mimic realistic ED patterns: most patients are moderate urgency (Level 3),
with ~15% critical (Level 5) cases. Resource requirements vary by patient.

---

### SLIDE 7: Reinforcement Learning Design
**Title:** Reinforcement Learning Design — Q-Learning Agent

**Bullets:**
- State = (urgency_bucket, wait_time_bucket, capacity_pressure_bucket, fairness_pressure_bucket) → 120 total states
- Action = 4 decisions: Allocate Full Care | Stabilize and Queue | Queue for Later | Transfer / Refer
- Reward = +10×urgency for treating; +20 bonus for critical patients; –30 for queuing critical; –2×overtime for unsafe waits; +5 fairness bonus for underserved groups
- Q-learning update: Q(s,a) ← Q(s,a) + α[r + γ·max Q(s') − Q(s,a)]
- Hyperparameters: α=0.10, γ=0.90, ε=0.30 (decaying to 0.01), 500 training episodes
- ε-greedy exploration: agent randomly explores early and exploits learned policy later

**Speaker Notes:**
The Q-learning agent formulates resource allocation as a Markov Decision Process.
Over 500 episodes, it learns that critical patients (urgency 5) must always receive
Full Care even at moderate capacity pressure, while low-urgency patients can be safely queued.

---

### SLIDE 8: Constraint Optimisation and Fairness
**Title:** Constraint Optimisation and Fairness Trade-Off

**Bullets:**
- Constraint-Optimisation Baseline: deterministic feasibility rules with weighted priority scoring
- Priority Score = urgency_weight × urgency + fairness_weight × fairness_bonus + efficiency_weight × wait_norm − 0.10 × resource_burden
- Hard constraints: beds, doctors, nurses, equipment must not exceed available_count
- Emergency buffer: 5 beds and 2 doctors reserved exclusively for urgency ≥ 4 patients
- Fairness weight controls the trade-off: higher fairness weight → more equitable treatment across groups but lower overall throughput
- Jain's Fairness Index measures equal treatment across Group A / B / C (target ≥ 0.90)

**Speaker Notes:**
The constraint optimiser serves as our baseline comparison. It always produces
feasible allocations, but it is static — it cannot learn from experience or adapt
to changing patient patterns the way the RL agent can.

---

### SLIDE 9: User Interface / Dashboard
**Title:** Streamlit Dashboard — Interactive Decision Support

**Bullets:**
- 10-page interactive dashboard built with Streamlit, fully deployable on Streamlit Community Cloud
- Page 1: Executive Dashboard — KPI cards, urgency pie chart, resource bar chart, project links
- Pages 2–5: Data Preview, Simulation, RL Policy Engine, Optimisation Baseline
- Pages 6–8: Policy Comparison, Sensitivity Analysis, Metrics and Performance
- Pages 9–10: RDMU Concept Mapping, Limitations and Future Scope
- All charts built with Matplotlib (no Seaborn); CSV download buttons for all major outputs

**Links (replace after deployment):**
- Google Colab: [PASTE_ACCESSIBLE_COLAB_LINK_HERE]
- GitHub: [PASTE_GITHUB_REPO_LINK_HERE]
- Streamlit App: [PASTE_STREAMLIT_APP_LINK_HERE]

**Speaker Notes:**
The Streamlit dashboard allows non-technical users to interact with the RL agent
and optimiser without writing any code. Clinicians can adjust resource levels,
retrain the RL agent, and compare strategies — all through a browser interface.

---

### SLIDE 10: Results, Limitations, and Future Scope
**Title:** Results, Limitations, and Future Scope

**Results:**
- RL agent converges within ~300 episodes; learns to prioritise critical patients
- Sensitivity analysis confirms: fairness_weight ↑ → fairness_index ↑, resource_utilisation ↓
- Optimiser: deterministic, always feasible, higher throughput in low-resource scenarios
- RL: adaptive, learns fairness-aware policies, improves with more episodes

**Limitations:**
- Synthetic data only — not validated against real clinical outcomes
- Tabular Q-learning does not scale to continuous state spaces
- No formal LP/IP solver; heuristic constraint handling

**Future Scope:**
- Deep RL (DQN/PPO) for larger state spaces
- Real hospital data integration via HL7 FHIR APIs
- Formal LP/IP optimisation using PuLP or OR-Tools
- Explainable AI using SHAP values for clinical transparency
- Federated learning across multiple hospital sites

**Footer:** Made by Krishna Mathur | AS25DXB018 | MAIB September
