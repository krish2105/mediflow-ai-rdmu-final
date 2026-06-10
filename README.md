# MediFlow AI
## Dynamic Hospital Emergency Resource Allocation System using Reinforcement Learning

**Course:** Reasoning and Decision Making under Uncertainty — MAIB DSC 103  
**Student:** Krishna Mathur | Student ID: AS25DXB018 | MAIB September  
**Selected Topic:** Topic 1 — Dynamic Resource Allocation using Reinforcement Learning  

---

## Project Links (Replace after deployment)

| Resource | Link |
|---|---|
| Google Colab Notebook | `[PASTE_ACCESSIBLE_COLAB_LINK_HERE]` |
| GitHub Repository | `[PASTE_GITHUB_REPO_LINK_HERE]` |
| Streamlit App | `[PASTE_STREAMLIT_APP_LINK_HERE]` |

---

## 1. Project Overview

MediFlow AI is a hospital emergency department resource allocation simulator that uses
**Reinforcement Learning (Q-Learning)** and **Constraint-Based Optimisation** to make
intelligent allocation decisions under uncertainty. The system learns adaptive policies
that balance urgency, fairness, and resource efficiency.

---

## 2. Selected Exam Topic

**Topic 1:** Develop a dynamic resource allocation system using Reinforcement Learning.

Simulates a hospital emergency department where limited resources (beds, staff, equipment)
must be allocated to patients with varying urgency levels.

---

## 3. Requirement Understanding

Build a working software prototype that:
- Simulates stochastic patient arrivals in an ED
- Allocates limited resources (beds, doctors, nurses, equipment) to patients
- Uses RL to learn an allocation policy
- Compares RL-driven vs. constraint-based strategies
- Includes fairness vs. efficiency sensitivity analysis
- Provides an interactive Streamlit dashboard

---

## 4. RDMU Key Concepts Used (Minimum 4)

| # | Concept | Application in MediFlow AI |
|---|---|---|
| 1 | **Uncertainty** | Patient arrivals follow Poisson process; severity is stochastic |
| 2 | **State / Action / Reward (MDP)** | Q-learning formulated as full MDP with 120 states and 4 actions |
| 3 | **Reinforcement Learning (Q-Learning)** | Tabular Q-learning agent trained over 500 episodes |
| 4 | **Constraint-Based Decision Making** | Hard resource limits + emergency buffers enforced |
| 5 | **Fairness** | Jain's Fairness Index; fairness weight controls equity vs. efficiency |
| 6 | **Sensitivity Analysis** | Parameter sweep over fairness weight 0.05 → 0.90 |
| 7 | **Policy Comparison** | RL vs. Optimiser compared on 10 metrics |
| 8 | **Human-in-the-Loop** | Clinician can override decisions via Streamlit controls |

---

## 5. Application Flow

```
Patient Data (500 synthetic) → Resource Capacity Input
              ↓                           ↓
      RL Policy Engine         Constraint-Optimisation Baseline
      (Q-Learning Agent)       (Heuristic Feasibility Rules)
              ↓                           ↓
              └──── Policy Comparison ────┘
                          ↓
                 Dashboard Metrics
         (Fairness Index, Utilisation, etc.)
                          ↓
              Human Review Decision (Streamlit)
```

---

## 6. Prerequisites

- Python 3.8+
- Google Colab or local Jupyter environment
- Libraries: see `requirements.txt`

---

## 7. Data Description

| File | Rows | Description |
|---|---|---|
| `data/patients.csv` | 500 | Synthetic ED patients — urgency 1-5, resources needed, fairness group |
| `data/resources.csv` | 4 | Hospital resource capacity — Beds(30), Doctors(10), Nurses(20), Equipment(15) |

All data is synthetically generated using NumPy (seed=42). No real patient data is used.

---

## 8. Dashboard Pages

1. **Executive Dashboard** — KPIs, urgency chart, resource chart
2. **Data Preview** — CSV previews, distributions, regenerate button
3. **Resource Allocation Simulator** — Time-step simulation
4. **RL Policy Engine** — Q-learning training + policy visualisation
5. **Optimisation Baseline** — Constraint-based allocation
6. **Policy Comparison** — RL vs Optimiser metrics and charts
7. **Sensitivity Analysis** — Fairness vs efficiency sweep
8. **Metrics and Performance** — Heatmap, KPIs, export
9. **RDMU Concept Mapping** — All concepts mapped to app
10. **Limitations and Future Scope** — Academic disclaimer

---

## 9. Tech Stack

| Component | Technology |
|---|---|
| Dashboard | Streamlit |
| Data Processing | pandas, numpy |
| RL Agent | numpy (from scratch Q-learning) |
| Visualisation | matplotlib (no seaborn) |
| Notebook | Jupyter / Google Colab (nbformat) |
| Presentation | python-pptx |

---

## 10. Installation

```bash
# 1. Clone or download the repository
git clone [PASTE_GITHUB_REPO_LINK_HERE]
cd MediFlowAI

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate    # macOS/Linux
# venv\Scripts\activate      # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 11. Local Run Steps

```bash
# Run Streamlit dashboard
streamlit run app.py

# Generate the Jupyter notebook
python create_notebook.py

# Generate the PowerPoint presentation
python create_ppt.py
```

The app will open at http://localhost:8501

---

## 12. Google Colab Setup

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Upload notebook** → select `MediFlowAI_Notebook.ipynb`
3. Click **Runtime → Run all** to execute all cells
4. Once complete, click **Share** (top-right corner)
5. Under **General access** → click **Change** → select **Anyone with the link**
6. Set role as **Viewer**
7. Click **Copy link**
8. Replace `[PASTE_ACCESSIBLE_COLAB_LINK_HERE]` in:
   - `README.md`
   - `app.py` (COLAB_LINK variable)
   - `create_ppt.py` (COLAB_LINK variable)
   - `slides/ppt_content.md`

---

## 13. GitHub Upload Steps

```bash
# Initialise git repository
git init

# Add all files
git add app.py requirements.txt README.md .gitignore
git add src/ data/ outputs/ slides/ create_notebook.py create_ppt.py MediFlowAI_Notebook.ipynb

# Create initial commit
git commit -m "Initial commit - MediFlow AI RDMU prototype"

# Set main branch
git branch -M main

# Add remote (replace with your GitHub username and repo name)
git remote add origin https://github.com/YOUR_USERNAME/MediFlowAI.git

# Push to GitHub
git push -u origin main
```

After pushing, copy the repository URL and replace `[PASTE_GITHUB_REPO_LINK_HERE]`.

---

## 14. Streamlit Cloud Deployment Steps

1. Push all files to GitHub (see step 13 above)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with your GitHub account
4. Click **Create app** / **New app**
5. Select your **GitHub repository** from the dropdown
6. Select branch: **main**
7. Main file path: **app.py**
8. Click **Deploy**
9. Wait for deployment (2-3 minutes)
10. Copy the deployed app URL (e.g., `https://yourapp.streamlit.app`)
11. Replace `[PASTE_STREAMLIT_APP_LINK_HERE]` in README, app.py, create_ppt.py

---

## 15. How to Replace Link Placeholders in PPT

After completing steps 12, 13, and 14:

1. Run `python create_ppt.py` again after updating the link variables at the top of the script:
   ```python
   COLAB_LINK = "https://colab.research.google.com/drive/YOUR_NOTEBOOK_ID"
   GITHUB_LINK = "https://github.com/YOUR_USERNAME/MediFlowAI"
   STREAMLIT_LINK = "https://YOUR_APP_NAME.streamlit.app"
   ```
2. Or open the generated `slides/MediFlowAI_RDMU_Final_Presentation.pptx` in PowerPoint/Google Slides
3. Use Find & Replace (Ctrl+H) to replace all three placeholder strings with the real URLs

---

## 16. Dashboard Testing Checklist

- [ ] App opens at localhost:8501
- [ ] All 10 sidebar pages are accessible
- [ ] Synthetic data is generated (data/patients.csv, data/resources.csv)
- [ ] Data Preview page shows patient and resource tables
- [ ] Simulator runs and produces timeline chart
- [ ] RL training runs and reward curve is displayed
- [ ] Policy heatmap renders correctly
- [ ] Optimisation baseline produces allocation table
- [ ] Policy comparison shows side-by-side metrics
- [ ] Sensitivity analysis generates charts
- [ ] Metrics heatmap displays correctly
- [ ] CSV export buttons work
- [ ] RDMU Concept Mapping table is complete
- [ ] Jupyter notebook runs all cells without errors
- [ ] PPT has exactly 10 slides with link placeholders
- [ ] Streamlit deployment succeeds after GitHub push

---

## 17. Troubleshooting

| Issue | Fix |
|---|---|
| `streamlit: command not found` | Run `pip install streamlit` or activate venv first |
| `ModuleNotFoundError: No module named 'src'` | Run from MediFlowAI/ root directory: `streamlit run app.py` |
| CSV files missing | Click "Regenerate Synthetic Hospital Data" on Data Preview page |
| Charts not showing | Ensure matplotlib is installed: `pip install matplotlib` |
| GitHub push fails | Check your remote URL: `git remote -v` and re-add with correct URL |
| Streamlit deployment fails | Check requirements.txt lists all imports; remove `seaborn` if present |
| Colab cannot access files | Files are generated in-cell — notebook is self-contained, no local file needed |
| PPT links still show placeholders | Update COLAB_LINK/GITHUB_LINK/STREAMLIT_LINK in create_ppt.py and re-run |

---

## 18. Limitations

- Synthetic data only — not validated against real clinical outcomes
- Tabular Q-learning does not scale to continuous state spaces
- Deterministic optimiser; no formal LP/IP solver
- No authentication, audit logging, or regulatory compliance
- Not production-ready; for educational purposes only

---

## 19. Future Scope

- Deep RL (DQN/PPO) for continuous state spaces
- Real hospital data integration via HL7 FHIR APIs
- Formal optimisation using PuLP or OR-Tools
- Explainable AI (SHAP values) for clinical transparency
- Federated learning across hospital networks
- Dashboard authentication and audit logging

---

## 20. Academic Disclaimer

This software prototype was developed as a final examination submission for
MAIB DSC 103 — Reasoning and Decision Making under Uncertainty,
SP Jain School of Global Management, MAIB September 2025 cohort.
It is intended for educational demonstration only and is not suitable for
clinical or operational use without substantial further development, validation,
and regulatory approval.

---

*Made by Krishna Mathur | AS25DXB018 | MAIB September*
