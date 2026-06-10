"""
create_ppt.py
Generates slides/MediFlowAI_RDMU_Final_Presentation.pptx
Run: python create_ppt.py
Made by: Krishna Mathur | AS25DXB018 | MAIB September
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pathlib import Path

Path("slides").mkdir(exist_ok=True)

COLAB_LINK = "[PASTE_ACCESSIBLE_COLAB_LINK_HERE]"
GITHUB_LINK = "https://github.com/krish2105/mediflow-ai-rdmu-final"
STREAMLIT_LINK = "[PASTE_STREAMLIT_APP_LINK_HERE]"

FOOTER_TEXT = "Made by Krishna Mathur  |  AS25DXB018  |  MAIB September  |  MAIB DSC 103"
LINK_FOOTER = f"Colab: {COLAB_LINK}   GitHub: {GITHUB_LINK}   App: {STREAMLIT_LINK}"

# ── Colors ──────────────────────────────────────────────────────────────────
DARK_BLUE = RGBColor(0x1E, 0x3A, 0x5F)      # navy
MED_BLUE  = RGBColor(0x25, 0x63, 0xEB)      # button blue
GREEN     = RGBColor(0x16, 0xA3, 0x4A)
WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GREY= RGBColor(0xF1, 0xF5, 0xF9)
ACCENT    = RGBColor(0xDC, 0x26, 0x26)      # red


def new_prs():
    prs = Presentation()
    prs.slide_width  = Inches(13.33)
    prs.slide_height = Inches(7.5)
    return prs


def _set_bg(slide, color):
    from pptx.util import Pt
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def _add_text_box(slide, text, left, top, width, height,
                  size=18, bold=False, color=WHITE, align=PP_ALIGN.LEFT,
                  wrap=True, italic=False):
    txBox = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    tf = txBox.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return txBox


def _add_bullet_box(slide, bullets, left, top, width, height,
                    size=15, color=WHITE, title=None):
    txBox = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    tf = txBox.text_frame
    tf.word_wrap = True
    first = True
    if title:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        run = p.add_run()
        run.text = title
        run.font.bold = True
        run.font.size = Pt(size + 2)
        run.font.color.rgb = RGBColor(0xFA, 0xCC, 0x15)
        first = False
    for bullet in bullets:
        p = tf.paragraphs[0] if (first and not title) else tf.add_paragraph()
        run = p.add_run()
        run.text = f"  •  {bullet}"
        run.font.size = Pt(size)
        run.font.color.rgb = color
        first = False
    return txBox


def _footer(slide, show_links=False):
    _add_text_box(slide, FOOTER_TEXT, 0.3, 7.10, 12.73, 0.35,
                  size=9, color=RGBColor(0xCB, 0xD5, 0xE1), align=PP_ALIGN.CENTER)
    if show_links:
        _add_text_box(slide, LINK_FOOTER, 0.3, 7.00, 12.73, 0.30,
                      size=8, italic=True, color=RGBColor(0x93, 0xC5, 0xFD), align=PP_ALIGN.CENTER)


def _header_bar(slide, text, sub=None):
    """Dark blue header bar at top."""
    bar = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        Inches(0), Inches(0), Inches(13.33), Inches(1.35)
    )
    bar.fill.solid(); bar.fill.fore_color.rgb = DARK_BLUE
    bar.line.fill.background()
    _add_text_box(slide, text, 0.25, 0.08, 12.8, 0.75,
                  size=26, bold=True, color=WHITE)
    if sub:
        _add_text_box(slide, sub, 0.25, 0.82, 12.8, 0.45,
                      size=13, italic=True, color=RGBColor(0x93, 0xC5, 0xFD))


# ════════════════════════════════════════════════════════════════════════════
# Slide Builders
# ════════════════════════════════════════════════════════════════════════════

def slide_title(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    _set_bg(slide, DARK_BLUE)

    # Accent bar
    bar = slide.shapes.add_shape(1, Inches(0), Inches(2.4), Inches(13.33), Inches(0.07))
    bar.fill.solid(); bar.fill.fore_color.rgb = MED_BLUE
    bar.line.fill.background()

    _add_text_box(slide, "MediFlow AI",
                  0.5, 0.5, 12.3, 1.2, size=44, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    _add_text_box(slide,
                  "Dynamic Hospital Emergency Resource Allocation using Reinforcement Learning",
                  0.5, 1.5, 12.3, 0.9, size=18, italic=True,
                  color=RGBColor(0x93, 0xC5, 0xFD), align=PP_ALIGN.CENTER)

    details = [
        "Course   :  Reasoning and Decision Making under Uncertainty — MAIB DSC 103",
        "Student  :  Krishna Mathur   |   Student ID: AS25DXB018   |   MAIB September",
        "Topic    :  Topic 1 — Dynamic Resource Allocation using Reinforcement Learning",
        "Date     :  June 2026",
    ]
    for i, line in enumerate(details):
        _add_text_box(slide, line, 1.0, 2.65 + i * 0.52, 11.3, 0.48,
                      size=14, color=RGBColor(0xE2, 0xE8, 0xF0))

    _add_text_box(slide, f"Google Colab  :  {COLAB_LINK}", 0.5, 5.10, 12.3, 0.40,
                  size=12, italic=True, color=RGBColor(0x86, 0xEF, 0xAC))
    _add_text_box(slide, f"GitHub Repo   :  {GITHUB_LINK}", 0.5, 5.52, 12.3, 0.40,
                  size=12, italic=True, color=RGBColor(0x86, 0xEF, 0xAC))
    _add_text_box(slide, f"Streamlit App :  {STREAMLIT_LINK}", 0.5, 5.94, 12.3, 0.40,
                  size=12, italic=True, color=RGBColor(0x86, 0xEF, 0xAC))
    _footer(slide)
    return slide


def slide_requirement(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_bg(slide, RGBColor(0x0F, 0x17, 0x2A))
    _header_bar(slide, "Requirement Understanding",
                "MAIB DSC 103 — Topic 1: Dynamic Resource Allocation using RL")
    bullets = [
        "Build a decision-making software prototype using at least 4 RDMU key concepts",
        "Selected Topic: Topic 1 — Hospital emergency resource allocation under uncertainty",
        "4 RDMU Concepts: (1) Uncertainty  (2) State-Action-Reward  (3) Q-Learning  (4) Constraint Optimisation",
        "Simulate patient arrivals, limited hospital resources, and allocation under constraints",
        "Compare RL-driven vs. constraint-based allocation strategies",
        "Submission: Google Colab notebook  +  GitHub repo  +  Streamlit app  +  PPT",
    ]
    _add_bullet_box(slide, bullets, 0.5, 1.45, 12.3, 5.5, size=15)
    _footer(slide)
    return slide


def slide_business_problem(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_bg(slide, RGBColor(0x0F, 0x17, 0x2A))
    _header_bar(slide, "Business Problem", "Why Hospital Resource Allocation Needs AI")
    bullets = [
        "Emergency departments deal with unpredictable patient arrivals every hour (Uncertainty)",
        "Limited resources — beds, doctors, nurses, equipment — cannot serve all patients at once",
        "Manual triage is slow, subjective, and does not optimise across multiple constraints",
        "Poor allocation decisions increase patient waiting time and raise health risk",
        "Fairness must be maintained — all demographic groups deserve equal access to care",
        "AI-driven RL allocation learns adaptive policies that improve outcomes over time",
    ]
    _add_bullet_box(slide, bullets, 0.5, 1.45, 12.3, 5.5, size=15)
    _footer(slide)
    return slide


def slide_flowchart(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_bg(slide, RGBColor(0x0F, 0x17, 0x2A))
    _header_bar(slide, "Application Flow Chart", "System Architecture — MediFlow AI")

    flow_text = (
        "Patient Data Input (500 synthetic patients)\n"
        "          ↓\n"
        "Resource Capacity Input  (Beds, Doctors, Nurses, Equipment)\n"
        "          ↓\n"
        "┌─── RL Policy Engine ────┐    ┌─── Constraint-Optimisation ───┐\n"
        "│  Q-Learning Agent       │    │  Baseline (Heuristic Rules)   │\n"
        "│  State → Q-table        │    │  Priority Score + Feasibility │\n"
        "│  → Best Allocation      │    │  → Feasible Allocation Table  │\n"
        "└─────────────────────────┘    └───────────────────────────────┘\n"
        "          └──────────── Policy Comparison ────────────┘\n"
        "                            ↓\n"
        "               Dashboard Metrics  (Fairness, Utilisation, Wait Time)\n"
        "                            ↓\n"
        "                Human Review Decision  (Clinician Override via Streamlit)"
    )
    _add_text_box(slide, flow_text, 0.5, 1.45, 12.3, 5.5,
                  size=13, color=RGBColor(0xE2, 0xE8, 0xF0))
    _footer(slide)
    return slide


def slide_prerequisites(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_bg(slide, RGBColor(0x0F, 0x17, 0x2A))
    _header_bar(slide, "Prerequisites for the Application", "Tools, Libraries, and Environment")
    bullets = [
        "Python 3.8 or later (local or Google Colab)",
        "Libraries: pandas, numpy, matplotlib, streamlit, nbformat, python-pptx, scikit-learn",
        "Install: pip install streamlit pandas numpy scikit-learn matplotlib nbformat python-pptx",
        "Basic understanding of Reinforcement Learning and Markov Decision Processes",
        "GitHub account for code hosting and version control",
        "Streamlit Community Cloud account for free web deployment",
        "No paid APIs, no secret keys, no external databases — fully self-contained",
    ]
    _add_bullet_box(slide, bullets, 0.5, 1.45, 12.3, 5.5, size=15)
    _footer(slide)
    return slide


def slide_data(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_bg(slide, RGBColor(0x0F, 0x17, 0x2A))
    _header_bar(slide, "Description of Data Used", "Synthetic Hospital Data — 500 Patient Records")
    bullets = [
        "patients.csv (500 rows): patient_id, arrival_time, urgency_level (1–5), severity_score, age_group, required_bed, required_doctor, required_nurse, required_equipment, max_safe_wait_time, fairness_group, expected_treatment_time",
        "resources.csv (4 rows): Beds=30, Doctors=10, Nurses=20, Equipment=15 (with emergency buffers)",
        "Patient arrivals follow a Poisson process; urgency follows a weighted discrete distribution",
        "Fairness groups (A=40%, B=35%, C=25%) simulate demographic diversity in patient population",
        "All data generated with NumPy random seed=42 — fully reproducible",
        "Why synthetic: Real patient data requires HIPAA / GDPR compliance and ethical approval",
    ]
    _add_bullet_box(slide, bullets, 0.5, 1.45, 12.3, 5.5, size=14)
    _footer(slide)
    return slide


def slide_rl_design(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_bg(slide, RGBColor(0x0F, 0x17, 0x2A))
    _header_bar(slide, "Reinforcement Learning Design", "Q-Learning Agent — MDP Formulation")
    bullets = [
        "State = (urgency_bucket × 5,  wait_bucket × 4,  capacity_bucket × 3,  fairness_bucket × 2)  →  120 total states",
        "Action = 4 decisions:  Allocate Full Care  |  Stabilize and Queue  |  Queue for Later  |  Transfer / Refer",
        "Reward = +10×urgency for treating; +20 bonus for critical patients (urgency≥4); –30 for queuing critical",
        "Safety penalty: –2× time past max-safe-wait-limit; +5 fairness bonus for underserved groups",
        "Bellman Update: Q(s,a) ← Q(s,a) + α [ r + γ · max Q(s') − Q(s,a) ]",
        "Hyperparameters: α=0.10, γ=0.90, ε=0.30 (decays to 0.01 over 500 episodes)",
    ]
    _add_bullet_box(slide, bullets, 0.5, 1.45, 12.3, 5.5, size=14)
    _footer(slide)
    return slide


def slide_constraint_fairness(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_bg(slide, RGBColor(0x0F, 0x17, 0x2A))
    _header_bar(slide, "Constraint Optimisation and Fairness", "Deterministic Baseline + Fairness Trade-Off")
    bullets = [
        "Baseline allocates patients via weighted priority score (urgency + fairness + wait time − resource burden)",
        "Hard constraints: total used resources must never exceed available_count for beds, doctors, nurses, equipment",
        "Emergency buffer: 5 beds and 2 doctors reserved exclusively for critical patients (urgency ≥ 4)",
        "Fairness weight slider controls the trade-off: higher fairness → more equitable care, lower throughput",
        "Jain's Fairness Index measures equal treatment: (Σxi)² / (n · Σxi²)  — target ≥ 0.90",
        "Sensitivity analysis sweeps fairness weight 0.05→0.90 to reveal the efficiency–fairness Pareto frontier",
    ]
    _add_bullet_box(slide, bullets, 0.5, 1.45, 12.3, 5.5, size=14)
    _footer(slide)
    return slide


def slide_dashboard(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_bg(slide, RGBColor(0x0F, 0x17, 0x2A))
    _header_bar(slide, "User Interface — Streamlit Dashboard",
                "10-Page Interactive Decision Support System")
    pages = [
        "1. Executive Dashboard — KPI cards, urgency pie chart, resource bar chart, project links",
        "2. Data Preview — patients.csv and resources.csv preview, regenerate data button",
        "3. Resource Allocation Simulator — Time-step simulation with adjustable arrival rate",
        "4. RL Policy Engine — Q-learning training, reward curve, policy heatmap, recommendations",
        "5. Optimisation Baseline — Constraint-based allocation with fairness/urgency/efficiency weights",
        "6. Policy Comparison — RL vs Optimiser side-by-side metrics and bar chart",
        "7. Sensitivity Analysis — Fairness weight sweep, trade-off scatter charts",
        "8. Metrics and Performance — Full metrics heatmap, KPIs, export CSV",
        "9. RDMU Concept Mapping — All 10 concepts mapped to app features",
        "10. Limitations and Future Scope — Academic disclaimer and improvement roadmap",
    ]
    _add_bullet_box(slide, pages, 0.5, 1.45, 12.3, 5.3, size=13)
    _add_text_box(slide, f"App: {STREAMLIT_LINK}   |   GitHub: {GITHUB_LINK}",
                  0.5, 6.80, 12.3, 0.35, size=10, italic=True,
                  color=RGBColor(0x86, 0xEF, 0xAC))
    _footer(slide, show_links=True)
    return slide


def slide_results_limitations(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_bg(slide, RGBColor(0x0F, 0x17, 0x2A))
    _header_bar(slide, "Results, Limitations, and Future Scope",
                "Expected Outputs + Academic Disclaimer")
    results = [
        "RL agent converges within ~300 episodes; learns to always prioritise critical patients",
        "Sensitivity analysis: fairness_weight ↑ → fairness_index ↑, resource_utilisation ↓",
        "Optimiser: deterministic, always feasible; RL: adaptive, learns fairness-aware policies",
    ]
    limits = [
        "Synthetic data only — not validated against real clinical outcomes",
        "Tabular Q-learning does not scale to continuous state spaces",
        "No formal LP/IP solver; heuristic constraint handling used",
    ]
    future = [
        "Deep RL (DQN/PPO) for large continuous state spaces",
        "Real hospital data integration via HL7 FHIR APIs",
        "Formal optimisation with PuLP / OR-Tools; Explainable AI (SHAP values)",
    ]
    _add_bullet_box(slide, results, 0.5, 1.50, 12.3, 1.3, size=13,
                    title="Key Results")
    _add_bullet_box(slide, limits, 0.5, 3.05, 6.0, 1.5, size=12,
                    title="Limitations")
    _add_bullet_box(slide, future, 6.7, 3.05, 5.8, 1.5, size=12,
                    title="Future Scope")
    _footer(slide, show_links=True)
    return slide


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
def build_pptx():
    prs = new_prs()
    slide_title(prs)
    slide_requirement(prs)
    slide_business_problem(prs)
    slide_flowchart(prs)
    slide_prerequisites(prs)
    slide_data(prs)
    slide_rl_design(prs)
    slide_constraint_fairness(prs)
    slide_dashboard(prs)
    slide_results_limitations(prs)

    out = "slides/MediFlowAI_RDMU_Final_Presentation.pptx"
    prs.save(out)
    print(f"PPTX saved to {out}")
    print(f"Total slides: {len(prs.slides)}")
    return out


if __name__ == "__main__":
    build_pptx()
