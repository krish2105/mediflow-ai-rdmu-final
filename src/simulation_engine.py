"""
simulation_engine.py
Hospital emergency department time-step simulation engine.
Made by: Krishna Mathur | AS25DXB018 | MAIB September
"""
import pandas as pd
import numpy as np


def run_simulation(patients_df, resources_df, arrival_rate=1.0, n_steps=100, seed=42):
    """
    Simulate patient flow over n_steps time steps.
    arrival_rate controls mean Poisson arrivals per step.
    Returns a timeline DataFrame with per-step metrics.
    """
    np.random.seed(seed)

    res = resources_df.set_index("resource_name")
    total_beds = int(res.loc["Beds", "available_count"])
    total_doctors = int(res.loc["Doctors", "available_count"])
    total_nurses = int(res.loc["Nurses", "available_count"])
    total_equipment = int(res.loc["Equipment", "available_count"])

    current = {
        "beds": total_beds,
        "doctors": total_doctors,
        "nurses": total_nurses,
        "equipment": total_equipment,
    }
    totals = {k: v for k, v in current.items()}

    queue = []
    patient_idx = 0
    n_patients = len(patients_df)
    timeline = []
    cumulative_treated = 0

    for step in range(n_steps):
        # --- Arrivals ---
        n_arrivals = np.random.poisson(arrival_rate)
        for _ in range(min(n_arrivals, n_patients - patient_idx)):
            if patient_idx < n_patients:
                queue.append({
                    **patients_df.iloc[patient_idx].to_dict(),
                    "arrival_step": step,
                    "wait_steps": 0,
                })
                patient_idx += 1

        # Increment wait for everyone in queue
        for p in queue:
            p["wait_steps"] += 1

        # --- Triage sort: urgency DESC then wait DESC ---
        queue.sort(key=lambda x: (-x["urgency_level"], -x["wait_steps"]))

        # --- Allocate resources ---
        still_queued = []
        step_treated = 0
        for p in queue:
            need_bed = int(p.get("required_bed", 1))
            need_doc = int(p.get("required_doctor", 1))
            need_nur = int(p.get("required_nurse", 1))
            need_equ = int(p.get("required_equipment", 0))

            if (current["beds"] >= need_bed and
                    current["doctors"] >= need_doc and
                    current["nurses"] >= need_nur and
                    current["equipment"] >= need_equ):
                current["beds"] -= need_bed
                current["doctors"] -= need_doc
                current["nurses"] -= need_nur
                current["equipment"] -= need_equ
                step_treated += 1
                cumulative_treated += 1
            else:
                still_queued.append(p)

        queue = still_queued

        # --- Discharge / release resources ---
        n_discharge = np.random.poisson(1.5)
        current["beds"] = min(totals["beds"], current["beds"] + n_discharge)
        current["doctors"] = min(totals["doctors"], current["doctors"] + max(0, n_discharge // 2))
        current["nurses"] = min(totals["nurses"], current["nurses"] + n_discharge)
        current["equipment"] = min(totals["equipment"], current["equipment"] + max(0, n_discharge // 3))

        avg_wait = float(np.mean([p["wait_steps"] for p in queue])) if queue else 0.0

        timeline.append({
            "step": step,
            "queue_length": len(queue),
            "step_treated": step_treated,
            "cumulative_treated": cumulative_treated,
            "bed_utilization": round(1 - current["beds"] / max(totals["beds"], 1), 3),
            "doctor_utilization": round(1 - current["doctors"] / max(totals["doctors"], 1), 3),
            "nurse_utilization": round(1 - current["nurses"] / max(totals["nurses"], 1), 3),
            "avg_wait_time": round(avg_wait, 2),
            "available_beds": current["beds"],
            "available_doctors": current["doctors"],
        })

    return pd.DataFrame(timeline)
