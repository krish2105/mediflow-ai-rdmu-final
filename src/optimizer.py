"""
optimizer.py
Constraint-Optimization Baseline using deterministic feasibility rules.
RDMU Concepts: Constraint-based decision-making, Fairness, Uncertainty handling.
Made by: Krishna Mathur | AS25DXB018 | MAIB September

NOTE: This is NOT a formal LP/IP solver. It implements a deterministic weighted
priority queue with hard resource constraints and emergency buffers — a practical
baseline for comparing against the RL policy.
"""
import pandas as pd
import numpy as np


class ConstraintOptimizer:
    """
    Deterministic feasibility-rule allocator.
    Priority score = urgency_weight * urgency + fairness_weight * fairness_bonus
                     + efficiency_weight * wait_norm - 0.1 * resource_burden
    Hard constraints: available resources must not be exceeded.
    Emergency buffer: reserved for urgency >= 4 patients.
    """

    def __init__(self, fairness_weight=0.30, urgency_weight=0.50, efficiency_weight=0.20):
        self.fairness_weight = fairness_weight
        self.urgency_weight = urgency_weight
        self.efficiency_weight = efficiency_weight

    # ------------------------------------------------------------------
    def _priority(self, patient, wait_time, group_counts, group_treated):
        urgency_norm = patient["urgency_level"] / 5.0
        wait_norm = min(wait_time / 120.0, 1.0)

        # Fairness: underserved group gets higher score
        group = patient["fairness_group"]
        g_total = max(group_counts.get(group, 1), 1)
        g_treated = group_treated.get(group, 0)
        fair_bonus = 1.0 - (g_treated / g_total)

        resource_burden = (
            int(patient.get("required_bed", 1)) +
            int(patient.get("required_doctor", 1)) +
            int(patient.get("required_nurse", 1)) +
            int(patient.get("required_equipment", 0))
        ) / 6.0

        return (
            self.urgency_weight * urgency_norm
            + self.fairness_weight * fair_bonus
            + self.efficiency_weight * wait_norm
            - 0.10 * resource_burden
        )

    # ------------------------------------------------------------------
    def allocate(self, patients_df, resources_df):
        """
        Allocate patients respecting hard resource constraints.
        Returns (allocation_df, resource_summary_dict).
        """
        res = resources_df.set_index("resource_name")

        def _safe_int(name, col):
            try:
                return int(res.loc[name, col])
            except KeyError:
                return 0

        available = {
            "Beds": _safe_int("Beds", "available_count"),
            "Doctors": _safe_int("Doctors", "available_count"),
            "Nurses": _safe_int("Nurses", "available_count"),
            "Equipment": _safe_int("Equipment", "available_count"),
        }
        buffer = {
            "Beds": _safe_int("Beds", "emergency_buffer"),
            "Doctors": _safe_int("Doctors", "emergency_buffer"),
            "Nurses": _safe_int("Nurses", "emergency_buffer"),
            "Equipment": _safe_int("Equipment", "emergency_buffer"),
        }

        group_counts = patients_df["fairness_group"].value_counts().to_dict()
        group_treated = {g: 0 for g in group_counts}

        # Build priority scores for sorting
        scored = []
        for i, (_, patient) in enumerate(patients_df.iterrows()):
            # Wait time in minutes: ramp 0 → 90 min across the batch (realistic ED queue)
            wait_time = min(float(i) * 0.5, 90.0)
            score = self._priority(patient, wait_time, group_counts, group_treated)
            scored.append({**patient.to_dict(), "_score": score, "_wait": wait_time})

        scored.sort(key=lambda x: -x["_score"])

        used = {"Beds": 0, "Doctors": 0, "Nurses": 0, "Equipment": 0}
        capacity_violations = 0
        allocation = []

        for p in scored:
            urgency = p["urgency_level"]
            group = p["fairness_group"]

            # Non-critical patients respect the emergency buffer
            buf_mult = 0 if urgency >= 4 else 1
            eff_avail = {k: available[k] - buf_mult * buffer[k] for k in available}

            need = {
                "Beds": int(p.get("required_bed", 1)),
                "Doctors": int(p.get("required_doctor", 1)),
                "Nurses": int(p.get("required_nurse", 1)),
                "Equipment": int(p.get("required_equipment", 0)),
            }

            can_allocate = all(used[k] + need[k] <= eff_avail[k] for k in need)

            if can_allocate:
                action = "Allocate Full Care"
                status = "Treated"
                for k in need:
                    used[k] += need[k]
                group_treated[group] = group_treated.get(group, 0) + 1
            elif urgency >= 4:
                # Critical — try to at least stabilise
                if used["Doctors"] < available["Doctors"]:
                    action = "Stabilize and Queue"
                    status = "Stabilized"
                    used["Doctors"] += 1
                    capacity_violations += 1
                else:
                    action = "Transfer / Refer"
                    status = "Transferred"
            elif urgency >= 3:
                action = "Stabilize and Queue"
                status = "Queued"
            else:
                action = "Queue for Later"
                status = "Queued"

            allocation.append({
                "patient_id": p["patient_id"],
                "urgency_level": urgency,
                "fairness_group": group,
                "priority_score": round(p["_score"], 3),
                "wait_time": p["_wait"],
                "action": action,
                "status": status,
            })

        alloc_df = pd.DataFrame(allocation)

        resource_summary = {
            "used": used,
            "available": available,
            "utilization": {
                k: round(used[k] / max(available[k], 1), 3)
                for k in used
            },
            "capacity_violations": capacity_violations,
        }
        return alloc_df, resource_summary
