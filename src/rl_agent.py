"""
rl_agent.py
Q-Learning agent for hospital resource allocation.
RDMU Concepts: State, Action, Reward, Policy, Reinforcement Learning.
Made by: Krishna Mathur | AS25DXB018 | MAIB September
"""
import numpy as np
import pandas as pd


ACTION_NAMES = ["Allocate Full Care", "Stabilize and Queue", "Queue for Later", "Transfer / Refer"]


class QLearningAgent:
    """
    Tabular Q-learning agent.
    State = (urgency_bucket[5], wait_bucket[4], capacity_bucket[3], fairness_bucket[2])
    Actions = 4 allocation decisions.
    """

    N_URGENCY = 5
    N_WAIT = 4
    N_CAPACITY = 3
    N_FAIRNESS = 2
    N_ACTIONS = 4

    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.3, n_episodes=500):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_episodes = n_episodes
        self.q_table = np.zeros((
            self.N_URGENCY, self.N_WAIT, self.N_CAPACITY, self.N_FAIRNESS, self.N_ACTIONS
        ))
        self.reward_history = []

    # ------------------------------------------------------------------
    # State discretisation
    # ------------------------------------------------------------------
    def discretize_state(self, urgency, wait_time, capacity_pressure, fairness_pressure):
        urgency_b = int(np.clip(int(urgency) - 1, 0, 4))

        if wait_time < 15:
            wait_b = 0
        elif wait_time < 30:
            wait_b = 1
        elif wait_time < 60:
            wait_b = 2
        else:
            wait_b = 3

        if capacity_pressure < 0.5:
            cap_b = 0
        elif capacity_pressure < 0.8:
            cap_b = 1
        else:
            cap_b = 2

        fair_b = 1 if fairness_pressure > 0.5 else 0
        return (urgency_b, wait_b, cap_b, fair_b)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def choose_action(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.N_ACTIONS)
        return int(np.argmax(self.q_table[state]))

    # ------------------------------------------------------------------
    # Reward function
    # ------------------------------------------------------------------
    def compute_reward(self, urgency, wait_time, action, capacity_pressure, fairness_pressure):
        """
        Reward encourages treating high-urgency patients quickly,
        penalises capacity violations and unsafe delays.
        """
        reward = 0.0

        if action == 0:  # Allocate Full Care
            reward += urgency * 10.0
            if urgency >= 4:
                reward += 20.0
            if capacity_pressure > 0.8:
                reward -= 15.0   # over-allocation at high load
        elif action == 1:  # Stabilize and Queue
            reward += urgency * 4.0
        elif action == 2:  # Queue for Later
            if urgency >= 4:
                reward -= 30.0
            reward -= wait_time * 0.5
        else:              # Transfer / Refer
            if urgency >= 4:
                reward -= 20.0
            reward -= 10.0

        # Fairness bonus: reward treating underserved groups
        if fairness_pressure > 0.5 and action in (0, 1):
            reward += 5.0

        # Unsafe wait penalty
        safe_limit = max(10, 100 - urgency * 20)
        if wait_time > safe_limit:
            reward -= (wait_time - safe_limit) * 2.0

        return reward

    # ------------------------------------------------------------------
    # Bellman update
    # ------------------------------------------------------------------
    def update(self, state, action, reward, next_state):
        old_q = self.q_table[state][action]
        best_next = float(np.max(self.q_table[next_state]))
        self.q_table[state][action] = old_q + self.alpha * (
            reward + self.gamma * best_next - old_q
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train(self, patients_df, resources_df):
        """Train Q-learning on synthetic patient episodes. Returns reward history."""
        self.reward_history = []
        n_patients = len(patients_df)

        res = resources_df.set_index("resource_name")
        total_beds = int(res.loc["Beds", "available_count"])

        eps = self.epsilon   # local copy for decay

        for episode in range(self.n_episodes):
            total_reward = 0.0
            current_beds = total_beds
            idx_order = np.random.permutation(n_patients)[:50]

            for idx in idx_order:
                patient = patients_df.iloc[idx]
                urgency = patient["urgency_level"]
                wait_time = float(np.random.uniform(0, 90))
                cap_pressure = 1.0 - current_beds / max(total_beds, 1)
                fair_pressure = float(np.random.uniform(0, 1))

                state = self.discretize_state(urgency, wait_time, cap_pressure, fair_pressure)

                # epsilon-greedy
                if np.random.random() < eps:
                    action = np.random.randint(self.N_ACTIONS)
                else:
                    action = int(np.argmax(self.q_table[state]))

                reward = self.compute_reward(urgency, wait_time, action, cap_pressure, fair_pressure)

                # Simulate resource change
                if action == 0 and current_beds > 0:
                    current_beds = max(0, current_beds - int(patient.get("required_bed", 1)))

                # Next state (time progresses)
                next_wait = max(0.0, wait_time - 5.0)
                next_cap = 1.0 - current_beds / max(total_beds, 1)
                next_state = self.discretize_state(urgency, next_wait, next_cap, fair_pressure)

                self.update(state, action, reward, next_state)
                total_reward += reward

            eps = max(0.01, eps * 0.995)
            self.reward_history.append(total_reward / 50.0)

        self.epsilon = eps
        return self.reward_history

    # ------------------------------------------------------------------
    # Policy extraction
    # ------------------------------------------------------------------
    def get_policy(self):
        """Return dict mapping every state to the best action name."""
        policy = {}
        for u in range(self.N_URGENCY):
            for w in range(self.N_WAIT):
                for c in range(self.N_CAPACITY):
                    for f in range(self.N_FAIRNESS):
                        state = (u, w, c, f)
                        policy[state] = ACTION_NAMES[int(np.argmax(self.q_table[state]))]
        return policy

    def get_policy_df(self):
        """Return policy as a readable DataFrame."""
        urgency_labels = [f"Urgency {i+1}" for i in range(self.N_URGENCY)]
        wait_labels = ["<15 min", "15-30 min", "30-60 min", ">60 min"]
        cap_labels = ["Low (<50%)", "Medium (50-80%)", "High (>80%)"]
        fair_labels = ["Not Underserved", "Underserved"]

        rows = []
        policy = self.get_policy()
        for (u, w, c, f), action in policy.items():
            rows.append({
                "Urgency": urgency_labels[u],
                "Wait Time": wait_labels[w],
                "Capacity Pressure": cap_labels[c],
                "Fairness Pressure": fair_labels[f],
                "Recommended Action": action,
                "Q-Values": str(np.round(self.q_table[(u, w, c, f)], 2).tolist()),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Allocation using learned policy
    # ------------------------------------------------------------------
    def allocate_patients(self, patients_df, resources_df):
        """Apply trained policy to allocate a patient list."""
        res = resources_df.set_index("resource_name")
        current = {
            "beds": int(res.loc["Beds", "available_count"]),
            "doctors": int(res.loc["Doctors", "available_count"]),
            "nurses": int(res.loc["Nurses", "available_count"]),
            "equipment": int(res.loc["Equipment", "available_count"]),
        }
        totals = dict(current)

        group_counts = patients_df["fairness_group"].value_counts().to_dict()
        group_treated = {g: 0 for g in group_counts}

        allocation = []
        for i, (_, patient) in enumerate(patients_df.iterrows()):
            urgency = patient["urgency_level"]
            # Wait time in minutes: ramp 0 → 90 min across the batch (realistic ED queue)
            wait_time = float(min(i * 0.5, 90.0))
            cap_pressure = 1.0 - current["beds"] / max(totals["beds"], 1)

            group = patient["fairness_group"]
            group_rate = group_treated.get(group, 0) / max(group_counts.get(group, 1), 1)
            all_treated = sum(group_treated.values())
            all_total = max(sum(group_counts.values()), 1)
            avg_rate = all_treated / all_total
            fair_pressure = max(0.0, avg_rate - group_rate)

            state = self.discretize_state(urgency, wait_time, cap_pressure, fair_pressure)
            action_idx = int(np.argmax(self.q_table[state]))
            action = ACTION_NAMES[action_idx]

            # Enforce resource feasibility
            need_bed = int(patient.get("required_bed", 1))
            need_doc = int(patient.get("required_doctor", 1))
            need_nur = int(patient.get("required_nurse", 1))
            need_equ = int(patient.get("required_equipment", 0))

            if action == "Allocate Full Care":
                if (current["beds"] >= need_bed and current["doctors"] >= need_doc
                        and current["nurses"] >= need_nur and current["equipment"] >= need_equ):
                    current["beds"] -= need_bed
                    current["doctors"] -= need_doc
                    current["nurses"] -= need_nur
                    current["equipment"] -= need_equ
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
                "patient_id": patient["patient_id"],
                "urgency_level": urgency,
                "fairness_group": group,
                "wait_time": wait_time,
                "action": action,
                "status": status,
                "priority_score": round(urgency / 5.0 - cap_pressure * 0.1, 3),
            })

        return pd.DataFrame(allocation)

    # ------------------------------------------------------------------
    # Single-patient recommendation
    # ------------------------------------------------------------------
    def recommend(self, urgency, wait_time, capacity_pressure, fairness_pressure):
        """Return (action_name, q_values_array) for a single patient."""
        state = self.discretize_state(urgency, wait_time, capacity_pressure, fairness_pressure)
        action_idx = int(np.argmax(self.q_table[state]))
        return ACTION_NAMES[action_idx], self.q_table[state]
