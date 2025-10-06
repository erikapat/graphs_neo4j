# policy_assignment_full_demo_savefigs.py
# ------------------------------------------------------------
# Synthetic, runnable demo of policy-to-agency assignment with PuLP.
# Saves all plots to a timestamped output folder under ./figs/
#
# Run:
#   python policy_assignment_full_demo_savefigs.py
#
# Requirements:
#   pip install pulp==3.3.0 pandas==2.3.2 matplotlib==3.10.6 numpy
# ------------------------------------------------------------

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pulp as pl
from datetime import datetime

# ---------- Output folder ----------
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join("figs", RUN_ID)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_fig(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")

# ---------- Synthetic inputs ----------
AGENCIES = ["A1", "A2", "A3", "A4"]           # rows
BUCKETS  = ["Gold", "Silver", "Bronze"]           # cols

# Per-agency productivity q_a (used by q^T X 1 objective)
q = {"A1": 10, "A2": 7, "A3": 5, "A4": 8}

# Per-agency capacities U_a (slots available now)
U = {"A1": 4, "A2": 3, "A3": 1, "A4": 2}      # sum = 10
Utot = sum(U.values())

# Floors L_a for live mode (will be overridden with baseline totals to prevent negative deltas)
L_seed = {"A1": 3, "A2": 2, "A3": 1, "A4": 1}

# ZIP admissibility for the incoming policy (live mode)
res_zip = "28001"
ziplist = {"A1": {"28001", "28002"}, "A2": None, "A3": {"28003"}, "A4": {"28004"}}
def zip_admissible(a: str) -> bool:
    z = ziplist[a]
    return True if z in (None, set()) else (res_zip in z)

# Bucket-specific weights W_{a,c} to create variety (used by <W,X> objective)
random.seed(42)
bucket_factor = {"Bronze": 0.85, "Silver": 1.00, "Gold": 1.20}
def Wac(a, c):
    return q[a] * bucket_factor[c] + 0.05*random.random()  # small noise to avoid ties
W = {(a, c): Wac(a, c) for a in AGENCIES for c in BUCKETS}

# Optional: global bucket mix constraints (to force spread across columns)
L_bucket = {"Bronze": 2, "Silver": 3, "Gold": 2}   # mins (sum <= Utot)
U_bucket = {"Bronze": 6, "Silver": 7, "Gold": 7}   # maxes (>= Utot+1 collectively)

# ---------- Vectorization helpers for matrices ----------
VAR_COLS = [(a, c) for a in AGENCIES for c in BUCKETS]
n = len(VAR_COLS)
colnames = [f"x_{a}_{c}" for (a, c) in VAR_COLS]

def row_for_agency(a_target: str, sign: int = +1):
    """Row with 1 (or -1) in columns belonging to agency a_target across all buckets, else 0."""
    row = np.zeros(n, dtype=int)
    for j, (a, c) in enumerate(VAR_COLS):
        if a == a_target:
            row[j] = sign
    return row

def row_for_bucket(c_target: str, sign: int = +1):
    """Row with 1 (or -1) in columns belonging to bucket c_target across all agencies, else 0."""
    row = np.zeros(n, dtype=int)
    for j, (a, c) in enumerate(VAR_COLS):
        if c == c_target:
            row[j] = sign
    return row

def ones_row():
    return np.ones(n, dtype=int)

# ---------- Build constraint matrices ----------
def build_constraint_matrices(sum_inv: int, L_use: dict, use_bucket_bounds: bool = True):
    """
    Returns (A_ineq, b_vec, Aeq_mat, beq_vec)
    sum_inv = 0 -> baseline; = 1 -> live (+1 policy)
    L_use: floors per agency used in live mode (ignored in baseline)
    """
    A_rows, b_vals = [], []
    Aeq_rows, beq_vals = [], []

    # Per-agency capacity with headroom in live (only impacts feasibility)
    for a in AGENCIES:
        headroom = 1 if (sum_inv == 1 and U[a] > 0) else 0
        A_rows.append(row_for_agency(a, +1))
        b_vals.append(U[a] + headroom)

    # Global conservation
    Aeq_rows.append(ones_row())
    beq_vals.append(Utot + sum_inv)

    # Bucket mix (optional)
    if use_bucket_bounds:
        for c in BUCKETS:
            # min per bucket: sum_a x_{a,c} >= L_bucket[c]  ->  -sum_a x_{a,c} <= -L_bucket[c]
            A_rows.append(row_for_bucket(c, -1))
            b_vals.append(-L_bucket[c])
            # max per bucket: sum_a x_{a,c} <= U_bucket[c]
            A_rows.append(row_for_bucket(c, +1))
            b_vals.append(U_bucket[c])

    # Live-only floors and ZIP locking
    if sum_inv == 1:
        for a in AGENCIES:
            # floors: sum_c x_{a,c} >= L_use[a]  ->  -sum_c x_{a,c} <= -L_use[a]
            A_rows.append(row_for_agency(a, -1))
            b_vals.append(-L_use[a])
        for a in AGENCIES:
            if not zip_admissible(a):
                Aeq_rows.append(row_for_agency(a, +1))
                beq_vals.append(L_use[a])

    A_ineq = np.vstack(A_rows) if A_rows else np.zeros((0, n), dtype=int)
    b_vec  = np.array(b_vals, dtype=int)
    Aeq_mat = np.vstack(Aeq_rows) if Aeq_rows else np.zeros((0, n), dtype=int)
    beq_vec = np.array(beq_vals, dtype=int)
    return A_ineq, b_vec, Aeq_mat, beq_vec

# ---------- Solve with PuLP ----------
def solve(sum_inv: int, L_override: dict | None = None, use_W: bool = True, use_bucket_bounds: bool = True):
    """
    sum_inv: 0 baseline, 1 live
    L_override: if provided in live mode, floors are taken from this dict (enforces no negative deltas)
    use_W: objective uses W_{a,c} (variation). If False, uses q_a only.
    """
    L_use = L_override if (sum_inv == 1 and L_override is not None) else L_seed

    prob = pl.LpProblem("Assignment", pl.LpMaximize)
    x = {(a, c): pl.LpVariable(f"x_{a}_{c}", lowBound=0, cat=pl.LpInteger)
         for a in AGENCIES for c in BUCKETS}

    # Objective
    if use_W:
        prob += pl.lpSum(W[(a, c)] * x[(a, c)] for a in AGENCIES for c in BUCKETS), "Obj_WdotX"
    else:
        prob += pl.lpSum(q[a] * x[(a, c)] for a in AGENCIES for c in BUCKETS), "Obj_qTX1"

    # Per-agency capacity (+1 headroom in live if U[a]>0)
    for a in AGENCIES:
        headroom = 1 if (sum_inv == 1 and U[a] > 0) else 0
        prob += pl.lpSum(x[(a, c)] for c in BUCKETS) <= U[a] + headroom, f"Cap_{a}"

    # Global conservation
    prob += pl.lpSum(x[(a, c)] for a in AGENCIES for c in BUCKETS) == Utot + sum_inv, "Global"

    # Bucket mix (optional)
    if use_bucket_bounds:
        for c in BUCKETS:
            prob += pl.lpSum(x[(a, c)] for a in AGENCIES) >= L_bucket[c], f"BucketMin_{c}"
            prob += pl.lpSum(x[(a, c)] for a in AGENCIES) <= U_bucket[c], f"BucketMax_{c}"

    # Live-only: floors and ZIP lock
    if sum_inv == 1:
        for a in AGENCIES:
            prob += pl.lpSum(x[(a, c)] for c in BUCKETS) >= L_use[a], f"Floor_{a}"
        for a in AGENCIES:
            if not zip_admissible(a):
                prob += pl.lpSum(x[(a, c)] for c in BUCKETS) == L_use[a], f"ZIP_{a}_eq"

    status = prob.solve(pl.PULP_CBC_CMD(msg=False))
    status_str = pl.LpStatus[status]

    # Extract solution
    X = pd.DataFrame(0, index=AGENCIES, columns=BUCKETS, dtype=int)
    for a in AGENCIES:
        for c in BUCKETS:
            val = x[(a, c)].value()
            X.loc[a, c] = int(val) if val is not None else 0

    totals = X.sum(axis=1)
    obj_q = sum(q[a] * totals[a] for a in AGENCIES)
    obj_W = sum(W[(a, c)] * X.loc[a, c] for a in AGENCIES for c in BUCKETS)
    return status_str, X, totals, obj_q, obj_W

# ---------- Plot helpers ----------
def heatmap_matrix(M: pd.DataFrame, title: str, filename: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(M.values, aspect="auto")
    ax.set_xticks(range(len(M.columns))); ax.set_xticklabels(M.columns)
    ax.set_yticks(range(len(M.index)));   ax.set_yticklabels(M.index)
    ax.set_title(title)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{int(M.iat[i, j])}", ha="center", va="center")
    plt.tight_layout()
    save_fig(fig, filename)

def heatmap_W(W_mat: pd.DataFrame, title: str, filename: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(W_mat.values, aspect="auto")
    ax.set_xticks(range(len(W_mat.columns))); ax.set_xticklabels(W_mat.columns)
    ax.set_yticks(range(len(W_mat.index)));   ax.set_yticklabels(W_mat.index)
    ax.set_title(title)
    for i in range(W_mat.shape[0]):
        for j in range(W_mat.shape[1]):
            ax.text(j, i, f"{W_mat.iat[i, j]:.1f}", ha="center", va="center")
    plt.tight_layout()
    save_fig(fig, filename)

def stacked_by_bucket(X: pd.DataFrame, title: str, filename: str):
    bottom = np.zeros(len(AGENCIES))
    fig, ax = plt.subplots(figsize=(8, 4))
    for c in BUCKETS:
        vals = X[c].values
        ax.bar(AGENCIES, vals, bottom=bottom, label=c)
        bottom += vals
    ax.set_title(title); ax.set_xlabel("Agency"); ax.set_ylabel("Policies"); ax.legend()
    plt.tight_layout()
    save_fig(fig, filename)

# ---------- Main ----------
def main():
    print(f"Saving figures to: {OUTPUT_DIR}")

    # Show weights
    q_vec = pd.Series(q, name="q_a")
    W_mat = pd.DataFrame({c: [W[(a, c)] for a in AGENCIES] for c in BUCKETS}, index=AGENCIES)
    print("\n=== Productivity vector q ===")
    print(q_vec)
    print("\n=== Productivity matrix W (bucket-adjusted) ===")
    print(W_mat.round(2))
    heatmap_W(W_mat, "Productivity matrix W (bucket-adjusted)", "W_matrix_heatmap.png")

    # ----- BASELINE -----
    st0, X0, tot0, obj0_q, obj0_W = solve(sum_inv=0, use_W=True, use_bucket_bounds=True)
    print("\n=== BASELINE (sum_inv=0) ===")
    print("Status:", st0)
    print("Assignment X0:\n", X0)
    print("Totals:", tot0.to_dict())
    print("Objective (q):", round(obj0_q, 3), " Objective (W):", round(obj0_W, 3))
    heatmap_matrix(X0, "Baseline X (varied)", "baseline_X_heatmap.png")
    stacked_by_bucket(X0, "Baseline: per-agency bucket mix", "baseline_bucket_mix.png")

    # Floors from baseline totals to prevent negative deltas in live
    L_from_baseline = {a: int(tot0[a]) for a in AGENCIES}

    # ----- LIVE (+1) -----
    st1, X1, tot1, obj1_q, obj1_W = solve(sum_inv=1, L_override=L_from_baseline,
                                          use_W=True, use_bucket_bounds=True)
    print("\n=== LIVE (sum_inv=1) ===")
    print("Status:", st1)
    print("Assignment X1:\n", X1)
    print("Totals:", tot1.to_dict())
    print("Objective (q):", round(obj1_q, 3), " Objective (W):", round(obj1_W, 3))

    # Deltas (must be non-negative and sum to 1)
    delta = (tot1 - tot0).astype(int)
    print("\nDelta (Live - Baseline):", delta.to_dict(),
          "  Obj Δ (q):", round(obj1_q - obj0_q, 3),
          "  Obj Δ (W):", round(obj1_W - obj0_W, 3))
    assert all(d >= 0 for d in delta.values), f"Negative deltas found: {delta.to_dict()}"
    assert int(delta.sum()) == 1, f"Delta sum is not 1: {int(delta.sum())}"

    heatmap_matrix(X1, "Live X (varied, +1)", "live_X_heatmap.png")
    stacked_by_bucket(X1, "Live: per-agency bucket mix (+1)", "live_bucket_mix.png")

    print("\nDone.")

if __name__ == "__main__":
    main()
