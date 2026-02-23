# optimization_v2.py
# ------------------------------------------------------------
# Policy-to-agency assignment with PuLP + explainability output.
# Saves plots to a timestamped output folder under ./figs/
#
# Run:
#   python optimization_v2.py
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
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "figs", SCRIPT_NAME, RUN_ID)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_fig(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")

# ---------- Synthetic inputs ----------
AGENCIES = ["A1", "A2", "A3", "A4"]           # rows
BUCKETS  = ["Gold", "Silver", "Bronze"]       # cols

# Per-agency productivity q_a (used by q^T X 1 objective)
q = {"A1": 10, "A2": 7, "A3": 5, "A4": 8}

# Per-agency capacities U_a (slots available now)
U = {"A1": 4, "A2": 3, "A3": 1, "A4": 2}      # sum = 10
Utot = sum(U.values())

# Floors L_a for live mode (will be overridden with baseline totals to prevent negative deltas)
L_seed = {"A1": 3, "A2": 2, "A3": 1, "A4": 1}

# Number of online arrivals to simulate after the batch plan
U_MONTH = 2

# Per-agency headroom to allow U_MONTH arrivals to remain feasible in the toy example
ONLINE_HEADROOM = U_MONTH

# If True, online re-optimization cannot reshuffle bucket totals:
# each step keeps previous column totals and only increments the incoming bucket by +1.
FIX_BUCKET_TOTALS_ONLINE = True
# If True, ZIP-inadmissible agencies are frozen by cell (not only by row total) in online mode.
STRICT_ZIP_CELL_LOCK_ONLINE = True
# Incoming policy bucket by online step (length should be >= U_MONTH).
ONLINE_BUCKET_SEQUENCE = ["Gold", "Gold"]

# ZIP admissibility for incoming policies
ziplist_batch = {"A1": None, "A2": None, "A3": None, "A4": None}
ziplist_online_by_step = {
    1: {"A1": {"28001", "28002"}, "A2": None, "A3": {"28003"}, "A4": {"28004"}},
    # Step 2 uses a ZIP inadmissible for A1 to match the toy figure:
    # the incremental Gold policy is routed to A2.
    2: {"A1": {"28002"}, "A2": None, "A3": {"28003"}, "A4": {"28004"}},
}
res_zip_by_step = {1: "28001", 2: "28001"}

def zip_admissible(a: str, online: bool, online_step: int | None = None) -> bool:
    if not online:
        z = ziplist_batch[a]
        return True if z in (None, set()) else (res_zip_by_step[1] in z)

    step = 1 if online_step is None else online_step
    z_map = ziplist_online_by_step.get(step, ziplist_online_by_step[1])
    res_zip = res_zip_by_step.get(step, res_zip_by_step[1])
    z = z_map[a]
    return True if z in (None, set()) else (res_zip in z)

# Bucket-specific weights W_{a,c} to create variety (used by <W,X> objective)
random.seed(42)
bucket_factor = {"Bronze": 0.85, "Silver": 1.00, "Gold": 1.20}

def Wac(a, c):
    return q[a] * bucket_factor[c] + 0.05 * random.random()  # small noise to avoid ties

W = {(a, c): Wac(a, c) for a in AGENCIES for c in BUCKETS}

# Optional: global bucket mix constraints (to force spread across columns)
L_bucket = {"Bronze": 2, "Silver": 3, "Gold": 2}   # mins (sum <= Utot)
U_bucket = {"Bronze": 6, "Silver": 7, "Gold": 7}   # maxes (>= Utot+1 collectively)

# ---------- Vectorization helpers for matrices ----------
VAR_COLS = [(a, c) for a in AGENCIES for c in BUCKETS]
n = len(VAR_COLS)


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

def build_constraint_matrices(total_inventory: int, online: bool, L_use: dict,
                              use_bucket_bounds: bool = True, online_step: int | None = None):
    """
    Returns (A_ineq, b_vec, Aeq_mat, beq_vec)
    total_inventory: total policies to assign in this run
    online: True for live mode (incremental policy), False for batch
    L_use: floors per agency used in live mode (ignored in baseline)
    """
    A_rows, b_vals = [], []
    Aeq_rows, beq_vals = [], []

    # Per-agency capacity with headroom in live (only impacts feasibility)
    for a in AGENCIES:
        headroom = ONLINE_HEADROOM if (online and U[a] > 0) else 0
        A_rows.append(row_for_agency(a, +1))
        b_vals.append(U[a] + headroom)

    # Global conservation
    Aeq_rows.append(ones_row())
    beq_vals.append(total_inventory)

    # Bucket mix (optional)
    if use_bucket_bounds:
        for c in BUCKETS:
            # min per bucket: sum_a x_{a,c} >= L_bucket[c]  ->  -sum_a x_{a,c} <= -L_bucket[c]
            A_rows.append(row_for_bucket(c, -1))
            b_vals.append(-L_bucket[c])
            # max per bucket: sum_a x_{a,c} <= U_bucket[c]
            A_rows.append(row_for_bucket(c, +1))
            b_vals.append(U_bucket[c])

    # Floors and ZIP locking
    if online:
        for a in AGENCIES:
            # floors: sum_c x_{a,c} >= L_use[a]  ->  -sum_c x_{a,c} <= -L_use[a]
            A_rows.append(row_for_agency(a, -1))
            b_vals.append(-L_use[a])
    for a in AGENCIES:
        if not zip_admissible(a, online, online_step):
            Aeq_rows.append(row_for_agency(a, +1))
            beq_vals.append(L_use[a] if online else 0)

    A_ineq = np.vstack(A_rows) if A_rows else np.zeros((0, n), dtype=int)
    b_vec = np.array(b_vals, dtype=int)
    Aeq_mat = np.vstack(Aeq_rows) if Aeq_rows else np.zeros((0, n), dtype=int)
    beq_vec = np.array(beq_vals, dtype=int)
    return A_ineq, b_vec, Aeq_mat, beq_vec

# ---------- Solve with PuLP ----------

def solve(total_inventory: int, online: bool, L_override: dict | None = None,
          use_W: bool = True, use_bucket_bounds: bool = True,
          online_step: int | None = None,
          prev_bucket_totals: dict[str, int] | None = None,
          incoming_bucket: str | None = None,
          prev_X: pd.DataFrame | None = None):
    """
    total_inventory: total policies to assign in this run
    online: True for live mode (incremental policy), False for batch
    L_override: if provided in live mode, floors are taken from this dict (enforces no negative deltas)
    use_W: objective uses W_{a,c} (variation). If False, uses q_a only.
    """
    L_use = L_override if (online and L_override is not None) else L_seed

    prob = pl.LpProblem("Assignment", pl.LpMaximize)
    x = {(a, c): pl.LpVariable(f"x_{a}_{c}", lowBound=0, cat=pl.LpInteger)
         for a in AGENCIES for c in BUCKETS}

    # Objective
    if use_W:
        prob += pl.lpSum(W[(a, c)] * x[(a, c)] for a in AGENCIES for c in BUCKETS), "Obj_WdotX"
    else:
        prob += pl.lpSum(q[a] * x[(a, c)] for a in AGENCIES for c in BUCKETS), "Obj_qTX1"

    # Per-agency capacity (headroom in live if U[a]>0)
    for a in AGENCIES:
        headroom = ONLINE_HEADROOM if (online and U[a] > 0) else 0
        prob += pl.lpSum(x[(a, c)] for c in BUCKETS) <= U[a] + headroom, f"Cap_{a}"

    # Global conservation
    prob += pl.lpSum(x[(a, c)] for a in AGENCIES for c in BUCKETS) == total_inventory, "Global"

    # Bucket mix (optional)
    if use_bucket_bounds:
        for c in BUCKETS:
            prob += pl.lpSum(x[(a, c)] for a in AGENCIES) >= L_bucket[c], f"BucketMin_{c}"
            prob += pl.lpSum(x[(a, c)] for a in AGENCIES) <= U_bucket[c], f"BucketMax_{c}"

    # Optional strict online bucket conservation:
    # keep previous bucket totals fixed except +1 in the incoming bucket.
    if online and FIX_BUCKET_TOTALS_ONLINE:
        if prev_bucket_totals is None or incoming_bucket is None:
            raise ValueError("prev_bucket_totals and incoming_bucket are required in online mode "
                             "when FIX_BUCKET_TOTALS_ONLINE=True")
        for c in BUCKETS:
            target_c = int(prev_bucket_totals[c]) + (1 if c == incoming_bucket else 0)
            prob += pl.lpSum(x[(a, c)] for a in AGENCIES) == target_c, f"BucketFixed_{c}"

    # Optional strict ZIP lock by cell:
    # if incoming ZIP is inadmissible for agency a, freeze all its cells at previous values.
    if online and STRICT_ZIP_CELL_LOCK_ONLINE:
        if prev_X is None:
            raise ValueError("prev_X is required in online mode when STRICT_ZIP_CELL_LOCK_ONLINE=True")
        for a in AGENCIES:
            if not zip_admissible(a, online, online_step):
                for c in BUCKETS:
                    prob += x[(a, c)] == int(prev_X.loc[a, c]), f"ZIP_{a}_{c}_cell"

    # Floors and ZIP lock
    if online:
        for a in AGENCIES:
            prob += pl.lpSum(x[(a, c)] for c in BUCKETS) >= L_use[a], f"Floor_{a}"
    for a in AGENCIES:
        if not zip_admissible(a, online, online_step):
            lock_value = L_use[a] if online else 0
            prob += pl.lpSum(x[(a, c)] for c in BUCKETS) == lock_value, f"ZIP_{a}_eq"

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

# ---------- Explainability ----------

def explain_solution(label: str, online: bool, total_inventory: int, X: pd.DataFrame,
                     totals: pd.Series, L_use: dict, use_bucket_bounds: bool = True,
                     online_step: int | None = None):
    print(f"\n--- Explainability: {label} ---")

    # Objective contributions
    contrib = []
    for a in AGENCIES:
        for c in BUCKETS:
            if X.loc[a, c] > 0:
                contrib.append(((a, c), W[(a, c)] * X.loc[a, c]))
    contrib.sort(key=lambda t: t[1], reverse=True)
    top = contrib[:5]
    if top:
        print("Top objective contributions:")
        for (a, c), val in top:
            print(f"  {a}-{c}: x={int(X.loc[a, c])}  W={W[(a, c)]:.2f}  contrib={val:.2f}")

    # Per-agency capacity slack
    print("\nCapacity usage (row sums):")
    for a in AGENCIES:
        headroom = ONLINE_HEADROOM if (online and U[a] > 0) else 0
        cap = U[a] + headroom
        used = int(totals[a])
        slack = cap - used
        tag = "binding" if slack == 0 else "slack"
        print(f"  {a}: used {used} / cap {cap} -> {tag} (slack={slack})")

    # Global conservation
    total_assigned = int(X.values.sum())
    target = total_inventory
    g_slack = target - total_assigned
    g_tag = "binding" if g_slack == 0 else "violation"
    print(f"\nGlobal conservation: total {total_assigned} vs target {target} -> {g_tag}")

    # Bucket mix
    if use_bucket_bounds:
        print("\nBucket mix totals:")
        for c in BUCKETS:
            tot_c = int(X[c].sum())
            slack_low = tot_c - L_bucket[c]
            slack_high = U_bucket[c] - tot_c
            low_tag = "binding" if slack_low == 0 else "slack"
            high_tag = "binding" if slack_high == 0 else "slack"
            print(
                f"  {c}: total {tot_c} in [{L_bucket[c]}, {U_bucket[c]}]"
                f" -> min {low_tag} (slack={slack_low}), max {high_tag} (slack={slack_high})"
            )

    # Live-only rules
    if online:
        print("\nOnline-only rules:")
        for a in AGENCIES:
            floor = L_use[a]
            used = int(totals[a])
            slack = used - floor
            tag = "binding" if slack == 0 else "slack"
            print(f"  Floor {a}: used {used} >= {floor} -> {tag} (slack={slack})")
        locked = [a for a in AGENCIES if not zip_admissible(a, online, online_step)]
        if locked:
            print(f"  ZIP locks active: {', '.join(locked)}")
        else:
            print("  ZIP locks active: none")

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

    # ----- BASELINE (BATCH) -----
    st0, X0, tot0, obj0_q, obj0_W = solve(
        Utot, online=False, use_W=True, use_bucket_bounds=True
    )
    print("\n=== BASELINE (batch) ===")
    print("Status:", st0)
    print("Assignment X0:\n", X0)
    print("Totals:", tot0.to_dict())
    print("Objective (q):", round(obj0_q, 3), " Objective (W):", round(obj0_W, 3))
    heatmap_matrix(X0, "Baseline X (varied)", "baseline_X_heatmap.png")
    stacked_by_bucket(X0, "Baseline: per-agency bucket mix", "baseline_bucket_mix.png")
    explain_solution("Baseline", online=False, total_inventory=Utot,
                     X=X0, totals=tot0, L_use=L_seed, use_bucket_bounds=True)

    online_arrivals = U_MONTH
    if FIX_BUCKET_TOTALS_ONLINE and len(ONLINE_BUCKET_SEQUENCE) < online_arrivals:
        raise ValueError("ONLINE_BUCKET_SEQUENCE must have at least U_MONTH elements")

    # ----- ONLINE ARRIVALS -----
    X_prev, tot_prev, obj_prev_q, obj_prev_W = X0, tot0, obj0_q, obj0_W
    for step in range(1, online_arrivals + 1):
        total_inventory = Utot + step
        L_from_prev = {a: int(tot_prev[a]) for a in AGENCIES}
        prev_bucket_totals = {c: int(X_prev[c].sum()) for c in BUCKETS}
        incoming_bucket = ONLINE_BUCKET_SEQUENCE[step - 1]
        st1, X1, tot1, obj1_q, obj1_W = solve(total_inventory, online=True,
                                              L_override=L_from_prev,
                                              use_W=True, use_bucket_bounds=True,
                                              online_step=step,
                                              prev_bucket_totals=prev_bucket_totals,
                                              incoming_bucket=incoming_bucket,
                                              prev_X=X_prev)
        print(f"\n=== ONLINE (+{step}) ===")
        print("Status:", st1)
        print("Assignment X:\n", X1)
        print("Totals:", tot1.to_dict())
        print("Objective (q):", round(obj1_q, 3), " Objective (W):", round(obj1_W, 3))

        delta = (tot1 - tot_prev).astype(int)
        print("\nDelta (Online - Previous):", delta.to_dict(),
              "  Obj Δ (q):", round(obj1_q - obj_prev_q, 3),
              "  Obj Δ (W):", round(obj1_W - obj_prev_W, 3))
        assert all(d >= 0 for d in delta.values), f"Negative deltas found: {delta.to_dict()}"
        assert int(delta.sum()) == 1, f"Delta sum is not 1: {int(delta.sum())}"

        heatmap_matrix(X1, f"Online X (varied, +{step})", f"live_X_heatmap_{step}.png")
        stacked_by_bucket(X1, f"Online: per-agency bucket mix (+{step})",
                          f"live_bucket_mix_{step}.png")
        explain_solution(f"Online (+{step})", online=True, total_inventory=total_inventory,
                         X=X1, totals=tot1, L_use=L_from_prev, use_bucket_bounds=True,
                         online_step=step)

        X_prev, tot_prev, obj_prev_q, obj_prev_W = X1, tot1, obj1_q, obj1_W

    print("\nDone.")

if __name__ == "__main__":
    main()
