# optimization_maxent.py
# ------------------------------------------------------------
# Maximum-entropy assignment (continuous) with the same constraints.
# Uses CVXPY to maximize entropy subject to linear constraints.
#
# Run:
#   python optimization_maxent.py
#
# Requirements (example):
#   pip install cvxpy numpy pandas
# ------------------------------------------------------------

import os
import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Output folder ----------
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
RUN_ID = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "figs", SCRIPT_NAME, RUN_ID)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_fig(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")

# ---------- Synthetic inputs ----------
AGENCIES = ["A1", "A2", "A3", "A4"]
BUCKETS = ["Gold", "Silver", "Bronze"]

U = {"A1": 4, "A2": 3, "A3": 1, "A4": 2}
Utot = sum(U.values())

L_bucket = {"Bronze": 2, "Silver": 3, "Gold": 2}
U_bucket = {"Bronze": 6, "Silver": 7, "Gold": 7}

# Live-mode info (optional)
L_seed = {"A1": 3, "A2": 2, "A3": 1, "A4": 1}
res_zip = "28001"
ziplist = {"A1": {"28001", "28002"}, "A2": None, "A3": {"28003"}, "A4": {"28004"}}


def zip_admissible(a: str) -> bool:
    z = ziplist[a]
    return True if z in (None, set()) else (res_zip in z)


# ---------- Solve maximum-entropy problem ----------
use_online_constraints = False
# Continuous assignment matrix (relaxes integrality)
X = cp.Variable((len(AGENCIES), len(BUCKETS)), nonneg=True)

constraints = []

# Per-agency capacity (row sums)
row_sums = cp.sum(X, axis=1)
constraints += [row_sums <= np.array([U[a] for a in AGENCIES])]

# Global conservation
constraints += [cp.sum(X) == Utot]

# Bucket mix (column sums)
col_sums = cp.sum(X, axis=0)
constraints += [col_sums >= np.array([L_bucket[c] for c in BUCKETS])]
constraints += [col_sums <= np.array([U_bucket[c] for c in BUCKETS])]

# Live-only floors and ZIP lock
if use_online_constraints:
    L_use = L_seed
    constraints += [row_sums >= np.array([L_use[a] for a in AGENCIES])]
    for i, a in enumerate(AGENCIES):
        if not zip_admissible(a):
            constraints += [row_sums[i] == L_use[a]]

# Entropy objective: maximize sum of entropies of entries
# Use elementwise entropy; CVXPY handles 0*log(0) as 0.
objective = cp.Maximize(cp.sum(cp.entr(X)))

prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.SCS, verbose=False)

print("Status:", prob.status)
print("Objective (entropy):", prob.value)

if X.value is None:
    raise RuntimeError("No feasible solution. Try disabling online constraints or relaxing bounds.")
X_val = np.maximum(X.value, 0)
X_df = pd.DataFrame(X_val, index=AGENCIES, columns=BUCKETS)
print("\nAssignment (continuous, max-entropy):\n", X_df.round(3))
print("\nRow sums:", X_df.sum(axis=1).round(3).to_dict())
print("Column sums:", X_df.sum(axis=0).round(3).to_dict())

# Plot heatmap for visibility
fig, ax = plt.subplots(figsize=(6, 4))
im = ax.imshow(X_df.values, aspect="auto")
ax.set_xticks(range(len(BUCKETS))); ax.set_xticklabels(BUCKETS)
ax.set_yticks(range(len(AGENCIES))); ax.set_yticklabels(AGENCIES)
ax.set_title("Max-entropy assignment (continuous)")
for i in range(X_df.shape[0]):
    for j in range(X_df.shape[1]):
        ax.text(j, i, f"{X_df.iat[i, j]:.2f}", ha="center", va="center")
plt.tight_layout()
save_fig(fig, "maxent_assignment_heatmap.png")
