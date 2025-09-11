# leakage_test_runner.py
# End-to-end leakage checks with printed diagnostics.

import os
import sys
import json
import argparse
import pickle
import random
import numpy as np
import pandas as pd
import networkx as nx

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import add_self_loops

# ------------------------------ Determinism ----------------------------------

def seed_all(s=42):
    os.environ["PYTHONHASHSEED"] = str(s)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------------------ Utilities ------------------------------------

ENTITY_TYPES = [
    "claim",
    "insurer_license_plate",
    "insurer_phone_number",
    "insurer_email",
    "insurer_address",
    "repair_shop",
    "bank_account",
    "claim_location",
    "third_party_license_plate",
]
type_to_idx = {t: i for i, t in enumerate(ENTITY_TYPES)}

def confusion_and_metrics(y_true, y_pred):
    tp = ((y_true == 1) & (y_pred == 1)).sum().item()
    tn = ((y_true == 0) & (y_pred == 0)).sum().item()
    fp = ((y_true == 0) & (y_pred == 1)).sum().item()
    fn = ((y_true == 1) & (y_pred == 0)).sum().item()
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-12, (prec + rec))
    return {"acc": acc, "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}

def best_threshold_by_f1(y_true, probs, grid=None):
    if grid is None:
        grid = torch.linspace(0.05, 0.95, steps=19)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        pred = (probs >= t).long()
        m = confusion_and_metrics(y_true, pred)
        if m["f1"] > best_f1:
            best_f1, best_t = m["f1"], float(t)
    return best_t, best_f1

# For temporal direction checks, we need a notion of "node time".
def compute_node_time(G: nx.DiGraph):
    node_time = {}
    for n, d in G.nodes(data=True):
        if d.get("node_type") == "claim":
            node_time[n] = d.get("claim_date", None)
        else:
            ts = []
            for _, v, ed in G.out_edges(n, data=True):
                if ed.get("timestamp") is not None:
                    ts.append(ed["timestamp"])
            for u, _, ed in G.in_edges(n, data=True):
                if ed.get("timestamp") is not None:
                    ts.append(ed["timestamp"])
            node_time[n] = min(ts) if ts else None
    return node_time

# ------------------------------ Data builders --------------------------------

def build_temporal_data_variant(G, labels_dict, cutoff, edge_mode='undirected', use_prior=True, prior_alpha=1.0, prior_beta=3.0):
    """
    edge_mode: 'undirected' | 'forward' | 'backward'
      - forward: past -> future only (safe)
      - backward: future -> past only (should collapse ~random if no leakage)
      - undirected: both directions (risky if times ignored)
    use_prior: include neighbor-fraud prior feature
    """
    node_time = compute_node_time(G)

    # claims up to cutoff
    claim_nodes = [
        n for n, d in G.nodes(data=True)
        if d.get("node_type") == "claim" and d.get("claim_date") is not None and d["claim_date"] <= cutoff
    ]
    # edges up to cutoff
    eligible_edges = [
        (u, v, d) for u, v, d in G.edges(data=True)
        if d.get("timestamp") is not None and d["timestamp"] <= cutoff
    ]
    # nodes appearing on eligible edges (entities)
    nodes_in_edges = set()
    for u, v, _ in eligible_edges:
        nodes_in_edges.add(u); nodes_in_edges.add(v)
    keep_nodes = set(claim_nodes) | {n for n in nodes_in_edges if G.nodes[n].get("node_type") != "claim"}

    # deterministic subgraph
    H = nx.DiGraph()
    for n in sorted(keep_nodes, key=str):
        H.add_node(n, **G.nodes[n])

    def allow_edge(u, v):
        tu, tv = node_time.get(u, None), node_time.get(v, None)
        if edge_mode == 'undirected':
            return True
        if tu is None or tv is None:
            return False
        if edge_mode == 'forward':   # past -> future
            return tu <= tv
        if edge_mode == 'backward':  # future -> past
            return tu >= tv
        return True

    for u, v, d in sorted(eligible_edges, key=lambda e: (str(e[0]), str(e[1]), e[2].get("timestamp"))):
        if u in keep_nodes and v in keep_nodes and allow_edge(u, v):
            H.add_edge(u, v, **d)

    nodes = sorted(H.nodes(), key=str)
    nid = {n: i for i, n in enumerate(nodes)}

    # features
    T = len(ENTITY_TYPES)
    und = H.to_undirected()
    max_in  = max((H.in_degree(n)  for n in nodes), default=1)
    max_out = max((H.out_degree(n) for n in nodes), default=1)
    max_deg = max((und.degree(n)   for n in nodes), default=1)

    x = torch.zeros((len(nodes), T + 3 + 1), dtype=torch.float)  # +1 for prior
    y = torch.zeros((len(nodes),), dtype=torch.long)
    claim_mask = torch.zeros((len(nodes),), dtype=torch.bool)

    def node_label(n): return int(labels_dict.get(str(n), 0))
    def claim_time(n): return H.nodes[n].get("claim_date", None)
    def decision_time(n):
        nd = H.nodes[n]
        return nd.get("decision_date", nd.get("claim_date", None))

    # base features + labels
    for n in nodes:
        i = nid[n]; attrs = H.nodes[n]; tname = attrs.get("node_type", "claim")
        x[i, type_to_idx.get(tname, 0)] = 1.0
        x[i, T + 0] = (H.in_degree(n)  / max_in)  if max_in  > 0 else 0.0
        x[i, T + 1] = (H.out_degree(n) / max_out) if max_out > 0 else 0.0
        x[i, T + 2] = (und.degree(n)   / max_deg) if max_deg > 0 else 0.0
        if tname == "claim":
            claim_mask[i] = True
            y[i] = node_label(n)
        else:
            y[i] = 0

    # neighbor-fraud prior (optional, Laplace/Beta smoothing)
    if use_prior:
        alpha, beta = prior_alpha, prior_beta
        for n in nodes:
            if H.nodes[n].get("node_type") != "claim": continue
            i = nid[n]
            t_cur = claim_time(n)
            if t_cur is None:
                x[i, T + 3] = alpha / (alpha + beta)
                continue
            older = set()
            for p in H.predecessors(n):
                if H.nodes[p].get("node_type") == "claim":
                    t_p = claim_time(p)
                    if t_p is not None and t_p < t_cur:
                        older.add(p)
            for e in H.predecessors(n):
                if H.nodes[e].get("node_type") != "claim":
                    for p in H.predecessors(e):
                        if H.nodes[p].get("node_type") == "claim":
                            t_p = claim_time(p)
                            if t_p is not None and t_p < t_cur:
                                older.add(p)
            decided = []
            for p in older:
                d_p = decision_time(p)
                if d_p is None or d_p <= t_cur:
                    decided.append(p)
            pos = sum(node_label(p) for p in decided)
            n_dec = float(len(decided))
            frac = (pos + alpha) / (n_dec + alpha + beta) if n_dec > 0 else alpha / (alpha + beta)
            x[i, T + 3] = float(frac)

    # edge_index
    pairs = set()
    for u, v in H.edges():
        a, b = nid[u], nid[v]
        pairs.add((a, b))
        if edge_mode == 'undirected':
            pairs.add((b, a))
    if pairs:
        src, dst = zip(*sorted(pairs))
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_index, _ = add_self_loops(edge_index, num_nodes=len(nodes))

    return Data(x=x, edge_index=edge_index, y=y, claim_mask=claim_mask)

# ------------------------------ Model & Train --------------------------------

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden=128, out_channels=2, dropout=0.35):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.conv3 = SAGEConv(hidden, out_channels)
        self.dropout = dropout

    def forward(self, data):
        x, ei = data.x, data.edge_index
        x = self.conv1(x, ei); x = F.relu(x); x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, ei); x = F.relu(x); x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.conv3(x, ei)
        return out

def train_eval_temporal(train_data, val_data, test_data, epochs=220, lr=0.01, seed=42, outdir="outputs"):
    seed_all(seed)
    os.makedirs(outdir, exist_ok=True)

    train_ids = torch.where(train_data.claim_mask)[0]
    counts = torch.bincount(train_data.y[train_ids], minlength=2)
    tot = counts.sum().item()
    weights = torch.tensor([tot / max(1, counts[0].item()),
                            tot / max(1, counts[1].item())], dtype=torch.float)

    model = GraphSAGE(in_channels=train_data.num_node_features, hidden=128, out_channels=2, dropout=0.30)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = torch.nn.CrossEntropyLoss(weight=weights)

    best_val_f1, best_state, best_t = -1.0, None, 0.5
    patience, bad = 25, 0

    for ep in range(1, epochs + 1):
        model.train()
        logits_tr = model(train_data)
        loss = crit(logits_tr[train_ids], train_data.y[train_ids])
        opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            val_ids = torch.where(val_data.claim_mask)[0]
            logits_va = model(val_data)
            probs_va = F.softmax(logits_va[val_ids], dim=1)[:, 1]
            t_star, _ = best_threshold_by_f1(val_data.y[val_ids], probs_va)
            pred_va = (probs_va >= t_star).long()
            mva = confusion_and_metrics(val_data.y[val_ids], pred_va)

        if mva["f1"] > best_val_f1:
            best_val_f1 = mva["f1"]; best_t = t_star
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if ep % 25 == 0 or ep == epochs:
            print(f"Ep {ep:03d} | train loss {loss:.4f} | val F1 {mva['f1']:.3f} "
                  f"(t*={t_star:.2f}) | val acc {mva['acc']:.3f} "
                  f"prec {mva['precision']:.3f} rec {mva['recall']:.3f}")

        if bad >= patience:
            print(f"[early stop] patience reached at epoch {ep}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final on TEST with val-optimized threshold
    model.eval()
    with torch.no_grad():
        test_ids = torch.where(test_data.claim_mask)[0]
        logits_te = model(test_data)
        probs_te = F.softmax(logits_te[test_ids], dim=1)[:, 1].cpu()
        pred_te = (probs_te >= best_t).long()
        y_te = test_data.y[test_ids].cpu()

    m = confusion_and_metrics(y_te, pred_te)
    print(f"\n[TEST] @t*={best_t:.2f}: acc={m['acc']:.3f} prec={m['precision']:.3f} "
          f"rec={m['recall']:.3f} f1={m['f1']:.3f}  "
          f"cm=[tp:{m['tp']} fp:{m['fp']} tn:{m['tn']} fn:{m['fn']}]")
    return model, None, m

# ------------------------------ Leakage Suite --------------------------------

def run_leakage_tests(G, df, epochs=180, lr=0.01, outdir="outputs"):
    seed_all(42)
    labels = dict(zip(df["claim_id"].astype(str), df["is_fraud"].astype(int)))
    q_train = df["claim_date"].quantile(0.70)
    q_val   = df["claim_date"].quantile(0.85)
    t_train_end = pd.Timestamp(q_train).to_pydatetime()
    t_val_end   = pd.Timestamp(q_val).to_pydatetime()
    t_test_end  = df["claim_date"].max().to_pydatetime()

    print("Temporal cutoffs:")
    print("  Train ≤", t_train_end)
    print("  Val   ≤", t_val_end)
    print("  Test  ≤", t_test_end)

    def build_all(edge_mode='undirected', use_prior=True, lbls=labels):
        dtr = build_temporal_data_variant(G, lbls, cutoff=t_train_end, edge_mode=edge_mode, use_prior=use_prior)
        dva = build_temporal_data_variant(G, lbls, cutoff=t_val_end,   edge_mode=edge_mode, use_prior=use_prior)
        dte = build_temporal_data_variant(G, lbls, cutoff=t_test_end,  edge_mode=edge_mode, use_prior=use_prior)
        return dtr, dva, dte

    results = {}

    print("\n[A] Baseline: UNDIRECTED + PRIOR")
    dtr, dva, dte = build_all(edge_mode='undirected', use_prior=True)
    _, _, m = train_eval_temporal(dtr, dva, dte, epochs=epochs, lr=lr, seed=42, outdir=outdir)
    results["A_baseline_undirected+prior"] = m

    print("\n[B] FORWARD-ONLY (past→future) + PRIOR")
    dtr, dva, dte = build_all(edge_mode='forward', use_prior=True)
    _, _, m = train_eval_temporal(dtr, dva, dte, epochs=epochs, lr=lr, seed=42, outdir=outdir)
    results["B_forward+prior"] = m

    print("\n[C] FORWARD-ONLY + NO PRIOR (ablation)")
    dtr, dva, dte = build_all(edge_mode='forward', use_prior=False)
    _, _, m = train_eval_temporal(dtr, dva, dte, epochs=epochs, lr=lr, seed=42, outdir=outdir)
    results["C_forward_no_prior"] = m

    print("\n[D] BACKWARD-ONLY (future→past) + PRIOR  [should ~random if no leakage]")
    dtr, dva, dte = build_all(edge_mode='backward', use_prior=True)
    _, _, m = train_eval_temporal(dtr, dva, dte, epochs=epochs, lr=lr, seed=42, outdir=outdir)
    results["D_backward+prior"] = m

    print("\n[E] FORWARD-ONLY + PRIOR with TRAIN LABELS SHUFFLED  [should ~random]")
    shuffled = dict(labels)
    train_claims = df.loc[df["claim_date"] <= t_train_end, "claim_id"].astype(str).tolist()
    y_train_vals = [labels[c] for c in train_claims]
    rng = np.random.default_rng(999)
    rng.shuffle(y_train_vals)
    for c, yv in zip(train_claims, y_train_vals):
        shuffled[c] = int(yv)
    dtr, dva, dte = build_all(edge_mode='forward', use_prior=True, lbls=shuffled)
    _, _, m = train_eval_temporal(dtr, dva, dte, epochs=epochs, lr=lr, seed=42, outdir=outdir)
    results["E_forward+prior_train_labels_shuffled"] = m

    print("\n=== Leakage Test Summary (higher F1 is better, but D/E should be ~random) ===")
    for k, v in results.items():
        print(f"{k:42s} -> F1={v['f1']:.3f} | acc={v['acc']:.3f} | prec={v['precision']:.3f} | rec={v['recall']:.3f} "
              f"| cm=[tp:{v['tp']} fp:{v['fp']} tn:{v['tn']} fn:{v['fn']}]")

    # Simple verdicts
    def verdict(a, b, c, d, e):
        msgs = []
        if abs(a["f1"] - b["f1"]) <= 0.02:
            msgs.append("✅ Forward-only ≈ Baseline → low temporal bleed.")
        else:
            msgs.append("⚠️  Forward-only ≠ Baseline → directionality matters; check undirected edges.")
        if b["f1"] > c["f1"] + 0.05:
            msgs.append("✅ Prior helps but is not sole signal (forward no-prior drops).")
        else:
            msgs.append("⚠️  Prior ablation didn't drop much → structure may dominate; re-check features.")
        if d["f1"] < 0.15:
            msgs.append("✅ Backward-only ≈ random → good (no future→past info).")
        else:
            msgs.append("❌ Backward-only strong → leakage very likely.")
        if e["f1"] < 0.15:
            msgs.append("✅ Train label shuffle killed performance on val/test → training behaves properly.")
        else:
            msgs.append("❌ Label shuffle still high → leakage/miscalibration likely.")
        return msgs

    msgs = verdict(results["A_baseline_undirected+prior"],
                   results["B_forward+prior"],
                   results["C_forward_no_prior"],
                   results["D_backward+prior"],
                   results["E_forward+prior_train_labels_shuffled"])
    print("\n--- Quick Verdict ---")
    for mline in msgs:
        print(mline)

    return results

# ---------------------------------- Main -------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Leakage diagnostics for temporal GraphSAGE.")
    parser.add_argument("--csv",   default="data/sy_dataset_1.csv",
                        help="Path to claims CSV (must include claim_id, is_fraud, claim_date).")
    parser.add_argument("--graph", default="data/temporal_graph_with_edge_attrs.gpickle",
                        help="Path to NetworkX gpickle with edge timestamps.")
    parser.add_argument("--out",   default="outputs", help="Output directory.")
    parser.add_argument("--epochs", type=int, default=220, help="Training epochs per scenario.")
    parser.add_argument("--lr",     type=float, default=0.01, help="Learning rate.")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    seed_all(42)

    # Load data
    df = pd.read_csv(args.csv, parse_dates=["claim_date"])
    with open(args.graph, "rb") as f:
        G = pickle.load(f)

    # Sanity prints
    print(f"Loaded CSV: {args.csv}  rows={len(df):,}")
    print(f"Loaded graph: {args.graph}  | nodes={G.number_of_nodes():,} edges={G.number_of_edges():,}")

    results = run_leakage_tests(G, df, epochs=args.epochs, lr=args.lr, outdir=args.out)

    # Save JSON summary
    summary_path = os.path.join(args.out, "leakage_tests_summary.json")
    with open(summary_path, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\nSaved leakage test summary to: {summary_path}")

if __name__ == "__main__":
    main()


