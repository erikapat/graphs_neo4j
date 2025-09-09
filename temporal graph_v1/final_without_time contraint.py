# ===================== Hetero GraphSAGE + Degree Features + Val-Optimized Threshold =====================
# - Strong-signal simulation (risky shops denser + more fraudulent)
# - Heterogeneous graph (claims + entities), supervise only claim nodes
# - Node features: one-hot(node_type) + in/out/undirected degree (normalized)
# - 3-layer GraphSAGE, class-weighted CE, self-loops
# - Validation-chosen threshold maximizes F1; then evaluate on test
# - Lift table + Lift chart
# ======================================================================================================

import os
import pickle
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import networkx as nx

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import add_self_loops

import matplotlib.pyplot as plt

# --------------------------- 0) Simulate claims with STRONG signal ---------------------------
def simulate_claims_strong(n=4000, seed=42):
    rng = np.random.default_rng(seed)
    now = datetime(2024, 1, 1)

    plates = [f"PLT{10000+i}" for i in range(2500)]
    phones = [f"+34{600000000+i}" for i in range(2500)]
    emails = [f"user{i}@ex.com" for i in range(2500)]
    addrs  = [f"STREET_{i}" for i in range(1200)]
    shops  = [f"SHOP_{i}" for i in range(400)]
    banks  = [f"ES76{1000000000+i}" for i in range(1500)]
    locs   = [f"LOC_{i}" for i in range(500)]
    thirdp = [f"TP_{i}" for i in range(2500)]

    risky_idx = set(rng.choice(len(shops), size=max(1, int(0.20 * len(shops))), replace=False))
    risky_shops = set(shops[i] for i in risky_idx)

    # risky shops appear ~6x more often
    w_risky, w_safe = 0.6, 0.1
    weights = np.array([w_risky if s in risky_shops else w_safe for s in shops], dtype=float)
    weights /= weights.sum()

    rows = []
    for i in range(n):
        claim_id = str(10_000_000 + i)
        step_days = int(rng.exponential(scale=2.5))
        claim_date = now + timedelta(days=step_days)
        shop = shops[rng.choice(len(shops), p=weights)]
        p = 0.85 if shop in risky_shops else 0.05
        rows.append({
            "claim_id": claim_id,
            "claim_date": claim_date,
            "insurer_license_plate": plates[rng.integers(len(plates))],
            "insurer_phone_number":  phones[rng.integers(len(phones))],
            "insurer_email":         emails[rng.integers(len(emails))],
            "insurer_address":       addrs[rng.integers(len(addrs))],
            "repair_shop":           shop,
            "bank_account":          banks[rng.integers(len(banks))],
            "claim_location":        locs[rng.integers(len(locs))],
            "third_party_license_plate": thirdp[rng.integers(len(thirdp))],
            "is_fraud": int(rng.random() < p),
        })
    df = pd.DataFrame(rows).sort_values("claim_date").reset_index(drop=True)
    return df, risky_shops

# --------------------------- 1) Build heterogeneous temporal graph ---------------------------
ENTITY_COLUMNS = [
    "insurer_license_plate", "insurer_phone_number", "insurer_email",
    "insurer_address", "repair_shop", "bank_account",
    "claim_location", "third_party_license_plate"
]

def build_temporal_graph_with_edge_attrs(df, entity_cols):
    df = df.sort_values("claim_date")
    G = nx.DiGraph()
    seen_pairs = set()
    entity_index = {}

    for _, row in df.iterrows():
        cid = row["claim_id"]
        cdate = row["claim_date"]

        G.add_node(cid, node_type="claim", claim_date=cdate)

        for col in entity_cols:
            val = row[col]
            if pd.notna(val):
                ent = f"{col}:{val}"
                G.add_node(ent, node_type=col, label=val)

                # bidirectional edges to simulate undirected
                G.add_edge(cid, ent, edge_type="claim-entity", entity_type=col, timestamp=cdate)
                G.add_edge(ent, cid, edge_type="entity-claim", entity_type=col, timestamp=cdate)

                # connect prior claims sharing this entity
                prev = entity_index.get((col, val), [])
                for other_cid in prev:
                    if (other_cid, cid) not in seen_pairs:
                        G.add_edge(other_cid, cid, edge_type="claim-claim", via=col, shared_value=val, timestamp=cdate)
                        seen_pairs.add((other_cid, cid))
                entity_index.setdefault((col, val), []).append(cid)
    return G

# --------------------------- 2) Full graph -> PyG (one-hot type + degree features) ---------------------------
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

def full_graph_to_pyg(G, labels_dict, add_loops=True):
    nodes = list(G.nodes())
    nid = {n: i for i, n in enumerate(nodes)}
    T = len(ENTITY_TYPES)

    # degrees (normalize by max to keep scale stable)
    und = G.to_undirected()
    max_in  = max((G.in_degree(n)  for n in nodes), default=1)
    max_out = max((G.out_degree(n) for n in nodes), default=1)
    max_deg = max((und.degree(n)   for n in nodes), default=1)

    x = torch.zeros((len(nodes), T + 3), dtype=torch.float)  # one-hot type + [in,out,undeg]
    y = torch.zeros((len(nodes),), dtype=torch.long)
    claim_mask = torch.zeros((len(nodes),), dtype=torch.bool)

    for n, attrs in G.nodes(data=True):
        i = nid[n]
        t = attrs.get("node_type", "claim")
        x[i, type_to_idx.get(t, 0)] = 1.0
        x[i, T + 0] = (G.in_degree(n)  / max_in)  if max_in  > 0 else 0.0
        x[i, T + 1] = (G.out_degree(n) / max_out) if max_out > 0 else 0.0
        x[i, T + 2] = (und.degree(n)   / max_deg) if max_deg > 0 else 0.0

        if t == "claim":
            claim_mask[i] = True
            y[i] = int(labels_dict.get(str(n), 0))
        else:
            y[i] = 0  # unused for entities

    src, dst = [], []
    for u, v in G.edges():
        src.append(nid[u]); dst.append(nid[v])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    if add_loops:
        edge_index, _ = add_self_loops(edge_index, num_nodes=len(nodes))

    return Data(x=x, edge_index=edge_index, y=y, claim_mask=claim_mask)

# --------------------------- 3) GraphSAGE (3 layers + dropout) ---------------------------
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden=128, out_channels=2, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.conv3 = SAGEConv(hidden, out_channels)
        self.dropout = dropout
    def forward(self, data):
        x, ei = data.x, data.edge_index
        x = self.conv1(x, ei); x = F.relu(x); x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, ei); x = F.relu(x); x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, ei)
        return x

# --------------------------- 4) Metrics, lift, plots ---------------------------
def confusion_and_metrics(y_true, y_pred):
    tp = ((y_true == 1) & (y_pred == 1)).sum().item()
    tn = ((y_true == 0) & (y_pred == 0)).sum().item()
    fp = ((y_true == 0) & (y_pred == 1)).sum().item()
    fn = ((y_true == 1) & (y_pred == 0)).sum().item()
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp)
    rec  = tp / max(1, tp + fn)
    f1   = 2 * prec * rec / max(1e-12, (prec + rec))
    return {"acc": acc, "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}

def lift_table(probs_pos, y_true, bins=10):
    N = probs_pos.numel()
    scores, idx = torch.sort(probs_pos, descending=True)
    y_sorted = y_true[idx]
    base = (y_true == 1).float().mean().item()
    bin_size = max(1, N // bins)
    rows, cum, total_pos = [], 0, (y_true == 1).sum().item()
    for b in range(bins):
        s = b * bin_size
        e = (b + 1) * bin_size if b < bins - 1 else N
        if s >= N: break
        yb = y_sorted[s:e]
        n = yb.numel()
        pos = (yb == 1).sum().item()
        rate = pos / max(1, n)
        cum += pos
        capture = cum / max(1, total_pos)
        rows.append({
            "bin": b + 1,
            "n": n,
            "fraud_rate": round(rate, 4),
            "lift": round(rate / base if base > 0 else 0, 3),
            "cum_capture": round(capture, 3),
        })
    return rows

def plot_lift(rows, title="Lift (test)"):
    if not rows:
        print("No rows to plot."); return
    xs   = [r["bin"] for r in rows]
    lifts = [r["lift"] for r in rows]
    cum   = [r["cum_capture"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(xs, lifts, alpha=0.75)
    ax1.axhline(1.0, linestyle="--", linewidth=1)
    ax1.set_xlabel("Decile (1 = highest score)")
    ax1.set_ylabel("Lift")
    ax1.set_xticks(xs)

    ax2 = ax1.twinx()
    ax2.plot(xs, cum, marker="o")
    ax2.set_ylabel("Cumulative capture")
    ax2.set_ylim(0, 1.05)

    plt.title(title)
    plt.tight_layout()
    # To save instead of show:
    # plt.savefig("data/lift_chart.png", dpi=160, bbox_inches="tight")
    plt.show()

def best_threshold_by_f1(y_true, probs, grid=None):
    """Search threshold on validation to maximize F1."""
    if grid is None:
        grid = torch.linspace(0.05, 0.95, steps=19)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        pred = (probs >= t).long()
        m = confusion_and_metrics(y_true, pred)
        if m["f1"] > best_f1:
            best_f1, best_t = m["f1"], float(t)
    return best_t, best_f1

# --------------------------- 5) Train / Eval with val-optimized threshold ---------------------------
def train_eval_hetero(G, labels_dict, epochs=220, lr=0.01, split=(0.7, 0.1, 0.2), seed=42):
    torch.manual_seed(seed)
    data = full_graph_to_pyg(G, labels_dict, add_loops=True)

    # split only among claim nodes
    claim_idx = torch.where(data.claim_mask)[0]
    N = claim_idx.numel()
    perm = claim_idx[torch.randperm(N)]
    n_train = int(split[0] * N)
    n_val   = int(split[1] * N)
    train_ids = perm[:n_train]
    val_ids   = perm[n_train:n_train + n_val]
    test_ids  = perm[n_train + n_val:]

    # class weights from TRAIN
    counts = torch.bincount(data.y[train_ids], minlength=2)
    tot = counts.sum().item()
    w0 = tot / max(1, counts[0].item())
    w1 = tot / max(1, counts[1].item())
    weights = torch.tensor([w0, w1], dtype=torch.float)

    model = GraphSAGE(in_channels=data.num_node_features, hidden=128, out_channels=2, dropout=0.35)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = torch.nn.CrossEntropyLoss(weight=weights)

    best_val_f1 = -1.0
    best_state = None
    patience, bad = 25, 0

    for ep in range(1, epochs + 1):
        model.train()
        logits = model(data)
        loss = crit(logits[train_ids], data.y[train_ids])
        opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            probs_va = F.softmax(logits[val_ids], dim=1)[:, 1]
            t_star, f1_star = best_threshold_by_f1(data.y[val_ids], probs_va)
            pred_tr = logits[train_ids].argmax(1)
            pred_va = (probs_va >= t_star).long()
            mtr = confusion_and_metrics(data.y[train_ids], pred_tr)   # train still at argmax, ok
            mva = confusion_and_metrics(data.y[val_ids],  pred_va)

        if mva["f1"] > best_val_f1:
            best_val_f1 = mva["f1"]; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_t = t_star; bad = 0
        else:
            bad += 1

        if ep % 25 == 0 or ep == epochs:
            print(f"Ep {ep:03d} | loss {loss:.4f} | train F1 {mtr['f1']:.3f} | "
                  f"val F1 {mva['f1']:.3f} (t*={t_star:.2f}) | "
                  f"val acc {mva['acc']:.3f} prec {mva['precision']:.3f} rec {mva['recall']:.3f}")

        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_t = 0.5  # fallback

    # final test with threshold selected on validation
    model.eval()
    with torch.no_grad():
        logits = model(data)
        probs_va = F.softmax(logits[val_ids],  dim=1)[:, 1]
        # recompute t* just in case
        best_t, _ = best_threshold_by_f1(data.y[val_ids], probs_va)
        probs_te = F.softmax(logits[test_ids], dim=1)[:, 1].cpu()
        pred_te  = (probs_te >= best_t).long()
        y_te     = data.y[test_ids].cpu()

    m = confusion_and_metrics(y_te, pred_te)
    print(f"\nFINAL test@t*={best_t:.2f}: acc={m['acc']:.3f} prec={m['precision']:.3f} "
          f"rec={m['recall']:.3f} f1={m['f1']:.3f} "
          f"cm=[tp:{m['tp']} fp:{m['fp']} tn:{m['tn']} fn:{m['fn']}]")

    rows = lift_table(probs_te, y_te, bins=10)
    print("\nbin |   n   | rate   | lift | cum_capture")
    for r in rows:
        print(f"{r['bin']:>3} | {r['n']:>5} | {r['fraud_rate']:.4f} | {r['lift']:.2f} | {r['cum_capture']:.3f}")

    plot_lift(rows, title="Hetero GraphSAGE – Lift (test)")

    return model, rows, m

# --------------------------- 6) Main ---------------------------
def main():
    os.makedirs("data", exist_ok=True)

    df, risky_shops = simulate_claims_strong(n=4000, seed=42)
    df.to_csv("data/sy_dataset_1.csv", index=False)

    G = build_temporal_graph_with_edge_attrs(df, ENTITY_COLUMNS)
    with open("data/temporal_graph_with_edge_attrs.gpickle", "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

    labels = dict(zip(df["claim_id"].astype(str), df["is_fraud"].astype(int)))

    _model, _lift_rows, _metrics = train_eval_hetero(G, labels, epochs=220, lr=0.01, split=(0.7, 0.1, 0.2), seed=42)

if __name__ == "__main__":
    main()

