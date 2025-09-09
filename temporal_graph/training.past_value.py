# 03_train_temporal.py — GraphSAGE with neighbor-based prior-fraud feature (no leakage)
# Non-blocking: saves all figures to outputs/figs/ (no plt.show)

import os
import pickle
import pandas as pd
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import add_self_loops

# Use non-interactive backend to avoid blocking
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# t-SNE (optional)
try:
    from sklearn.manifold import TSNE
    _HAS_TSNE = True
except Exception:
    _HAS_TSNE = False

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

# ---------- Build temporal subgraph (adds prior-fraud-from-neighbors only) ----------
def build_temporal_data(G, labels_dict, cutoff):
    """
    Features per node:
      - one-hot(node_type)                        [len(ENTITY_TYPES)]
      - normalized in/out/undeg degrees          [3]
      - prior_fraud_from_neighbors (claims only) [1]
        = fraction of connected *older* claims that are fraud
          (older = claim_date(neighbor) < claim_date(current)
           and, if available, decision_date(neighbor) ≤ claim_date(current))
    """
    # Claims up to cutoff
    claim_nodes = [n for n, d in G.nodes(data=True)
                   if d.get("node_type") == "claim" and d.get("claim_date") <= cutoff]

    # Edges up to cutoff (edges store 'timestamp' = link_date)
    eligible_edges = [(u, v, d) for u, v, d in G.edges(data=True)
                      if d.get("timestamp") is not None and d["timestamp"] <= cutoff]

    # Nodes present in those edges (incl. entities)
    nodes_in_edges = set()
    for u, v, _ in eligible_edges:
        nodes_in_edges.add(u); nodes_in_edges.add(v)

    keep_nodes = set(claim_nodes) | {n for n in nodes_in_edges if G.nodes[n].get("node_type") != "claim"}

    # Subgraph H
    H = nx.DiGraph()
    for n in keep_nodes:
        H.add_node(n, **G.nodes[n])
    for u, v, d in eligible_edges:
        if u in keep_nodes and v in keep_nodes:
            H.add_edge(u, v, **d)

    # Map nodes
    nodes = list(H.nodes())
    nid = {n: i for i, n in enumerate(nodes)}

    # Degrees for structural features
    T = len(ENTITY_TYPES)
    und = H.to_undirected()
    max_in  = max((H.in_degree(n)  for n in nodes), default=1)
    max_out = max((H.out_degree(n) for n in nodes), default=1)
    max_deg = max((und.degree(n)   for n in nodes), default=1)

    # Feature dim = types + 3 degrees + 1 neighbor-prior-fraud
    x = torch.zeros((len(nodes), T + 3 + 1), dtype=torch.float)
    y = torch.zeros((len(nodes),), dtype=torch.long)
    claim_mask = torch.zeros((len(nodes),), dtype=torch.bool)

    def node_label(n):  # ground truth label for claim nodes
        return int(labels_dict.get(str(n), 0))
    def claim_time(n):
        return H.nodes[n].get("claim_date", None)
    def decision_time(n):  # optional; falls back to claim_time
        nd = H.nodes[n]
        return nd.get("decision_date", nd.get("claim_date", None))

    # Pass 1: base features + labels/mask
    for n, attrs in H.nodes(data=True):
        i = nid[n]
        t = attrs.get("node_type", "claim")

        x[i, type_to_idx.get(t, 0)] = 1.0
        x[i, T + 0] = (H.in_degree(n)  / max_in)  if max_in  > 0 else 0.0
        x[i, T + 1] = (H.out_degree(n) / max_out) if max_out > 0 else 0.0
        x[i, T + 2] = (und.degree(n)   / max_deg) if max_deg > 0 else 0.0

        if t == "claim":
            claim_mask[i] = True
            y[i] = node_label(n)
        else:
            y[i] = 0

    # Pass 2: compute prior_fraud_from_neighbors for claims (no self leakage)
    for n, attrs in H.nodes(data=True):
        if attrs.get("node_type") != "claim":
            continue
        i = nid[n]
        t_cur = claim_time(n)
        if t_cur is None:
            x[i, T + 3] = 0.0
            continue

        older_claims = set()

        # a) direct predecessors (claim->claim are prior→current by construction)
        for p in H.predecessors(n):
            if H.nodes[p].get("node_type") == "claim":
                t_p = claim_time(p)
                if t_p is not None and t_p < t_cur:
                    older_claims.add(p)

        # b) 2-hop claim–entity–claim older than current
        for e in H.predecessors(n):
            if H.nodes[e].get("node_type") != "claim":
                for p in H.predecessors(e):
                    if H.nodes[p].get("node_type") == "claim":
                        t_p = claim_time(p)
                        if t_p is not None and t_p < t_cur:
                            older_claims.add(p)

        # keep only neighbors whose decision_date ≤ t_cur (if available)
        decided_older = []
        for p in older_claims:
            d_p = decision_time(p)
            if d_p is None or d_p <= t_cur:
                decided_older.append(p)

        if decided_older:
            frac = sum(node_label(p) for p in decided_older) / float(len(decided_older))
        else:
            frac = 0.0

        x[i, T + 3] = float(frac)

    # Build edge_index (includes claim→claim, claim↔entity)
    src, dst = [], []
    for u, v in H.edges():
        src.append(nid[u]); dst.append(nid[v])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_index, _ = add_self_loops(edge_index, num_nodes=len(nodes))

    data = Data(x=x, edge_index=edge_index, y=y, claim_mask=claim_mask)
    return data

# ---------- Model ----------
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden=128, out_channels=2, dropout=0.35):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.conv3 = SAGEConv(hidden, out_channels)
        self.dropout = dropout

    def forward(self, data, return_embeddings: bool = False):
        x, ei = data.x, data.edge_index
        x = self.conv1(x, ei); x = F.relu(x); x = F.dropout(x, p=self.dropout, training=self.training)
        h = self.conv2(x, ei); h = F.relu(h); h = F.dropout(h, p=self.dropout, training=self.training)
        if return_embeddings:
            return h  # penultimate layer embeddings
        out = self.conv3(h, ei)
        return out

# ---------- Metrics / Lift ----------
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
        rows.append({"bin": b + 1, "n": n,
                     "fraud_rate": round(rate, 4),
                     "lift": round(rate / base if base > 0 else 0, 3),
                     "cum_capture": round(capture, 3)})
    return rows

def save_lift_plot(rows, outpath, title="Lift (test)"):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    xs   = [r["bin"] for r in rows]
    lifts = [r["lift"] for r in rows]
    cum   = [r["cum_capture"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(xs, lifts, alpha=0.75)
    ax1.axhline(1.0, linestyle="--", linewidth=1)
    ax1.set_xlabel("Decile (1 = highest score)"); ax1.set_ylabel("Lift"); ax1.set_xticks(xs)
    ax2 = ax1.twinx(); ax2.plot(xs, cum, marker="o"); ax2.set_ylabel("Cumulative capture"); ax2.set_ylim(0, 1.05)
    plt.title(title); plt.tight_layout()
    plt.savefig(outpath, dpi=160); plt.close(fig)

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

# ---------- Embedding viz + neighbors ----------
def save_tsne_embeddings(embeddings: torch.Tensor, labels: torch.Tensor, outpath, title="t-SNE of claim embeddings"):
    if not _HAS_TSNE:
        print("[warn] scikit-learn not available; skipping t-SNE.")
        return
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    z = embeddings.detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    perplexity = min(30, max(5, len(z)//50)) if len(z) > 100 else min(30, max(5, len(z)//3))
    tsne = TSNE(n_components=2, perplexity=perplexity, init="random", learning_rate="auto")
    z2 = tsne.fit_transform(z)

    fig = plt.figure(figsize=(7.5, 5.5))
    nf = y == 0; fr = y == 1
    plt.scatter(z2[nf,0], z2[nf,1], s=6, alpha=0.7, label="no fraud")
    plt.scatter(z2[fr,0], z2[fr,1], s=12, alpha=0.9, label="fraud")
    plt.legend(loc="best"); plt.title(title); plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
    plt.tight_layout(); plt.savefig(outpath, dpi=160); plt.close(fig)
    print(f"[saved] {outpath}")

def top_k_similar(embeddings: torch.Tensor, labels: torch.Tensor, node_ids: list, anchor_idx: int, k=5):
    E = F.normalize(embeddings, dim=1)
    anchor = E[anchor_idx:anchor_idx+1]
    sims = (E @ anchor.T).squeeze(1)
    order = torch.argsort(sims, descending=True)
    rows = []
    for i in order[:k+1]:
        rows.append({
            "claim_node": str(node_ids[int(i)]),
            "similarity": float(sims[int(i)].cpu()),
            "fraud_label": int(labels[int(i)].cpu())
        })
    import pandas as pd
    return pd.DataFrame(rows)

# ---------- Train / Eval ----------
def train_eval_temporal(train_data, val_data, test_data, epochs=220, lr=0.01, seed=42, outdir="outputs"):
    torch.manual_seed(seed)

    # dirs for non-blocking figure output
    figs_dir = os.path.join(outdir, "figs")
    os.makedirs(figs_dir, exist_ok=True)

    train_ids = torch.where(train_data.claim_mask)[0]
    counts = torch.bincount(train_data.y[train_ids], minlength=2)
    tot = counts.sum().item()
    weights = torch.tensor([tot / max(1, counts[0].item()),
                            tot / max(1, counts[1].item())], dtype=torch.float)

    model = GraphSAGE(in_channels=train_data.num_node_features, hidden=128, out_channels=2, dropout=0.35)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
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
                  f"(t*={t_star:.2f}) | val acc {mva['acc']:.3f} prec {mva['precision']:.3f} rec {mva['recall']:.3f}")

        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final on TEST with val-optimized threshold
    model.eval()
    with torch.no_grad():
        test_ids = torch.where(test_data.claim_mask)[0]
        logits_te = model(test_data)
        probs_te = F.softmax(logits_te[test_ids], dim=1)[:, 1].cpu()
        pred_te  = (probs_te >= best_t).long()
        y_te     = test_data.y[test_ids].cpu()

    print("Score stats (pos):", probs_te[y_te==1].min().item(), probs_te[y_te==1].median().item(), probs_te[y_te==1].max().item())
    print("Score stats (neg):", probs_te[y_te==0].min().item(), probs_te[y_te==0].median().item(), probs_te[y_te==0].max().item())

    m = confusion_and_metrics(y_te, pred_te)
    print('confusion matrix')
    print(m)
    print(f"\nFINAL TEST @t*={best_t:.2f}: acc={m['acc']:.3f} prec={m['precision']:.3f} "
          f"rec={m['recall']:.3f} f1={m['f1']:.3f}  "
          f"cm=[tp:{m['tp']} fp:{m['fp']} tn:{m['tn']} fn:{m['fn']}]")

    rows = lift_table(probs_te, y_te, bins=10)
    # Save lift plot (non-blocking)
    lift_path = os.path.join(figs_dir, "lift_test_fraud_past_time.png")
    save_lift_plot(rows, lift_path, title="Temporal Graph – Lift (test)")
    print(f"[saved] {lift_path}")

    # ---- Embedding viz + neighbors (claims only) ----
    claim_ids_te = torch.where(test_data.claim_mask)[0]
    with torch.no_grad():
        h_test = model(test_data, return_embeddings=True)
    h_claim = h_test[claim_ids_te]
    y_claim = test_data.y[claim_ids_te]

    # Save t-SNE (non-blocking)
    tsne_path = os.path.join(figs_dir, "tsne_claims_fraud_past time.png")
    save_tsne_embeddings(h_claim, y_claim, tsne_path, title="Temporal Graph – t-SNE (claims)")

    # nearest-neighbor table for the top-scored fraud (or top overall)
    mask_true_fraud = (y_te == 1)
    if mask_true_fraud.any():
        order = torch.argsort(probs_te, descending=True)
        anchor = order[mask_true_fraud[order]][0]
    else:
        anchor = torch.argmax(probs_te)

    df_sim = top_k_similar(h_claim, y_te, node_ids=claim_ids_te.cpu().tolist(), anchor_idx=int(anchor), k=5)
    print("\nSimilar claims to the selected (fraudulent) claim:")
    print(df_sim.to_string(index=False))

    return model, rows, m

def main():
    # Load data & graph
    df = pd.read_csv("data/sy_dataset_1.csv", parse_dates=["claim_date"])
    with open("data/temporal_graph_with_edge_attrs.gpickle", "rb") as f:
        G = pickle.load(f)

    # labels for claim nodes
    labels = dict(zip(df["claim_id"].astype(str), df["is_fraud"].astype(int)))

    # Temporal cutoffs from claim dates
    q_train = df["claim_date"].quantile(0.70)
    q_val   = df["claim_date"].quantile(0.85)
    t_train_end = pd.Timestamp(q_train).to_pydatetime()
    t_val_end   = pd.Timestamp(q_val).to_pydatetime()
    t_test_end  = df["claim_date"].max().to_pydatetime()

    print("Temporal cutoffs:")
    print("  Train ≤", t_train_end)
    print("  Val   ≤", t_val_end)
    print("  Test  ≤", t_test_end)

    # Build temporal subgraphs (no future leakage)
    data_train = build_temporal_data(G, labels, cutoff=t_train_end)
    data_val   = build_temporal_data(G, labels, cutoff=t_val_end)
    data_test  = build_temporal_data(G, labels, cutoff=t_test_end)

    # Train/validate/test (non-blocking saves)
    _model, _lift_rows, _metrics = train_eval_temporal(
        data_train, data_val, data_test, epochs=220, lr=0.01, seed=42, outdir="data"
    )

if __name__ == "__main__":
    main()
