# 03_train_temporal.py — GraphSAGE with neighbor-based prior-fraud feature (no leakage)
# Non-blocking: saves all figures to outputs/figs/ (no plt.show)

import os
import pickle
import pandas as pd
import networkx as nx
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import add_self_loops
from sklearn.manifold import TSNE

# Use non-interactive backend to avoid blocking
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

import torch

def describe_pyg_split(name: str, data):
    N = data.num_nodes
    E = int(data.edge_index.size(1))
    D = data.num_node_features
    # claim nodes & positives (labels only meaningful on claims)
    claim_ids = torch.where(data.claim_mask)[0]
    num_claims = int(claim_ids.numel())
    num_pos = int((data.y[claim_ids] == 1).sum().item())
    # isolated nodes (no in & no out)
    deg = torch.bincount(data.edge_index.flatten(), minlength=N)
    num_isolated = int((deg == 0).sum().item())
    # self-loops
    self_loops = int((data.edge_index[0] == data.edge_index[1]).sum().item())
    # (optional) directedness heuristic: how many (u,v) also have (v,u)?
    ei = data.edge_index
    key = (ei[0] * N + ei[1]).cpu()
    rev = (ei[1] * N + ei[0]).cpu()
    undup = set(key.tolist())
    rev_match = sum((rk in undup) for rk in rev.tolist())
    approx_undirected_ratio = rev_match / max(1, E)

    print(f"[{name}] nodes={N:,} edges={E:,} feats={D} claims={num_claims:,} pos_claims={num_pos:,}")
    print(f"        isolated_nodes={num_isolated:,} self_loops={self_loops:,} approx_undirected_ratio={approx_undirected_ratio:.3f}")

def describe_temporal_edges(G, cutoff):
    """Quick edge-type counts <= cutoff, using the original NetworkX graph."""
    eligible = [(u,v,d) for u,v,d in G.edges(data=True)
                if d.get("timestamp") is not None and d["timestamp"] <= cutoff]
    by_type = {}
    for _,_,d in eligible:
        et = d.get("edge_type","<none>")
        by_type[et] = by_type.get(et, 0) + 1
    total = sum(by_type.values())
    pretty = ", ".join(f"{k}:{v:,}" for k,v in sorted(by_type.items()))
    print(f"[edges ≤ {cutoff}] total={total:,} | {pretty}")


# ---------- Build temporal subgraph (adds prior-fraud-from-neighbors only) ----------
def build_temporal_data(G, labels_dict, cutoff):
    """
    Build a temporal PyG Data object from G using edges with timestamp <= cutoff.
    Assumes G contains only:
      - entity -> claim edges (timestamp = claim_date)
      - older_claim -> newer_claim edges (timestamp = newer claim_date)

    Node features (per node):
      - one-hot(node_type)                        [len(ENTITY_TYPES)]
      - normalized in/out/undirected degrees     [3]
      - prior_fraud_from_neighbors (claims only) [1]
        = fraction of predecessor *older* claims that are fraud
          (and, if available, with decision_date <= current claim_date)

    Labels live only on claim nodes; `claim_mask` identifies them.
    """
    # ---- select nodes/edges up to cutoff (deterministic) --------------------
    eligible_edges = [
        (u, v, d) for u, v, d in G.edges(data=True)
        if d.get("timestamp") is not None and d["timestamp"] <= cutoff
    ]

    # claims present up to cutoff
    claim_nodes = [
        n for n, a in G.nodes(data=True)
        if a.get("node_type") == "claim" and a.get("claim_date") is not None and a["claim_date"] <= cutoff
    ]

    # entities that appear on eligible edges
    nodes_in_edges = set()
    for u, v, _ in eligible_edges:
        nodes_in_edges.add(u); nodes_in_edges.add(v)

    keep_nodes = set(claim_nodes) | {
        n for n in nodes_in_edges if G.nodes[n].get("node_type") != "claim"
    }

    # ---- build subgraph H with sorted insertion (stable) --------------------
    H = nx.DiGraph()
    for n in sorted(keep_nodes, key=str):
        H.add_node(n, **G.nodes[n])

    # keep eligible edges among kept nodes (already time-safe by construction)
    for u, v, d in sorted(eligible_edges, key=lambda e: (str(e[0]), str(e[1]), e[2].get("timestamp"))):
        if u in keep_nodes and v in keep_nodes:
            H.add_edge(u, v, **d)

    # deterministic node list / id map
    nodes = sorted(H.nodes(), key=str)
    nid = {n: i for i, n in enumerate(nodes)}

    # ---- features/labels ----------------------------------------------------
    T = len(ENTITY_TYPES)
    und = H.to_undirected()

    max_in  = max((H.in_degree(n)  for n in nodes), default=1)
    max_out = max((H.out_degree(n) for n in nodes), default=1)
    max_deg = max((und.degree(n)   for n in nodes), default=1)

    x = torch.zeros((len(nodes), T + 3 + 1), dtype=torch.float)  # +1 for prior
    y = torch.zeros((len(nodes),), dtype=torch.long)
    claim_mask = torch.zeros((len(nodes),), dtype=torch.bool)

    def node_label(n):  # ground truth label for claim nodes
        return int(labels_dict.get(str(n), 0))

    def claim_time(n):
        return H.nodes[n].get("claim_date", None)

    def decision_time(n):  # optional; falls back to claim_time
        nd = H.nodes[n]
        return nd.get("decision_date", nd.get("claim_date", None))

    # pass 1: base features + labels/mask
    for n in nodes:
        i = nid[n]
        attrs = H.nodes[n]
        tname = attrs.get("node_type", "claim")

        # one-hot type
        x[i, type_to_idx.get(tname, 0)] = 1.0
        # normalized degrees
        x[i, T + 0] = (H.in_degree(n)  / max_in)  if max_in  > 0 else 0.0
        x[i, T + 1] = (H.out_degree(n) / max_out) if max_out > 0 else 0.0
        x[i, T + 2] = (und.degree(n)   / max_deg) if max_deg > 0 else 0.0

        if tname == "claim":
            claim_mask[i] = True
            y[i] = node_label(n)
        else:
            y[i] = 0

    # pass 2: prior_fraud_from_neighbors for claims (predecessor claims only)
    for n in nodes:
        if H.nodes[n].get("node_type") != "claim":
            continue
        i = nid[n]
        t_cur = claim_time(n)
        if t_cur is None:
            x[i, T + 3] = 0.0
            continue

        older_claims = []
        # only direct predecessors (which, by construction, are older claims
        # that share an entity or a claim→claim edge)
        for p in H.predecessors(n):
            if H.nodes[p].get("node_type") == "claim":
                t_p = claim_time(p)
                if t_p is not None and t_p < t_cur:
                    # keep only if decided <= current time (when available)
                    d_p = decision_time(p)
                    if d_p is None or d_p <= t_cur:
                        older_claims.append(p)

        if older_claims:
            frac = sum(node_label(p) for p in older_claims) / float(len(older_claims))
        else:
            frac = 0.0

        x[i, T + 3] = float(frac)

    # ---- edge_index (deterministic order) -----------------------------------
    pairs = [(nid[u], nid[v]) for u, v in H.edges()]
    if pairs:
        src, dst = zip(*sorted(pairs))
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # self-loops are fine and do not affect temporal direction
    edge_index, _ = add_self_loops(edge_index, num_nodes=len(nodes))

    return Data(x=x, edge_index=edge_index, y=y, claim_mask=claim_mask)


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
        x = self.conv1(x, ei)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        h = self.conv2(x, ei)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
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
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-12, (prec + rec))
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
    xs = [r["bin"] for r in rows]
    lifts = [r["lift"] for r in rows]
    cum = [r["cum_capture"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(xs, lifts, alpha=0.75)
    ax1.axhline(1.0, linestyle="--", linewidth=1)
    ax1.set_xlabel("Decile (1 = highest score)");
    ax1.set_ylabel("Lift");
    ax1.set_xticks(xs)
    ax2 = ax1.twinx();
    ax2.plot(xs, cum, marker="o");
    ax2.set_ylabel("Cumulative capture");
    ax2.set_ylim(0, 1.05)
    plt.title(title);
    plt.tight_layout()
    plt.savefig(outpath, dpi=160);
    plt.close(fig)


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
def save_tsne_embeddings(h_claim: torch.Tensor, y_claim: torch.Tensor, outpath, title="GraphSAGE t-SNE (claims)"):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    z = h_claim.detach().cpu().numpy();
    y = y_claim.detach().cpu().numpy()
    perplexity = min(30, max(5, len(z) // 50)) if len(z) > 100 else min(30, max(5, len(z) // 3))
    print('Calculating T-SNE')
    tsne = TSNE(n_components=2, perplexity=perplexity,
                init="pca",
                n_iter=3000,  # More iterations for convergence
                learning_rate=1500,  # Higher learning rate for sharper clusters
                random_state=25,
                # learning_rate="auto"
                )
    z2 = tsne.fit_transform(z)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.scatter(z2[y == 0, 0], z2[y == 0, 1], s=6, alpha=0.6, label="no fraud")
    ax.scatter(z2[y == 1, 0], z2[y == 1, 1], s=12, alpha=0.85, label="fraud")
    ax.legend();
    ax.set_title(f"Temporal Graph – {title}")
    ax.set_xlabel("t-SNE 1");
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close(fig)
    print(f"[saved] {outpath}")


def top_k_similar(embeddings: torch.Tensor, labels: torch.Tensor, node_ids: list, anchor_idx: int, k=5):
    E = F.normalize(embeddings, dim=1)
    anchor = E[anchor_idx:anchor_idx + 1]
    sims = (E @ anchor.T).squeeze(1)
    order = torch.argsort(sims, descending=True)
    rows = []
    for i in order[:k + 1]:
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
    figs_dir = os.path.join(outdir, "figs_model_5")
    tables_dir = os.path.join(outdir, "tables_model_5")
    os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

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
        opt.zero_grad();
        loss.backward();
        opt.step()

        model.eval()
        with torch.no_grad():
            val_ids = torch.where(val_data.claim_mask)[0]
            logits_va = model(val_data)
            probs_va = F.softmax(logits_va[val_ids], dim=1)[:, 1]
            t_star, _ = best_threshold_by_f1(val_data.y[val_ids], probs_va)
            pred_va = (probs_va >= t_star).long()
            mva = confusion_and_metrics(val_data.y[val_ids], pred_va)

        if mva["f1"] > best_val_f1:
            best_val_f1 = mva["f1"];
            best_t = t_star
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
        pred_te = (probs_te >= best_t).long()
        y_te = test_data.y[test_ids].cpu()

    print("Score stats (pos):", probs_te[y_te == 1].min().item(), probs_te[y_te == 1].median().item(),
          probs_te[y_te == 1].max().item())
    print("Score stats (neg):", probs_te[y_te == 0].min().item(), probs_te[y_te == 0].median().item(),
          probs_te[y_te == 0].max().item())

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

    sim_csv = os.path.join(tables_dir, "similar_claims_graphsage.csv")
    df_sim.to_csv(sim_csv, index=False)
    print(f"[saved] {sim_csv}")

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
    q_val = df["claim_date"].quantile(0.85)
    t_train_end = pd.Timestamp(q_train).to_pydatetime()
    t_val_end = pd.Timestamp(q_val).to_pydatetime()
    t_test_end = df["claim_date"].max().to_pydatetime()

    print("Temporal cutoffs:")
    print("  Train ≤", t_train_end)
    print("  Val   ≤", t_val_end)
    print("  Test  ≤", t_test_end)

    # Build temporal subgraphs (no future leakage)
    data_train = build_temporal_data(G, labels, cutoff=t_train_end)
    data_val = build_temporal_data(G, labels, cutoff=t_val_end)
    data_test = build_temporal_data(G, labels, cutoff=t_test_end)

    # Summaries for graph dataset
    print('Graph statistics')
    describe_temporal_edges(G, t_train_end)
    describe_pyg_split("train", data_train)
    describe_temporal_edges(G, t_val_end)
    describe_pyg_split("val",   data_val)
    describe_temporal_edges(G, t_test_end)
    describe_pyg_split("test",  data_test)
    print('End Graph statistics')


    # Train/validate/test (non-blocking saves)
    _model, _lift_rows, _metrics = train_eval_temporal(
        data_train, data_val, data_test, epochs=220, lr=0.01, seed=42, outdir="data"
    )


if __name__ == "__main__":
    main()
