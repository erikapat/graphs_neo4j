# 04: first attempt of train a temporal graph — GraphSAGE embeddings + non-blocking plots saved to disk
import os
import pickle
import pandas as pd
import networkx as nx
import torch
import torch.nn.functional as F

# non-interactive backend to avoid blocking
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import add_self_loops
from sklearn.manifold import TSNE


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


# ---------- Build temporal subgraph (no future nodes/edges) ----------
def build_temporal_data(G, labels_dict, cutoff):
    # Claims up to cutoff
    claim_nodes = [n for n, d in G.nodes(data=True)
                   if d.get("node_type") == "claim" and d.get("claim_date") <= cutoff]

    # Edges up to cutoff
    eligible_edges = [(u, v, d) for u, v, d in G.edges(data=True)
                      if d.get("timestamp") is not None and d["timestamp"] <= cutoff]

    # Nodes present in those edges (incl. entities)
    nodes_in_edges = set()
    for u, v, _ in eligible_edges:
        nodes_in_edges.add(u);
        nodes_in_edges.add(v)

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

    # Features: one-hot(type) + [in,out,undeg] normalized within H
    T = len(ENTITY_TYPES)
    und = H.to_undirected()
    max_in = max((H.in_degree(n) for n in nodes), default=1)
    max_out = max((H.out_degree(n) for n in nodes), default=1)
    max_deg = max((und.degree(n) for n in nodes), default=1)

    x = torch.zeros((len(nodes), T + 3), dtype=torch.float)
    y = torch.zeros((len(nodes),), dtype=torch.long)
    claim_mask = torch.zeros((len(nodes),), dtype=torch.bool)
    node_type_idx = torch.zeros((len(nodes),), dtype=torch.long)

    for n, attrs in H.nodes(data=True):
        i = nid[n]
        t = attrs.get("node_type", "claim")
        node_type_idx[i] = type_to_idx.get(t, 0)
        x[i, node_type_idx[i]] = 1.0
        x[i, T + 0] = (H.in_degree(n) / max_in) if max_in > 0 else 0.0
        x[i, T + 1] = (H.out_degree(n) / max_out) if max_out > 0 else 0.0
        x[i, T + 2] = (und.degree(n) / max_deg) if max_deg > 0 else 0.0

        if t == "claim":
            claim_mask[i] = True
            y[i] = int(labels_dict.get(str(n), 0))
        else:
            y[i] = 0

    src, dst = [], []
    for u, v in H.edges():
        src.append(nid[u]);
        dst.append(nid[v])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_index, _ = add_self_loops(edge_index, num_nodes=len(nodes))

    data = Data(x=x, edge_index=edge_index, y=y, claim_mask=claim_mask)
    # useful metadata
    data.node_ids = [str(n) for n in nodes]
    data.node_type_idx = node_type_idx
    data.claim_idx = torch.where(claim_mask)[0]
    data.claim_node_ids = [data.node_ids[i] for i in data.claim_idx.tolist()]
    return data


def describe_temporal_edges(G, cutoff):
    eligible = [(u, v, d) for u, v, d in G.edges(data=True)
                if d.get("timestamp") and d["timestamp"] <= cutoff]
    by_type = {}
    for _, _, d in eligible:
        et = d.get("edge_type", "<none>")
        by_type[et] = by_type.get(et, 0) + 1
    print(f"Edges ≤ {cutoff}: ", {k: by_type[k] for k in sorted(by_type)})


# ---------- Model ----------
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden=128, out_channels=2, dropout=0.35):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.head = torch.nn.Linear(hidden, out_channels)
        self.dropout = dropout

    def forward(self, data, return_embeddings: bool = False):
        x, ei = data.x, data.edge_index
        x = self.conv1(x, ei);
        x = F.relu(x);
        x = F.dropout(x, p=self.dropout, training=self.training)
        h = self.conv2(x, ei);
        h = F.relu(h);
        h = F.dropout(h, p=self.dropout, training=self.training)
        if return_embeddings:
            return h  # GraphSAGE embeddings
        out = self.head(h)
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


def save_lift_plot(rows, outpath, title="Temporal Graph – Lift (test)"):
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
    plt.savefig(outpath, dpi=160)
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


# ---------- Utilities: export + neighbors + (optional) t-SNE ----------
def export_embeddings(h: torch.Tensor, data: Data, out_prefix: str):
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    # All nodes
    df_all = pd.DataFrame(h.detach().cpu().numpy())
    df_all.insert(0, "node_id", data.node_ids)
    df_all.insert(1, "node_type", [ENTITY_TYPES[i] for i in data.node_type_idx.tolist()])
    df_all["is_claim"] = data.claim_mask.cpu().numpy().astype(int)
    df_all["label"] = data.y.cpu().numpy().astype(int)
    df_all.to_csv(out_prefix + "_all_nodes.csv", index=False)
    try:
        df_all.to_parquet(out_prefix + "_all_nodes.parquet", index=False)
    except Exception:
        pass

    # Claims only
    ci = data.claim_idx.cpu().tolist()
    df_claim = pd.DataFrame(h[ci].detach().cpu().numpy())
    df_claim.insert(0, "claim_node_id", [data.node_ids[i] for i in ci])
    df_claim["label"] = data.y[ci].cpu().numpy().astype(int)
    df_claim.to_csv(out_prefix + "_claims.csv", index=False)
    try:
        df_claim.to_parquet(out_prefix + "_claims.parquet", index=False)
    except Exception:
        pass
    print(f"[saved] {out_prefix}_all_nodes.(csv|parquet), {out_prefix}_claims.(csv|parquet)")


def top_k_similar_claims(h_claim: torch.Tensor, y_claim: torch.Tensor, claim_node_ids, anchor_idx: int, k=5):
    E = F.normalize(h_claim, dim=1)
    sims = (E @ E[anchor_idx:anchor_idx + 1].T).squeeze(1)
    order = torch.argsort(sims, descending=True)
    rows = []
    for i in order[:k + 1]:  # include anchor
        rows.append({
            "claim_node": str(claim_node_ids[int(i)]),
            "similarity": float(sims[int(i)].cpu()),
            "fraud_label": int(y_claim[int(i)].cpu())
        })
    return pd.DataFrame(rows)


def save_tsne_embeddings(h_claim: torch.Tensor, y_claim: torch.Tensor, outpath, title="GraphSAGE t-SNE (claims)"):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    z = h_claim.detach().cpu().numpy();
    y = y_claim.detach().cpu().numpy()
    perplexity = min(30, max(5, len(z) // 50)) if len(z) > 100 else min(30, max(5, len(z) // 3))
    tsne = TSNE(n_components=2, perplexity=perplexity, init="random", learning_rate="auto")
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


# ---------- Train / Eval; SAVE EVERYTHING ----------
def train_eval_temporal(train_data, val_data, test_data, epochs=220, lr=0.01, seed=42,
                        outdir="data"):
    torch.manual_seed(seed)
    os.makedirs(outdir, exist_ok=True)
    figs_dir = os.path.join(outdir, "figs_model_4")
    tables_dir = os.path.join(outdir, "tables_model_4")
    os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    # Class weights from train claims
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

        if bad >= patience:
            break

        if ep % 25 == 0 or ep == epochs:
            print(f"Ep {ep:03d} | train loss {loss:.4f} | val F1 {mva['f1']:.3f} "
                  f"(t*={t_star:.2f}) | val acc {mva['acc']:.3f} prec {mva['precision']:.3f} rec {mva['recall']:.3f}")

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

    # Metrics + lift
    m = confusion_and_metrics(y_te, pred_te)
    print(f"\nFINAL TEST @t*={best_t:.2f}: acc={m['acc']:.3f} prec={m['precision']:.3f} "
          f"rec={m['recall']:.3f} f1={m['f1']:.3f}  "
          f"cm=[tp:{m['tp']} fp:{m['fp']} tn:{m['tn']} fn:{m['fn']}]")

    rows = lift_table(probs_te, y_te, bins=10)
    lift_png = os.path.join(figs_dir, "lift_test.png")
    save_lift_plot(rows, lift_png, title="Lift (test)")
    print(f"[saved] {lift_png}")

    # ---------- GraphSAGE embeddings (this is what you want) ----------
    with torch.no_grad():
        h_train = model(train_data, return_embeddings=True)
        h_val = model(val_data, return_embeddings=True)
        h_test = model(test_data, return_embeddings=True)

    # Save embeddings
    export_embeddings(h_test, test_data, out_prefix=os.path.join(outdir, "graphsage_test"))
    # Optional: also save train/val
    # export_embeddings(h_train, train_data, out_prefix=os.path.join(outdir, "graphsage_train"))
    # export_embeddings(h_val,   val_data,   out_prefix=os.path.join(outdir, "graphsage_val"))

    # Claims-only view (neighbors + t-SNE saved)
    h_claim = h_test[test_data.claim_idx]
    y_claim = test_data.y[test_data.claim_idx]

    if len(y_claim) > 0:
        # anchor = highest-scoring true fraud (or highest score if no positives)
        mask_true = (y_te == 1)
        if mask_true.any():
            order = torch.argsort(probs_te, descending=True)
            anchor = order[mask_true[order]][0].item()
        else:
            anchor = int(torch.argmax(probs_te).item())

        sim_df = top_k_similar_claims(
            h_claim, y_claim,
            claim_node_ids=test_data.claim_node_ids,
            anchor_idx=anchor, k=5
        )
        sim_csv = os.path.join(tables_dir, "similar_claims_graphsage.csv")
        sim_df.to_csv(sim_csv, index=False)
        print(f"[saved] {sim_csv}")

        tsne_png = os.path.join(figs_dir, "tsne_graphsage_claims.png")
        save_tsne_embeddings(h_claim, y_claim, tsne_png, title="GraphSAGE t-SNE (claims)")

    return model, rows, m, h_test


def main():
    # Load data & graph
    df = pd.read_csv("data/sy_dataset_1.csv", parse_dates=["claim_date"])
    with open("data/temporal_graph_with_edge_attrs.gpickle", "rb") as f:
        G = pickle.load(f)

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

    # describe node
    print('describe_temporal_edges')
    describe_temporal_edges(G, t_train_end)
    describe_temporal_edges(G, t_val_end)
    describe_temporal_edges(G, t_test_end)

    # Train/validate/test and export embeddings + figures
    _model, _lift_rows, _metrics, _h_test = train_eval_temporal(
        data_train, data_val, data_test, epochs=220, lr=0.01, seed=42, outdir="data"
    )


if __name__ == "__main__":
    main()
