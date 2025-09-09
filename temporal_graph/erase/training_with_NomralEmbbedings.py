# 03_train_temporal.py
import os
import pickle
import pandas as pd
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import add_self_loops
import matplotlib.pyplot as plt

# t-SNE (install scikit-learn if needed)
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

def describe_temporal_edges(G, cutoff):
    eligible = [(u,v,d) for u,v,d in G.edges(data=True)
                if d.get("timestamp") and d["timestamp"] <= cutoff]
    by_type = {}
    for _,_,d in eligible:
        et = d.get("edge_type","<none>")
        by_type[et] = by_type.get(et, 0) + 1
    print(f"Edges ≤ {cutoff}: ", {k: by_type[k] for k in sorted(by_type)})


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

    # Features: one-hot(type) + [in,out,undeg] normalized within H
    T = len(ENTITY_TYPES)
    und = H.to_undirected()
    max_in  = max((H.in_degree(n)  for n in nodes), default=1)
    max_out = max((H.out_degree(n) for n in nodes), default=1)
    max_deg = max((und.degree(n)   for n in nodes), default=1)

    x = torch.zeros((len(nodes), T + 3), dtype=torch.float)
    y = torch.zeros((len(nodes),), dtype=torch.long)
    claim_mask = torch.zeros((len(nodes),), dtype=torch.bool)

    for n, attrs in H.nodes(data=True):
        i = nid[n]
        t = attrs.get("node_type", "claim")
        x[i, type_to_idx.get(t, 0)] = 1.0
        x[i, T + 0] = (H.in_degree(n)  / max_in)  if max_in  > 0 else 0.0
        x[i, T + 1] = (H.out_degree(n) / max_out) if max_out > 0 else 0.0
        x[i, T + 2] = (und.degree(n)   / max_deg) if max_deg > 0 else 0.0

        if t == "claim":
            claim_mask[i] = True
            y[i] = int(labels_dict.get(str(n), 0))
        else:
            y[i] = 0

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

def plot_lift(rows, title="Lift (test)"):
    xs   = [r["bin"] for r in rows]
    lifts = [r["lift"] for r in rows]
    cum   = [r["cum_capture"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(xs, lifts, alpha=0.75)
    ax1.axhline(1.0, linestyle="--", linewidth=1)
    ax1.set_xlabel("Decile (1 = highest score)"); ax1.set_ylabel("Lift"); ax1.set_xticks(xs)
    ax2 = ax1.twinx(); ax2.plot(xs, cum, marker="o"); ax2.set_ylabel("Cumulative capture"); ax2.set_ylim(0, 1.05)
    plt.title(title); plt.tight_layout(); plt.show()

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
def plot_tsne_embeddings(embeddings: torch.Tensor, labels: torch.Tensor, title="t-SNE of claim embeddings"):
    """
    embeddings: [num_claims, d], labels: [num_claims] in {0,1}
    """
    if not _HAS_TSNE:
        print("scikit-learn not available; skipping t-SNE plot.")
        return
    z = embeddings.detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    # sensible perplexity for a wide range of N
    perplexity = min(30, max(5, len(z)//50)) if len(z) > 100 else min(30, max(5, len(z)//3))
    tsne = TSNE(n_components=2, perplexity=perplexity, init="random", learning_rate="auto")
    z2 = tsne.fit_transform(z)

    plt.figure(figsize=(7.5, 5.5))
    nf = y == 0; fr = y == 1
    plt.scatter(z2[nf,0], z2[nf,1], s=6, alpha=0.7, label="no fraud")
    plt.scatter(z2[fr,0], z2[fr,1], s=12, alpha=0.9, label="fraud")
    plt.legend(loc="best"); plt.title(title); plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
    plt.tight_layout(); plt.show()

def top_k_similar(embeddings: torch.Tensor, labels: torch.Tensor, node_ids: list, anchor_idx: int, k=5):
    """
    embeddings: [N_claims, d] for CLAIM NODES ONLY (same order as node_ids)
    labels: [N_claims]
    node_ids: list of original node IDs matching embeddings row order
    anchor_idx: index in that array for the anchor claim
    """
    E = F.normalize(embeddings, dim=1)                 # cosine similarity
    anchor = E[anchor_idx:anchor_idx+1]                # [1,d]
    sims = (E @ anchor.T).squeeze(1)                   # [N]
    order = torch.argsort(sims, descending=True)
    rows = []
    for i in order[:k+1]:                              # include anchor at rank 1
        rows.append({
            "claim_node": str(node_ids[int(i)]),
            "similarity": float(sims[int(i)].cpu()),
            "fraud_label": int(labels[int(i)].cpu())
        })
    import pandas as pd
    return pd.DataFrame(rows)

# ---------- Train on train graph; tune on val; test on test ----------
def train_eval_temporal(train_data, val_data, test_data, epochs=220, lr=0.01, seed=42):
    torch.manual_seed(seed)
    # Class weights from train
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
            # threshold picked on VAL to maximize F1
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
    print("\nbin |   n   | rate   | lift | cum_capture")
    for r in rows:
        print(f"{r['bin']:>3} | {r['n']:>5} | {r['fraud_rate']:.4f} | {r['lift']:.2f} | {r['cum_capture']:.3f}")

    plot_lift(rows, title="Temporal Graph – Lift (test)")

    print("Test claims:", int(test_data.claim_mask.sum()))
    print("Positives in test:", int((test_data.y[test_data.claim_mask]==1).sum()))
    print("Chosen threshold (from val):", best_t)

    # ---------- Embedding visualization + neighbors table ----------
    # claim indices in TEST graph (order aligns with probs_te / y_te)
    claim_ids_te = torch.where(test_data.claim_mask)[0]

    # get penultimate embeddings for all nodes, then take claims-only
    with torch.no_grad():
        h_test = model(test_data, return_embeddings=True)        # [N_nodes, hidden]
    h_claim = h_test[claim_ids_te]                                # [N_claims, hidden]
    y_claim = test_data.y[claim_ids_te]

    # t-SNE plot (skips automatically if sklearn isn't available)
    plot_tsne_embeddings(h_claim, y_claim, title="Temporal Graph – t-SNE (claims)")

    # pick anchor: highest-scored TRUE fraud (fallback: highest score overall)
    mask_true_fraud = (y_te == 1)
    if mask_true_fraud.any():
        te_order = torch.argsort(probs_te, descending=True)
        anchor_test_pos = te_order[mask_true_fraud[te_order]][0]  # index within claims order
    else:
        anchor_test_pos = torch.argmax(probs_te)

    df_sim = top_k_similar(
        h_claim, y_te,
        node_ids=claim_ids_te.cpu().tolist(),   # these are node indexes; swap with real claim_ids if you have a mapping
        anchor_idx=int(anchor_test_pos),
        k=5
    )
    print("\nSimilar claims to the selected (fraudulent) claim:")
    print(df_sim.to_string(index=False))

    return model, rows, m

def main():
    # Load data & graph
    df = pd.read_csv("../data/sy_dataset_1.csv", parse_dates=["claim_date"])
    with open("../data/temporal_graph_with_edge_attrs.gpickle", "rb") as f:
        G = pickle.load(f)

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

    # describe node
    print('describe_temporal_edges')
    describe_temporal_edges(G, t_train_end)
    describe_temporal_edges(G, t_val_end)
    describe_temporal_edges(G, t_test_end)

    # Train/validate/test
    _model, _lift_rows, _metrics = train_eval_temporal(
        data_train, data_val, data_test, epochs=220, lr=0.01, seed=42
    )

if __name__ == "__main__":
    main()
