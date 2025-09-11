# ---------------- GraphSAGE + t-SNE on your constructed temporal graph ----------------
# Uses pickle for graph I/O. Avoids tensor.numpy(). Limits threads during TSNE to dodge
# libiomp/libomp conflicts. Reads labels from data/sy_dataset_1.csv.
# --------------------------------------------------------------------------------------

import os
# Limit threads early to reduce OpenMP clashes during sklearn ops
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import pickle
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
from torch_geometric.nn import SAGEConv
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits
import matplotlib.pyplot as plt


# ---------------- 1) Load graph (pickled by your construction code) ----------------

def load_graph(path="data/temporal_graph_with_edge_attrs.gpickle"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Graph file not found at {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------- 2) Read labels from CSV ----------------

def load_labels_from_df(csv_path="data/sy_dataset_1.csv"):
    df = pd.read_csv(csv_path)
    df["claim_id"] = df["claim_id"].astype(str)
    label_col = "is_fraud" if "is_fraud" in df.columns else ("fraud" if "fraud" in df.columns else None)
    if label_col is None:
        print("⚠️  No 'is_fraud' or 'fraud' column in CSV. All labels default to 0.")
        return {}
    labels = df[["claim_id", label_col]].copy()
    labels[label_col] = labels[label_col].fillna(0).astype(int)
    return dict(zip(labels["claim_id"], labels[label_col]))


# ---------------- 3) Convert NetworkX -> PyG (claims only) ----------------

def nx_to_pyg_claim_graph(G, labels_dict=None, claim_node_type="claim"):
    # pick claim nodes
    claim_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == claim_node_type]
    if not claim_nodes:
        claim_nodes = [n for n, d in G.nodes(data=True) if "claim_date" in d]
    if not claim_nodes:
        raise ValueError("No claim nodes found (node_type='claim' or 'claim_date').")

    # claim-only subgraph and relabel 0..N-1
    G_claim = G.subgraph(claim_nodes).copy()
    mapping = {node: i for i, node in enumerate(G_claim.nodes())}
    inv_mapping = {i: node for node, i in mapping.items()}
    G_claim = nx.relabel_nodes(G_claim, mapping)

    # simple degree-based features
    N = G_claim.number_of_nodes()
    denom = max(1, N - 1)
    feats = []
    for n in G_claim.nodes:
        deg = G_claim.degree[n] / denom
        indeg = G_claim.in_degree[n] / denom if isinstance(G_claim, nx.DiGraph) else deg
        outdeg = G_claim.out_degree[n] / denom if isinstance(G_claim, nx.DiGraph) else deg
        feats.append([deg, indeg, outdeg])
    x = torch.tensor(feats, dtype=torch.float)

    # labels from CSV dict (fallback to node attrs if present)
    y_list = []
    for n in G_claim.nodes:
        orig_id = inv_mapping[n]
        if labels_dict is not None:
            y_list.append(int(labels_dict.get(str(orig_id), 0)))
        else:
            attrs = G_claim.nodes[n]
            y_list.append(int(attrs.get("is_fraud", attrs.get("fraud", 0))))
    y = torch.tensor(y_list, dtype=torch.long)

    # diagnostics w/out NumPy
    vals, counts = torch.unique(y, return_counts=True)
    print("Label distribution (value:count):", {int(v): int(c) for v, c in zip(vals, counts)})

    data = from_networkx(G_claim)  # builds edge_index etc.
    data.x = x
    data.y = y
    return data


# ---------------- 4) GraphSAGE ----------------

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def get_embeddings(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        return x.detach().cpu()  # tensor (no .numpy())


# ---------------- 5) Train / Eval ----------------

def train_one_epoch(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, data.y)
    loss.backward()
    optimizer.step()
    return float(loss.item())

def accuracy(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
        return (pred == data.y).float().mean().item()


# ---------------- 6) 2D embedding plot (t-SNE with safe fallbacks) ----------------

def plot_embedding_2d(emb_tensor, labels, title="GraphSAGE embeddings (claims only)"):
    # torch tensor -> list -> numpy (avoids torch->numpy bridge)
    if hasattr(emb_tensor, "detach"):
        emb_list = emb_tensor.detach().cpu().tolist()
    else:
        emb_list = emb_tensor
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().tolist()

    X = np.asarray(emb_list, dtype=float)
    y = np.asarray(labels)

    n = X.shape[0]
    if n < 2:
        print("Not enough samples to plot.")
        return

    # sanitize & standardize
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)

    perp = min(30, max(5, n - 1))
    with threadpool_limits(limits=1):
        try:
            X2 = TSNE(n_components=2, random_state=42, perplexity=perp, init="random", learning_rate="auto").fit_transform(X)
        except Exception as e:
            print(f"t-SNE failed ({e}). Falling back to PCA.")
            from sklearn.decomposition import PCA
            X2 = PCA(n_components=2, random_state=42).fit_transform(X)

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(X2[:, 0], X2[:, 1], c=y, cmap="coolwarm", alpha=0.85)
    plt.legend(*sc.legend_elements(), title="Fraud")
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    plt.show()


# ---------------- 7) Main ----------------

def main():
    # load graph and labels
    G = load_graph("data/temporal_graph_with_edge_attrs.gpickle")
    labels_dict = load_labels_from_df("data/sy_dataset_1.csv")

    # PyG data (claims only)
    data = nx_to_pyg_claim_graph(G, labels_dict=labels_dict)

    # model
    model = GraphSAGE(in_channels=data.num_node_features, hidden_channels=32, out_channels=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    # train
    for epoch in range(1, 201):
        loss = train_one_epoch(model, data, optimizer, criterion)
        if epoch % 20 == 0:
            acc = accuracy(model, data)
            print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")

    # embeddings + plot
    emb = model.get_embeddings(data)
    plot_embedding_2d(emb, data.y, title="GraphSAGE embeddings (claims only)")

if __name__ == "__main__":
    main()



