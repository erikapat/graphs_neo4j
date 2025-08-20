import os
import pickle
import pandas as pd
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
from torch_geometric.nn import SAGEConv
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# --------- Load or build your graph (example) ---------

# Example loading function (replace with your actual graph)
def load_graph():
    if os.path.exists("data/incremental_graph.gpickle"):
        G = nx.read_gpickle("data/incremental_graph.gpickle")
    else:
        G = nx.DiGraph()
        # Build your graph or load data here
    return G

# --------- Convert NetworkX graph to PyG Data ---------

def nx_to_pyg_data(G, claim_node_type='claim'):
    claim_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == claim_node_type]
    mapping = {node: i for i, node in enumerate(claim_nodes)}
    G_claim = G.subgraph(claim_nodes).copy()
    G_claim = nx.relabel_nodes(G_claim, mapping)

    features = []
    for n in G_claim.nodes:
        data = G_claim.nodes[n]
        feat = [
            data.get('degree_centrality', 0),
            data.get('closeness_centrality', 0),
            data.get('betweenness_centrality', 0),
        ]
        features.append(feat)
    x = torch.tensor(features, dtype=torch.float)

    # Fraud label: 1 if fraud, else 0 (dummy here, replace with real labels)
    y = torch.tensor([G_claim.nodes[n].get('is_fraud', 0) for n in G_claim.nodes], dtype=torch.long)

    data = from_networkx(G_claim)
    data.x = x
    data.y = y

    return data, mapping

# --------- Define GraphSAGE model ---------

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
        return x.detach().cpu()

# --------- Training and testing functions ---------

def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, data.y)
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, data):
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    correct = (pred == data.y).sum()
    acc = int(correct) / data.num_nodes
    return acc

# --------- t-SNE visualization ---------

def plot_tsne(embeddings, labels):
    tsne = TSNE(n_components=2, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(emb_2d[:,0], emb_2d[:,1], c=labels, cmap='coolwarm', alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Fraud")
    plt.title("t-SNE projection of GraphSAGE embeddings")
    plt.show()

# --------- Main pipeline ---------

def main():
    # Load your graph
    G = load_graph()

    # Convert to PyG data format
    data, mapping = nx_to_pyg_data(G)

    # Initialize model, optimizer, loss
    model = GraphSAGE(in_channels=data.num_node_features, hidden_channels=32, out_channels=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    # Train
    for epoch in range(1, 201):
        loss = train(model, data, optimizer, criterion)
        if epoch % 20 == 0:
            acc = test(model, data)
            print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")

    # Extract embeddings for claims
    embeddings = model.get_embeddings(data).numpy()
    labels = data.y.numpy()

    # Plot t-SNE
    plot_tsne(embeddings, labels)

if __name__ == "__main__":
    main()
