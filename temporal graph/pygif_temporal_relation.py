import matplotlib.pyplot as plt
import networkx as nx
import imageio
import os

# Create output directory for frames
os.makedirs("data/gif_frames", exist_ok=True)

# Define frames list
frames = []

# Common node and edge styling
node_styles = {
    "Claim_id_t-1": {"color": "red", "label": "Claim_id\nt-1"},
    "Phone": {"color": "orange", "label": "Phone"},
    "Claim_id_t": {"color": "red", "label": "Claim_id\nt"},
}

# Define the steps in the animation
steps = [
    {"nodes": ["Claim_id_t-1"], "edges": []},
    {"nodes": ["Claim_id_t-1", "Phone"], "edges": [("Claim_id_t-1", "Phone")]},
    {"nodes": ["Claim_id_t-1", "Phone", "Claim_id_t"], "edges": [("Claim_id_t-1", "Phone"), ("Claim_id_t", "Phone")]},
    {"nodes": ["Claim_id_t-1", "Phone", "Claim_id_t"], "edges": [("Claim_id_t-1", "Phone"), ("Claim_id_t", "Phone"), ("Claim_id_t-1", "Claim_id_t")]},
]

# Generate frames
for i, step in enumerate(steps):
    G = nx.DiGraph()
    G.add_nodes_from(step["nodes"])
    G.add_edges_from(step["edges"])

    pos = {
        "Claim_id_t-1": (-1, 0),
        "Phone": (0, 1),
        "Claim_id_t": (1, 0)
    }

    plt.figure(figsize=(6, 4))
    for node in G.nodes():
        nx.draw_networkx_nodes(G, pos, nodelist=[node],
                               node_color=node_styles[node]["color"],
                               node_size=1500,
                               alpha=0.8)
        nx.draw_networkx_labels(G, pos, labels={node: node_styles[node]["label"]}, font_size=10)

    edge_colors = ['red' if (u == "Claim_id_t-1" and v == "Claim_id_t") else 'black' for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color=edge_colors, arrows=True, arrowstyle="->")

    plt.axis("off")
    plt.title(f"Step {i + 1}", fontsize=12)
    frame_path = f"data/gif_frames/frame_{i:02d}.png"
    plt.savefig(frame_path)
    frames.append(imageio.imread(frame_path))
    plt.close()

# Save GIF
gif_path = "data/claim_temporal_graph.gif"
imageio.mimsave(gif_path, frames, duration=1.2)

gif_path
