import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load data and graph
df = pd.read_csv("data/sy_dataset_1.csv")
df["claim_date"] = pd.to_datetime(df["claim_date"])
df["claim_id"] = df["claim_id"].astype(str)
G = nx.read_gpickle("data/temporal_graph_with_edge_attrs.gpickle")

# Entity colors
entity_colors = {
    "insurer_phone_number": "gold",
    "insurer_license_plate": "orange",
    "insurer_email": "green",
    "insurer_address": "lightgreen",
    "repair_shop": "purple",
    "bank_account": "cyan",
    "claim_location": "brown",
    "third_party_license_plate": "magenta"
}

# Sidebar
st.title("🕸️ Temporal Claim Graph Viewer")

available_dates = sorted(df["claim_date"].dt.date.unique())
selected_date = st.selectbox("Select a date", available_dates)
selected_date = pd.to_datetime(selected_date)

show_all_nodes = st.checkbox("Show entity nodes", value=True)
show_entity_labels_on_edges = st.checkbox("Show shared entity name on claim–claim edges", value=True)

# Filter graph up to selected date
valid_claims = df[df["claim_date"] <= selected_date]["claim_id"].tolist()
G_sub = G.subgraph([n for n in G.nodes if (
        (G.nodes[n].get("node_type") == "claim" and n in valid_claims) or
        G.nodes[n].get("node_type") == "entity"
)]).copy()

# Remove future edges
edges_to_remove = [(u, v) for u, v, d in G_sub.edges(data=True)
                   if d.get("timestamp") and pd.to_datetime(d["timestamp"]) > selected_date]
G_sub.remove_edges_from(edges_to_remove)

# Add entity nodes and edges (if checked)
if show_all_nodes:
    for u, v, d in G.edges(data=True):
        if d.get("edge_type") == "claim-entity" and pd.to_datetime(d.get("timestamp", "1900-01-01")) <= selected_date:
            if u in valid_claims:
                G_sub.add_node(v, **G.nodes[v])
                G_sub.add_edge(u, v, **d)

# Define layout
pos = nx.spring_layout(G_sub, seed=42)
plt.figure(figsize=(14, 10))

# Node sets
fraud_claims = df[df["fraud"] == 1]["claim_id"].tolist()
normal_claims = df[df["fraud"] == 0]["claim_id"].tolist()

nx.draw_networkx_nodes(G_sub, pos,
                       nodelist=[n for n in G_sub.nodes if n in fraud_claims],
                       node_color="red", node_size=600, label="Fraud Claim")

nx.draw_networkx_nodes(G_sub, pos,
                       nodelist=[n for n in G_sub.nodes if n in normal_claims],
                       node_color="lightblue", node_size=400, label="Normal Claim")

if show_all_nodes:
    for entity_type, color in entity_colors.items():
        nodes = [n for n, d in G_sub.nodes(data=True) if d.get("node_type") == entity_type or entity_type in str(n)]
        if nodes:
            nx.draw_networkx_nodes(G_sub, pos, nodelist=nodes, node_color=color, node_size=300, label=entity_type)

# Separate edge types
claim_claim_edges = [(u, v) for u, v, d in G_sub.edges(data=True) if d.get("edge_type") == "claim-claim"]
entity_edges = [(u, v) for u, v, d in G_sub.edges(data=True) if d.get("edge_type") == "claim-entity"]

# Draw edges
nx.draw_networkx_edges(G_sub, pos, edgelist=entity_edges, edge_color='gray', alpha=0.4)
nx.draw_networkx_edges(G_sub, pos, edgelist=claim_claim_edges, edge_color='black', arrows=True, arrowstyle='->')

# Edge labels (for claim-claim only)
if show_entity_labels_on_edges:
    edge_labels = {
        (u, v): f"{d.get('via', '')}\n{str(pd.to_datetime(d.get('timestamp')).date())}"
        for u, v, d in G_sub.edges(data=True)
        if d.get("edge_type") == "claim-claim" and d.get("timestamp")
    }
    nx.draw_networkx_edge_labels(G_sub, pos, edge_labels=edge_labels, font_size=7)

# Node labels
nx.draw_networkx_labels(G_sub, pos, font_size=8)

plt.title(f"Graph on {selected_date.date()} ({len(G_sub.nodes())} nodes)", fontsize=14)
plt.axis("off")
plt.legend(markerscale=0.6, fontsize=8)
st.pyplot(plt)

