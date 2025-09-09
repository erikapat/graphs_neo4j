import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load data and graph
df = pd.read_csv("data/sy_dataset_1.csv")
df["claim_date"] = pd.to_datetime(df["claim_date"])
df["claim_id"] = df["claim_id"].astype(str)
G = nx.read_gpickle("data/temporal_graph_with_edge_attrs.gpickle")

# Load features
features_df = pd.read_csv("data/graph_features_from_edge_attrs.csv")
features_df["claim_date"] = pd.to_datetime(features_df["claim_date"])

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
show_only_shared_entities = st.checkbox("Highlight only entities used to connect claims", value=False)

# Filter graph up to selected date
valid_claims = df[df["claim_date"] <= selected_date]["claim_id"].tolist()
G_sub = nx.DiGraph()

# Add claim nodes
for n in G.nodes:
    if G.nodes[n].get("node_type") == "claim" and n in valid_claims:
        G_sub.add_node(n, **G.nodes[n])

# Determine shared entities between claims
shared_entities = set()
for u, v, d in G.edges(data=True):
    if (
            d.get("edge_type") == "claim-claim"
            and u in valid_claims and v in valid_claims
            and pd.to_datetime(d.get("timestamp", "1900-01-01")) <= selected_date
    ):
        col = d.get("via")
        val = d.get("shared_value")
        if col and val:
            entity_node = f"{col}:{val}"
            shared_entities.add(entity_node)

# Add edges
for u, v, d in G.edges(data=True):
    if pd.to_datetime(d.get("timestamp", "1900-01-01")) > selected_date:
        continue

    if d.get("edge_type") == "claim-claim" and u in valid_claims and v in valid_claims:
        G_sub.add_edge(u, v, **d)

    elif d.get("edge_type") == "claim-entity" and show_all_nodes and u in valid_claims:
        if not show_only_shared_entities or v in shared_entities:
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
        nodes = [n for n, d in G_sub.nodes(data=True)
                 if (d.get("node_type") == entity_type or entity_type in str(n))
                 and (not show_only_shared_entities or n in shared_entities)]
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

# ---------------------- Feature Tables ----------------------
st.subheader("📈 Graph-Based Features")
day_features = features_df[features_df["claim_date"] == selected_date]

main_graph_features = day_features[[
    "claim_id", "claim_date", "component_size",
    "degree_centrality", "closeness_centrality", "betweenness_centrality"
]]
st.markdown("**Main Graph Features:**")
st.dataframe(main_graph_features)

entity_connection_features = day_features[[
    col for col in day_features.columns if col.endswith("_used_to_connect_claims")
]]
if not entity_connection_features.empty:
    st.markdown("**Entity Connection Counts (claim-to-claim):**")
    st.dataframe(pd.concat([day_features[["claim_id"]], entity_connection_features], axis=1))

unique_entity_counts = day_features[[
    col for col in day_features.columns if col.startswith("unique_")
]]
if not unique_entity_counts.empty:
    st.markdown("**Unique Entities in Component:**")
    st.dataframe(pd.concat([day_features[["claim_id"]], unique_entity_counts], axis=1))

