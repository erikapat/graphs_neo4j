# app.py
import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pickle

# ---------------- Load data and graph ----------------
df = pd.read_csv("data/sy_dataset_1.csv")
df["claim_date"] = pd.to_datetime(df["claim_date"])
df["claim_id"] = df["claim_id"].astype(str)

with open("data/temporal_graph_with_edge_attrs.gpickle", "rb") as f:
    G = pickle.load(f)

# Load features (ignore if missing)
features_df = None
try:
    _tmp = pd.read_csv("data/graph_features_from_edge_attrs.csv")
    _tmp["claim_date"] = pd.to_datetime(_tmp["claim_date"])
    features_df = _tmp
except Exception:
    features_df = None

# ---------------- UI ----------------
st.title("🕸️ Temporal Claim Graph Viewer")

available_dates = sorted(df["claim_date"].dt.date.unique())
selected_date = st.selectbox("Select a date", available_dates)
selected_date = pd.to_datetime(selected_date)
day_end = selected_date.normalize() + pd.Timedelta(days=1)   # include full day

show_all_nodes = st.checkbox("Show entity nodes", value=True)
show_entity_labels_on_edges = st.checkbox("Show shared entity name on claim–claim edges", value=True)
show_only_shared_entities = st.checkbox("Highlight only entities used to connect claims", value=False)

# For rendering only (does NOT change the data)
max_draw_nodes = st.slider("Max nodes to draw (for visualization only)", 100, 3000, 800, step=100)

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

# ---------------- Build temporal subgraph ----------------
valid_claims = set(df.loc[df["claim_date"] < day_end, "claim_id"].astype(str))

from datetime import datetime as _dt
connecting_entities = set()
for u, v, d in G.edges(data=True):
    ts = d.get("timestamp")
    ts = pd.to_datetime(ts) if ts is not None else _dt(1900, 1, 1)
    if ts >= day_end:
        continue
    if d.get("edge_type") == "claim-claim":
        if str(u) in valid_claims and str(v) in valid_claims:
            via, val = d.get("via"), d.get("shared_value")
            if via and val:
                connecting_entities.add(f"{via}:{val}")

G_sub = nx.DiGraph()

for u, v, d in G.edges(data=True):
    ts = d.get("timestamp")
    ts = pd.to_datetime(ts) if ts is not None else _dt(1900, 1, 1)
    if ts >= day_end:
        continue

    et = d.get("edge_type")

    if et == "claim-claim":
        if str(u) in valid_claims and str(v) in valid_claims:
            if u not in G_sub: G_sub.add_node(u, **G.nodes[u])
            if v not in G_sub: G_sub.add_node(v, **G.nodes[v])
            G_sub.add_edge(u, v, **d)

    elif et == "claim-entity" and show_all_nodes and str(u) in valid_claims:
        if (not show_only_shared_entities) or (v in connecting_entities):
            if u not in G_sub: G_sub.add_node(u, **G.nodes[u])
            if v not in G_sub: G_sub.add_node(v, **G.nodes[v])
            G_sub.add_edge(u, v, **d)

# Ensure isolated claims are visible
for cid in valid_claims:
    if cid in G and cid not in G_sub:
        G_sub.add_node(cid, **G.nodes[cid])

# --- Diagnostics on full subgraph ---
st.caption(
    f"claims≤{selected_date.date()}: {len(valid_claims)} | "
    f"graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges | "
    f"subgraph: {G_sub.number_of_nodes()} nodes, {G_sub.number_of_edges()} edges"
)

# ---------------- Choose what to DRAW (never changes the data) ----------------
G_draw = G_sub
draw_note = ""

# If too big, switch to largest connected component (undirected view) to guarantee a visible picture
if G_draw.number_of_nodes() > max_draw_nodes:
    und = G_draw.to_undirected()
    comps = list(nx.connected_components(und))
    if comps:
        biggest = max(comps, key=len)
        G_draw = G_draw.subgraph(biggest).copy()
        draw_note = f"Drawing largest component only (size={G_draw.number_of_nodes()}). "

# If still too big, sample nodes for rendering only
if G_draw.number_of_nodes() > max_draw_nodes:
    # keep all claims, then fill with entities up to the cap
    claims_keep = [n for n, d in G_draw.nodes(data=True) if d.get("node_type") == "claim"]
    others = [n for n in G_draw.nodes if n not in claims_keep]
    keep = set(claims_keep)
    if len(keep) < max_draw_nodes:
        keep.update(others[: max_draw_nodes - len(keep)])
    G_draw = G_draw.subgraph(keep).copy()
    draw_note += f"Sampled to {G_draw.number_of_nodes()} nodes for rendering."

if draw_note:
    st.info(draw_note)

# ---------------- Draw ----------------
if G_draw.number_of_nodes() == 0:
    st.warning("No nodes/edges for this date filter. Try a later date.")
else:
    fig, ax = plt.subplots(figsize=(14, 10))
    # Faster, stable layout for larger graphs
    try:
        pos = nx.spring_layout(G_draw, seed=42, k=None, iterations=50, dim=2)
    except Exception:
        pos = nx.kamada_kawai_layout(G_draw)

    fraud_claims = set(df.loc[df["is_fraud"] == 1, "claim_id"])
    normal_claims = set(df.loc[df["is_fraud"] == 0, "claim_id"])

    # Claims (fraud / normal)
    fraud_nodes = [n for n in G_draw.nodes if n in fraud_claims]
    normal_nodes = [n for n in G_draw.nodes if n in normal_claims]
    if fraud_nodes:
        nx.draw_networkx_nodes(G_draw, pos, nodelist=fraud_nodes, node_color="red", node_size=600, label="Fraud Claim", ax=ax)
    if normal_nodes:
        nx.draw_networkx_nodes(G_draw, pos, nodelist=normal_nodes, node_color="lightblue", node_size=400, label="Normal Claim", ax=ax)

    # Entities
    if show_all_nodes:
        for entity_type, color in entity_colors.items():
            nodes = [n for n, d in G_draw.nodes(data=True)
                     if (d.get("node_type") == entity_type or entity_type in str(n))
                     and (not show_only_shared_entities or n in connecting_entities)]
            if nodes:
                nx.draw_networkx_nodes(G_draw, pos, nodelist=nodes, node_color=color, node_size=300, label=entity_type, ax=ax)

    # Edges
    claim_claim_edges = [(u, v) for u, v, d in G_draw.edges(data=True) if d.get("edge_type") == "claim-claim"]
    entity_edges      = [(u, v) for u, v, d in G_draw.edges(data=True) if d.get("edge_type") == "claim-entity"]

    if entity_edges:
        nx.draw_networkx_edges(G_draw, pos, edgelist=entity_edges, edge_color='gray', alpha=0.35, ax=ax)
    if claim_claim_edges:
        nx.draw_networkx_edges(G_draw, pos, edgelist=claim_claim_edges, edge_color='black', arrows=True, arrowstyle='->', ax=ax)

    # Edge labels only for claim-claim (optional)
    if show_entity_labels_on_edges and claim_claim_edges:
        edge_labels = {}
        for u, v, d in G_draw.edges(data=True):
            if d.get("edge_type") == "claim-claim" and d.get("timestamp"):
                ts = pd.to_datetime(d.get("timestamp")).date()
                edge_labels[(u, v)] = f"{d.get('via','')}\n{ts}"
        if edge_labels:
            nx.draw_networkx_edge_labels(G_draw, pos, edge_labels=edge_labels, font_size=7, ax=ax)

    nx.draw_networkx_labels(G_draw, pos, font_size=8, ax=ax)

    ax.set_title(f"Graph up to {selected_date.date()} "
                 f"(drawn {G_draw.number_of_nodes()} nodes / {G_draw.number_of_edges()} edges)", fontsize=14)
    ax.axis("off")
    ax.legend(markerscale=0.6, fontsize=8, loc="upper right")
    st.pyplot(fig)

# ---------------- Feature Tables (optional) ----------------
st.subheader("📈 Graph-Based Features")
if features_df is None:
    st.info("Feature file not found (ignored as requested).")
else:
    day_mask = (features_df["claim_date"] < day_end)
    day_features = features_df[day_mask & (features_df["claim_date"].dt.normalize() == selected_date.normalize())]

    main_cols = ["claim_id", "claim_date", "component_size",
                 "degree_centrality", "closeness_centrality", "betweenness_centrality"]
    have_main = [c for c in main_cols if c in day_features.columns]
    if have_main and not day_features.empty:
        st.markdown("**Main Graph Features:**")
        st.dataframe(day_features[have_main])

    conn_cols = [c for c in day_features.columns if c.endswith("_used_to_connect_claims")]
    if conn_cols and not day_features.empty:
        st.markdown("**Entity Connection Counts (claim-to-claim):**")
        st.dataframe(day_features[["claim_id"] + conn_cols])

    uniq_cols = [c for c in day_features.columns if c.startswith("unique_")]
    if uniq_cols and not day_features.empty:
        st.markdown("**Unique Entities in Component:**")
        st.dataframe(day_features[["claim_id"] + uniq_cols])
