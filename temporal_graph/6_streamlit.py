# app.py...
import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from datetime import datetime as _dt

# ---------------- Load data and graph ----------------
df = pd.read_csv("data/sy_dataset_1.csv")
df["claim_date"] = pd.to_datetime(df["claim_date"])
df["claim_id"] = df["claim_id"].astype(str)

with open("data/temporal_graph_with_edge_attrs.gpickle", "rb") as f:
    G = pickle.load(f)

# ---------- helpers: robust claim-id -> node-key lookup ----------
def claim_node_key_in(graph, claim_id_str: str):
    """Return the actual node key used in `graph` for this claim_id (str or int), or None."""
    if claim_id_str in graph:
        return claim_id_str
    if claim_id_str.isdigit():
        cid_int = int(claim_id_str)
        if cid_int in graph:
            return cid_int
    return None

def ensure_claim_node(graph_dst, graph_src, claim_id_str: str):
    """Ensure claim node is present in graph_dst by copying attributes from graph_src."""
    key = claim_node_key_in(graph_src, claim_id_str)
    if key is None:
        return False
    if key not in graph_dst:
        graph_dst.add_node(key, **graph_src.nodes[key])
    return True

# ---------------- UI ----------------
st.title("🕸️ Temporal Claim Graph Viewer")

available_dates = sorted(df["claim_date"].dt.date.unique())
selected_date = st.selectbox("Select a date", available_dates)
selected_date = pd.to_datetime(selected_date)
day_end = selected_date.normalize() + pd.Timedelta(days=1)   # include full day

# Focus claim (optional)
st.markdown("**Focus claim (optional):** show only this claim, its entities, and claims linked to it.")
focus_claim = st.selectbox(
    "Claim ID (leave empty to show all in subgraph):",
    options=[""] + sorted(df["claim_id"].unique().tolist()),
    index=0
)

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

# Collect entities used to connect claims (for optional highlighting)
connecting_entities = set()
for u, v, d in G.edges(data=True):
    ts = d.get("timestamp")
    ts = pd.to_datetime(ts) if ts is not None else _dt(1900, 1, 1)
    if ts >= day_end:
        continue
    if d.get("edge_type") == "claim-claim":
        u_str, v_str = str(u), str(v)
        if u_str in valid_claims and v_str in valid_claims:
            via, val = d.get("via"), d.get("shared_value")
            if via and val:
                connecting_entities.add(f"{via}:{val}")

# Build directed subgraph for the selected day (entity→claim and claim→claim)
G_sub = nx.DiGraph()

for u, v, d in G.edges(data=True):
    ts = d.get("timestamp")
    ts = pd.to_datetime(ts) if ts is not None else _dt(1900, 1, 1)
    if ts >= day_end:
        continue

    et = d.get("edge_type")
    u_str, v_str = str(u), str(v)

    if et == "claim-claim":
        # both endpoints must be valid claims (older -> newer)
        if u_str in valid_claims and v_str in valid_claims:
            if u not in G_sub: G_sub.add_node(u, **G.nodes[u])
            if v not in G_sub: G_sub.add_node(v, **G.nodes[v])
            G_sub.add_edge(u, v, **d)

    elif et == "entity-claim" and show_all_nodes:
        # entity (u) -> claim (v); the claim must be valid by date
        if v_str in valid_claims:
            if (not show_only_shared_entities) or (u in connecting_entities):
                if u not in G_sub: G_sub.add_node(u, **G.nodes[u])  # entity
                if v not in G_sub: G_sub.add_node(v, **G.nodes[v])  # claim
                G_sub.add_edge(u, v, **d)  # entity → claim

# Ensure ALL valid claims are visible (even if isolated)
for cid in valid_claims:
    ensure_claim_node(G_sub, G, cid)

# --- Diagnostics on full subgraph ---
st.caption(
    f"claims≤{selected_date.date()}: {len(valid_claims)} | "
    f"graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges | "
    f"subgraph: {G_sub.number_of_nodes():,} nodes, {G_sub.number_of_edges():,} edges"
)

# ---------------- Simple focus filter (ONE claim, direct neighbors only) ----------------
G_draw = G_sub
if focus_claim:
    focus_key = claim_node_key_in(G_sub, focus_claim)
    if focus_key is None:
        if focus_claim in valid_claims and ensure_claim_node(G_sub, G, focus_claim):
            focus_key = claim_node_key_in(G_sub, focus_claim)
    if focus_key is None:
        st.warning("Selected claim is not present in the subgraph at this date.")
        G_draw = nx.DiGraph()
    else:
        und = G_sub.to_undirected()
        nodes_keep = list(nx.ego_graph(und, focus_key, radius=1).nodes())
        G_draw = G_sub.subgraph(nodes_keep).copy()

# ---------------- Choose what to DRAW (never changes the data) ----------------
draw_note = ""
if G_draw.number_of_nodes() > max_draw_nodes:
    und = G_draw.to_undirected()
    comps = list(nx.connected_components(und))
    if comps:
        biggest = max(comps, key=len)
        G_draw = G_draw.subgraph(biggest).copy()
        draw_note = f"Drawing largest component only (size={G_draw.number_of_nodes()}). "
if G_draw.number_of_nodes() > max_draw_nodes:
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
    st.warning("No nodes/edges for this date/filter. Try a later date or clear the focus.")
else:
    fig, ax = plt.subplots(figsize=(14, 10))
    try:
        pos = nx.spring_layout(G_draw, seed=42, k=None, iterations=50, dim=2)
    except Exception:
        pos = nx.kamada_kawai_layout(G_draw)

    # Color claims by joining via stringified node ids
    df_labels = dict(zip(df["claim_id"], df["is_fraud"].astype(int)))
    def node_is_fraud(n):
        return df_labels.get(str(n), 0) == 1

    fraud_nodes  = [n for n, d in G_draw.nodes(data=True) if d.get("node_type") == "claim" and node_is_fraud(n)]
    normal_nodes = [n for n, d in G_draw.nodes(data=True) if d.get("node_type") == "claim" and not node_is_fraud(n)]

    if normal_nodes:
        nx.draw_networkx_nodes(G_draw, pos, nodelist=normal_nodes, node_color="lightblue", node_size=420, label="Normal Claim", ax=ax)
    if fraud_nodes:
        nx.draw_networkx_nodes(G_draw, pos, nodelist=fraud_nodes, node_color="red", node_size=560, label="Fraud Claim", ax=ax)

    # Highlight focused claim on top
    if focus_claim:
        fk = claim_node_key_in(G_draw, focus_claim)
        if fk is not None:
            nx.draw_networkx_nodes(G_draw, pos, nodelist=[fk], node_color="yellow",
                                   edgecolors="black", linewidths=1.2, node_size=750, label="Focused Claim", ax=ax)

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
    entity_edges      = [(u, v) for u, v, d in G_draw.edges(data=True) if d.get("edge_type") == "entity-claim"]

    if entity_edges:
        nx.draw_networkx_edges(G_draw, pos, edgelist=entity_edges,
                               edge_color='gray', alpha=0.35,
                               arrows=True, arrowstyle='-|>', ax=ax)  # entity → claim
    if claim_claim_edges:
        nx.draw_networkx_edges(G_draw, pos, edgelist=claim_claim_edges,
                               edge_color='black', arrows=True, arrowstyle='->', ax=ax)

    # Edge labels for claim-claim only (optional)
    if show_entity_labels_on_edges and claim_claim_edges:
        edge_labels = {}
        for u, v, d in G_draw.edges(data=True):
            if d.get("edge_type") == "claim-claim" and d.get("timestamp"):
                ts = pd.to_datetime(d.get("timestamp")).date()
                edge_labels[(u, v)] = f"{d.get('via','')}\n{ts}"
        if edge_labels:
            nx.draw_networkx_edge_labels(G_draw, pos, edge_labels=edge_labels, font_size=7, ax=ax)

    nx.draw_networkx_labels(G_draw, pos, font_size=8, ax=ax)

    title_focus = f" | focus={focus_claim}" if focus_claim else ""
    ax.set_title(
        f"Graph up to {selected_date.date()} (drawn {G_draw.number_of_nodes()} nodes / {G_draw.number_of_edges()} edges){title_focus}",
        fontsize=14
    )
    ax.axis("off")
    ax.legend(markerscale=0.6, fontsize=8, loc="upper right")
    st.pyplot(fig)

# (Feature tables removed as requested.)
