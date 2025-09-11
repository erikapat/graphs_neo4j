import pandas as pd
import networkx as nx
import pickle  # <-- added

pd.set_option('display.max_rows', 999)
pd.set_option('display.max_columns', 999)
# Load data
df = pd.read_csv("data/sy_dataset_1.csv")
df["claim_date"] = pd.to_datetime(df["claim_date"])
df["claim_id"] = df["claim_id"].astype(str)

# Define entity columns
entity_columns = [
    "insurer_license_plate", "insurer_phone_number", "insurer_email",
    "insurer_address", "repair_shop", "bank_account",
    "claim_location", "third_party_license_plate"
]

# Build graph with timestamped edges

def build_temporal_graph_with_edge_attrs(df, entity_cols):
    df = df.sort_values("claim_date")
    G = nx.DiGraph()  # Directed graph

    seen_pairs = set()  # Avoid duplicated claim-to-claim edges (as undirected pairs)
    entity_index = {}  # Track which claims are connected to which entity values

    for idx, row in df.iterrows():
        cid = row["claim_id"]
        cdate = row["claim_date"]

        # Add claim node
        G.add_node(cid, node_type="claim", claim_date=cdate)

        # Add entity nodes + undirected edges (claim ↔ entity)
        for col in entity_cols:
            val = row[col]
            if pd.notna(val):
                entity_node = f"{col}:{val}"
                G.add_node(entity_node, node_type=col, label=val)

                # Add both directions to simulate undirected relation
                G.add_edge(cid, entity_node,
                           edge_type="claim-entity",
                           entity_type=col,
                           timestamp=cdate)
                G.add_edge(entity_node, cid,
                           edge_type="entity-claim",
                           entity_type=col,
                           timestamp=cdate)

                # Connect to previous claims that share this entity
                prev_claims = entity_index.get((col, val), [])
                for other_cid in prev_claims:
                    if (other_cid, cid) not in seen_pairs:
                        G.add_edge(other_cid, cid,
                                   edge_type="claim-claim",
                                   via=col,
                                   shared_value=val,
                                   timestamp=cdate)
                        seen_pairs.add((other_cid, cid))

                entity_index.setdefault((col, val), []).append(cid)

    return G


# Usage
G_temporal = build_temporal_graph_with_edge_attrs(df, entity_columns)

from collections import Counter

attrs_seen = Counter()
for u, v, d in G_temporal.edges(data=True):
    if d.get("edge_type") == "claim-claim":
        key = (frozenset([u, v]), d.get("via"), d.get("shared_value"))
        attrs_seen[key] += 1

duplicated = {k: v for k, v in attrs_seen.items() if v > 1}
print(f"Duplicated claim-claim edges with same attrs: {len(duplicated)}")


# Save graph (using pickle)
with open("data/temporal_graph_with_edge_attrs.gpickle", "wb") as f:
    pickle.dump(G_temporal, f, protocol=pickle.HIGHEST_PROTOCOL)

# Preview some edge attributes
edge_samples = list(G_temporal.edges(data=True))[:10]
print(edge_samples)

# -----------------------------------------------------------------------------------------------------------

import pandas as pd
import networkx as nx
import json
import pickle  # ensure available in this section too

def compute_graph_features_from_saved_graph_v1(graph_path: str, df, entity_cols):
    with open(graph_path, "rb") as f:
        G_full = pickle.load(f)
    features = []

    # Orden estable para claims del mismo día
    df = df.copy()
    df["claim_id"] = df["claim_id"].astype(str)
    df = df.sort_values(["claim_date", "claim_id"])

    for idx, row in df.iterrows():
        cid = row["claim_id"]
        cdate = row["claim_date"]

        # Nodos válidos hasta esta fecha
        valid_nodes = df[df["claim_date"] <= cdate]["claim_id"].tolist()
        G_sub = G_full.subgraph(valid_nodes).copy()

        # Eliminar aristas futuras
        G_sub.remove_edges_from([
            (u, v) for u, v, d in G_sub.edges(data=True)
            if d.get("timestamp") and d["timestamp"] > cdate
        ])

        if not G_sub.has_node(cid):
            continue

        # Componente conectado (no dirigido)
        undirected = G_sub.to_undirected()
        try:
            component = nx.node_connected_component(undirected, cid)
            component_subgraph = undirected.subgraph(component)
        except:
            component = {cid}
            component_subgraph = undirected.subgraph(component)

        # Métricas centrales
        try:
            deg = nx.degree_centrality(component_subgraph)
            clo = nx.closeness_centrality(component_subgraph)
            bet = nx.betweenness_centrality(component_subgraph)
        except:
            deg = clo = bet = {}

        # Aristas incidentes al nodo actual (simétrico)
        node_connections = []
        edge_uniques = {f"{col}_used_to_connect_claims": 0 for col in entity_cols}
        col_values = {col: set() for col in entity_cols}

        for u, v, d in G_sub.edges(data=True):
            if cid in (u, v):
                other = v if u == cid else u
                via = d.get("via")
                val = d.get("shared_value")

                if via and val:
                    edge_uniques[f"{via}_used_to_connect_claims"] += 1
                    col_values[via].add(val)

                node_connections.append({
                    "connected_to": other,
                    "via": via,
                    "value": val,
                    "timestamp": str(d.get("timestamp"))
                })

        # Valores únicos en el componente
        component_claims = df[df["claim_id"].isin(component)]
        row_feat = {
            "claim_id": cid,
            "claim_date": cdate,
            "component_size": len(component),
            "connections_info": json.dumps(node_connections),
            "degree_centrality": deg.get(cid, 0),
            "closeness_centrality": clo.get(cid, 0),
            "betweenness_centrality": bet.get(cid, 0)
        }

        for col in entity_cols:
            row_feat[f"unique_{col}s_in_component"] = component_claims[col].nunique()

        row_feat.update(edge_uniques)
        features.append(row_feat)

    return pd.DataFrame(features)


def compute_graph_features_from_saved_graph_v2(graph_path: str, df, entity_cols):
    with open(graph_path, "rb") as f:
        G_full = pickle.load(f)
    features = []

    # Orden estable para claims del mismo día
    df = df.copy()
    df["claim_id"] = df["claim_id"].astype(str)
    df = df.sort_values(["claim_date", "claim_id"])

    for idx, row in df.iterrows():
        cid = row["claim_id"]
        cdate = row["claim_date"]

        # Nodos válidos hasta esta fecha
        valid_nodes = df[df["claim_date"] <= cdate]["claim_id"].tolist()
        G_sub = G_full.subgraph(valid_nodes).copy()

        # Eliminar aristas futuras
        G_sub.remove_edges_from([
            (u, v) for u, v, d in G_sub.edges(data=True)
            if d.get("timestamp") and d["timestamp"] > cdate
        ])

        if not G_sub.has_node(cid):
            continue

        # Componente conectado (no dirigido)
        undirected = G_sub.to_undirected()
        try:
            component = nx.node_connected_component(undirected, cid)
            component_subgraph = undirected.subgraph(component)
        except:
            component = {cid}
            component_subgraph = undirected.subgraph(component)

        # Métricas centrales
        try:
            deg = nx.degree_centrality(component_subgraph)
            clo = nx.closeness_centrality(component_subgraph)
            bet = nx.betweenness_centrality(component_subgraph)
        except:
            deg = clo = bet = {}

        # Aristas incidentes al nodo actual (simétrico)
        node_connections = []
        edge_uniques = {f"{col}_used_to_connect_claims": 0 for col in entity_cols}
        col_values = {col: set() for col in entity_cols}

        for u, v, d in G_sub.edges(data=True):
            if cid in (u, v):
                other = v if u == cid else u
                via = d.get("via")
                val = d.get("shared_value")

                if via and val:
                    edge_uniques[f"{via}_used_to_connect_claims"] += 1
                    col_values[via].add(val)

                node_connections.append({
                    "connected_to": other,
                    "via": via,
                    "value": val,
                    "timestamp": str(d.get("timestamp"))
                })

        # Valores únicos en el componente
        component_claims = df[df["claim_id"].isin(component)]
        row_feat = {
            "claim_id": cid,
            "claim_date": cdate,
            "component_size": len(component),
            "connections_info": json.dumps(node_connections),
            "degree_centrality": deg.get(cid, 0),
            "closeness_centrality": clo.get(cid, 0),
            "betweenness_centrality": bet.get(cid, 0)
        }

        for col in entity_cols:
            row_feat[f"unique_{col}s_in_component"] = component_claims[col].nunique()

        row_feat.update(edge_uniques)
        features.append(row_feat)

    return pd.DataFrame(features)


def compute_graph_features_from_saved_graph(graph_path: str, df, entity_cols):
    import json
    import math
    from collections import Counter
    with open(graph_path, "rb") as f:
        G_full = pickle.load(f)
    features = []

    df = df.copy()
    df["claim_id"] = df["claim_id"].astype(str)
    df = df.sort_values(["claim_date", "claim_id"])

    for idx, row in df.iterrows():
        cid = row["claim_id"]
        cdate = row["claim_date"]

        if not G_full.has_node(cid):
            continue

        # Nodos válidos hasta esta fecha
        valid_nodes = [n for n, d in G_full.nodes(data=True)
                       if (d.get("node_type") == "claim" and G_full.nodes[n].get("claim_date") <= cdate)
                       or d.get("node_type") == "entity"]

        G_sub = G_full.subgraph(valid_nodes).copy()

        # Eliminar aristas futuras
        G_sub.remove_edges_from([
            (u, v) for u, v, d in G_sub.edges(data=True)
            if d.get("timestamp") and d["timestamp"] > cdate
        ])

        if not G_sub.has_node(cid):
            continue

        # Componente conectado (no dirigido)
        undirected = G_sub.to_undirected()
        try:
            component = nx.node_connected_component(undirected, cid)
            component_subgraph = undirected.subgraph(component)
        except:
            component = {cid}
            component_subgraph = undirected.subgraph(component)

        # Métricas centrales
        try:
            deg = nx.degree_centrality(component_subgraph)
            clo = nx.closeness_centrality(component_subgraph)
            bet = nx.betweenness_centrality(component_subgraph)
        except:
            deg = clo = bet = {}

        # Nuevas métricas del componente
        claim_nodes = [n for n in component_subgraph.nodes if G_full.nodes[n].get("node_type") == "claim"]
        shared_entities = set()
        degrees = []
        entity_type_counter = Counter()

        for u, v, d in component_subgraph.edges(data=True):
            if d.get("edge_type") == "claim-claim":
                via = d.get("via")
                val = d.get("shared_value")
                if via and val:
                    shared_entities.add((via, val))
            if d.get("edge_type") == "claim-entity":
                entity_type = d.get("entity_type")
                if entity_type:
                    entity_type_counter[entity_type] += 1

        for n in component_subgraph.nodes():
            degrees.append(component_subgraph.degree(n))

        avg_degree = sum(degrees) / len(degrees) if degrees else 0
        total = sum(entity_type_counter.values())
        entropy = -sum((count / total) * math.log(count / total) for count in entity_type_counter.values()) if total > 0 else 0

        # Aristas incidentes al nodo actual (simétrico)
        node_connections = []
        edge_uniques = {f"{col}_used_to_connect_claims": 0 for col in entity_cols}
        col_values = {col: set() for col in entity_cols}

        for u, v, d in G_sub.edges(data=True):
            if cid in (u, v):
                other = v if u == cid else u
                via = d.get("via")
                val = d.get("shared_value")

                if via and val:
                    edge_uniques[f"{via}_used_to_connect_claims"] += 1
                    col_values[via].add(val)

                node_connections.append({
                    "connected_to": other,
                    "via": via,
                    "value": val,
                    "timestamp": str(d.get("timestamp"))
                })

        component_claims = df[df["claim_id"].isin(claim_nodes)]
        row_feat = {
            "claim_id": cid,
            "claim_date": cdate,
            "component_size": len(component),
            "connections_info": json.dumps(node_connections),
            "degree_centrality": deg.get(cid, 0),
            "closeness_centrality": clo.get(cid, 0),
            "betweenness_centrality": bet.get(cid, 0),
            "num_shared_entities_with_other_claims": len(shared_entities),
            "avg_component_degree": avg_degree,
            "entropy_of_entity_types": entropy
        }

        for col in entity_cols:
            row_feat[f"unique_{col}s_in_component"] = component_claims[col].nunique()

        row_feat.update(edge_uniques)
        features.append(row_feat)

    return pd.DataFrame(features)

features_df = compute_graph_features_from_saved_graph("data/temporal_graph_with_edge_attrs.gpickle", df, entity_columns)
features_df.to_csv("data/graph_features_from_edge_attrs.csv", index=False)

print(features_df.head(20))

