import os
import pickle
import pandas as pd
import networkx as nx
import json
import math
from collections import Counter

# --------------------------------------------
# CONFIGURACIÓN DE RUTAS Y PARÁMETROS GLOBALES
# --------------------------------------------

GRAPH_PATH = "data/incremental_graph.gpickle"
INDEX_PATH = "data/entity_index.pkl"
SEEN_PAIRS_PATH = "data/seen_pairs.pkl"
FEATURES_PATH = "data/graph_features_incremental.csv"

ENTITY_COLUMNS = [
    "insurer_license_plate", "insurer_phone_number", "insurer_email",
    "insurer_address", "repair_shop", "bank_account",
    "claim_location", "third_party_license_plate"
]

# --------------------------------------------
# FUNCIONES AUXILIARES
# --------------------------------------------

def load_incremental_graph_and_context(graph_path, index_path, seen_pairs_path):
    if os.path.exists(graph_path):
        G = nx.read_gpickle(graph_path)
        with open(index_path, "rb") as f:
            entity_index = pickle.load(f)
        with open(seen_pairs_path, "rb") as f:
            seen_pairs = pickle.load(f)
    else:
        G = nx.DiGraph()
        entity_index = {}
        seen_pairs = set()
    return G, entity_index, seen_pairs

def save_incremental_graph_and_context(G, graph_path, entity_index, index_path, seen_pairs, seen_pairs_path):
    nx.write_gpickle(G, graph_path)
    with open(index_path, "wb") as f:
        pickle.dump(entity_index, f)
    with open(seen_pairs_path, "wb") as f:
        pickle.dump(seen_pairs, f)

# --------------------------------------------
# CONSTRUCCIÓN INCREMENTAL DEL GRAFO
# --------------------------------------------

def update_temporal_graph_with_new_day(G, df_day, entity_cols, entity_index, seen_pairs):
    for idx, row in df_day.iterrows():
        cid = row["claim_id"]
        cdate = row["claim_date"]

        G.add_node(cid, node_type="claim", claim_date=cdate)

        for col in entity_cols:
            val = row[col]
            if pd.notna(val):
                entity_node = f"{col}:{val}"
                G.add_node(entity_node, node_type=col, label=val)
                G.add_edge(cid, entity_node,
                           edge_type="claim-entity",
                           entity_type=col,
                           timestamp=cdate)
                G.add_edge(entity_node, cid,
                           edge_type="entity-claim",
                           entity_type=col,
                           timestamp=cdate)

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

# --------------------------------------------
# CÁLCULO DE FEATURES INCREMENTALES
# --------------------------------------------

def compute_features_for_day(G_full, df_day, entity_cols):
    features = []
    for idx, row in df_day.iterrows():
        cid = row["claim_id"]
        cdate = row["claim_date"]

        if not G_full.has_node(cid):
            continue

        valid_nodes = [n for n, d in G_full.nodes(data=True)
                       if (d.get("node_type") == "claim" and G_full.nodes[n].get("claim_date") <= cdate)
                       or d.get("node_type") == "entity"]
        G_sub = G_full.subgraph(valid_nodes).copy()
        G_sub.remove_edges_from([
            (u, v) for u, v, d in G_sub.edges(data=True)
            if d.get("timestamp") and d["timestamp"] > cdate
        ])

        if not G_sub.has_node(cid):
            continue

        undirected = G_sub.to_undirected()
        try:
            component = nx.node_connected_component(undirected, cid)
            component_subgraph = undirected.subgraph(component)
        except:
            component = {cid}
            component_subgraph = undirected.subgraph(component)

        try:
            deg = nx.degree_centrality(component_subgraph)
            clo = nx.closeness_centrality(component_subgraph)
            bet = nx.betweenness_centrality(component_subgraph)
        except:
            deg = clo = bet = {}

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
            row_feat[f"unique_{col}s_in_component"] = df_day[col].nunique()

        row_feat.update(edge_uniques)
        features.append(row_feat)

    return pd.DataFrame(features)

