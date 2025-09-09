import pandas as pd
import networkx as nx
import time
import json

pd.set_option('display.max_rows', 999)
pd.set_option('display.max_columns', 999)
# Load data
df = pd.read_csv("data/sy_dataset_1.csv")
df["claim_date"] = pd.to_datetime(df["claim_date"])

# Define entity columns
entity_columns = [
    "insurer_license_plate", "insurer_phone_number", "insurer_email",
    "insurer_address", "repair_shop", "bank_account",
    "claim_location", "third_party_license_plate"
]

# Build temporal graph and extract features + connection metadata
def build_temporal_graph_and_features(df, entity_cols):
    print("[INFO]: Starting temporal graph feature generation...")
    features = []
    G = nx.Graph()
    connections_list = []

    for idx, row in df.iterrows():
        t0 = time.time()
        current_claim_id = row["claim_id"]
        current_date = row["claim_date"]

        past_claims = df[df["claim_date"] <= current_date]

        for _, claim in past_claims.iterrows():
            G.add_node(claim["claim_id"])

        claims_list = past_claims.to_dict("records")
        for i in range(len(claims_list)):
            for j in range(i + 1, len(claims_list)):
                ci = claims_list[i]
                cj = claims_list[j]
                for col in entity_cols:
                    if pd.notna(ci[col]) and pd.notna(cj[col]) and ci[col] == cj[col]:
                        G.add_edge(ci["claim_id"], cj["claim_id"], via=col, value=ci[col])
                        connections_list.append({
                            "from": ci["claim_id"],
                            "to": cj["claim_id"],
                            "via": col,
                            "value": ci[col]
                        })
                        break

        component = set()
        for comp in nx.connected_components(G):
            if current_claim_id in comp:
                component = comp
                break

        component_claims = df[df["claim_id"].isin(component)]

        # Get connection-based uniqueness
        component_subgraph = G.subgraph(component)
        edge_data = list(component_subgraph.edges(data=True))

        edge_based_uniques = {
            f"{col}_used_to_connect_claims": len(set(
                d["value"] for u, v, d in edge_data if d["via"] == col
            ))
            for col in entity_cols
        }

        node_connections = []
        for neighbor in G.neighbors(current_claim_id):
            edge = G.get_edge_data(current_claim_id, neighbor)
            node_connections.append({
                "connected_to": neighbor,
                "via": edge.get("via"),
                "value": edge.get("value")
            })

            # Compute centralities for subgraph
        try:
            deg_cent = nx.degree_centrality(component_subgraph)
            clo_cent = nx.closeness_centrality(component_subgraph)
            bet_cent = nx.betweenness_centrality(component_subgraph)
        except:
            deg_cent = clo_cent = bet_cent = {}

        feature_row = {
            "claim_id": current_claim_id,
            "claim_date": current_date,
            "component_size": len(component),
            "connections_info": json.dumps(node_connections),
            "degree_centrality": deg_cent.get(current_claim_id, 0),
            "closeness_centrality": clo_cent.get(current_claim_id, 0),
            "betweenness_centrality": bet_cent.get(current_claim_id, 0)
        }

        # Add both definitions
        for col in entity_cols:
            feature_row[f"unique_{col}s_in_component"] = component_claims[col].nunique()

        feature_row.update(edge_based_uniques)

        features.append(feature_row)

        print(f"[{idx+1}/{len(df)}] claim_id={current_claim_id} | Component size={len(component)} | Time={time.time()-t0:.2f}s")

    return pd.DataFrame(features), G, pd.DataFrame(connections_list)

# Run updated pipeline
df_features, G, df_connections = build_temporal_graph_and_features(df, entity_columns)

# Save and display results
df_full = df.merge(df_features, on=["claim_id", "claim_date"], how="left")
df_full.to_csv("data/claims_with_both_definitions.csv", index=False)
df_connections.to_csv("data/claim_pairwise_links.csv", index=False)
nx.write_gpickle(G, "data/temporal_claim_graph_with_edges.gpickle")


print(df_full)


# 02_train_model_and_predict.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load preprocessed data
df = pd.read_csv("data/claims_with_graph_features.csv")

feature_cols = [
    "component_size", "unique_license_plates", "shared_phone_numbers",
    "repair_shops_used", "banks_involved", "third_parties_involved",
    "locations_involved"
]

X = df[feature_cols]
y = df["fraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

df["fraud_prediction"] = clf.predict(X)
df["fraud_probability"] = clf.predict_proba(X)[:, 1]

print("\n[MODEL REPORT]:\n", classification_report(y_test, clf.predict(X_test)))
df.to_csv("data/final_claims_with_predictions.csv", index=False)
print("[INFO]: Saved model predictions to data/final_claims_with_predictions.csv")

#---------------------


import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

def draw_graph_for_date(df, entity_cols, date):
    df_day = df[df["claim_date"] == pd.to_datetime(date)]
    G = nx.Graph()

    # Define custom colors for each entity type
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

    # Add claim nodes
    for _, row in df_day.iterrows():
        G.add_node(row["claim_id"], label=row["claim_id"], type="claim", fraud=row["fraud"])

    # Add entity nodes and edges
    for _, row in df_day.iterrows():
        claim_id = row["claim_id"]
        for col in entity_cols:
            entity_val = row[col]
            if pd.notna(entity_val):
                entity_node = f"{col}:{entity_val}"
                G.add_node(entity_node, label=col, type=col, color=entity_colors.get(col, "gray"))
                G.add_edge(claim_id, entity_node)

    # Layout
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(14, 10))

    # Draw claim nodes
    fraud_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "claim" and d.get("fraud") == 1]
    normal_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "claim" and d.get("fraud") == 0]

    nx.draw_networkx_nodes(G, pos, nodelist=fraud_nodes, node_color="red", node_size=500, label="Fraud Claims")
    nx.draw_networkx_nodes(G, pos, nodelist=normal_nodes, node_color="lightblue", node_size=300, label="Normal Claims")

    # Draw entity nodes by type
    for col, color in entity_colors.items():
        entity_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == col]
        nx.draw_networkx_nodes(G, pos, nodelist=entity_nodes, node_color=color, node_size=200, label=col)

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.3)

    # Label only claims
    claim_labels = {n: str(n) for n in fraud_nodes + normal_nodes}
    nx.draw_networkx_labels(G, pos, labels=claim_labels, font_size=8)

    plt.title(f"Graph for {date} (Claims and Colored Entities)", fontsize=14)
    plt.axis("off")
    plt.legend(markerscale=0.6)
    plt.show()


draw_graph_for_date(df_full, entity_columns, "2024-07-01")
