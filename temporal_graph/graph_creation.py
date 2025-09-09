# 02_build_graph.py
import os
import pickle
import pandas as pd
import networkx as nx

ENTITY_COLUMNS = [
    "insurer_license_plate", "insurer_phone_number", "insurer_email",
    "insurer_address", "repair_shop", "bank_account",
    "claim_location", "third_party_license_plate"
]

def build_temporal_graph_with_edge_attrs(df, entity_cols):
    df = df.sort_values("claim_date")
    G = nx.DiGraph()
    seen_pairs = set()
    entity_index = {}

    for _, row in df.iterrows():
        cid = row["claim_id"]
        cdate = row["claim_date"]
        G.add_node(cid, node_type="claim", claim_date=cdate)

        for col in entity_cols:
            val = row[col]
            if pd.notna(val):
                ent = f"{col}:{val}"
                G.add_node(ent, node_type=col, label=val)

                # bidirectional edges with timestamp
                G.add_edge(cid, ent, edge_type="claim-entity", entity_type=col, timestamp=cdate)
                G.add_edge(ent, cid, edge_type="entity-claim", entity_type=col, timestamp=cdate)

                # connect prior claims sharing this entity (claim-claim edges)
                prev = entity_index.get((col, val), [])
                for other_cid in prev:
                    if (other_cid, cid) not in seen_pairs:
                        G.add_edge(other_cid, cid, edge_type="claim-claim", via=col,
                                   shared_value=val, timestamp=cdate)
                        seen_pairs.add((other_cid, cid))
                entity_index.setdefault((col, val), []).append(cid)
    return G

def main():
    os.makedirs("data", exist_ok=True)
    df = pd.read_csv("data/sy_dataset_1.csv", parse_dates=["claim_date"])
    G = build_temporal_graph_with_edge_attrs(df, ENTITY_COLUMNS)

    with open("data/temporal_graph_with_edge_attrs.gpickle", "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Saved: data/temporal_graph_with_edge_attrs.gpickle")

if __name__ == "__main__":
    main()
