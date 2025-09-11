# 02_build_graph.py  (leak-safe: past -> future only)
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
    # ensure chronological processing
    df = df.sort_values("claim_date")
    G = nx.DiGraph()
    seen_pairs = set()          # to avoid duplicate claim->claim edges
    entity_index = {}           # (col, value) -> list of prior claim_ids using this entity

    for _, row in df.iterrows():
        cid = row["claim_id"]
        cdate = row["claim_date"]

        # claim node
        G.add_node(cid, node_type="claim", claim_date=cdate)

        # entities for this claim
        for col in entity_cols:
            val = row[col]
            if pd.notna(val):
                ent = f"{col}:{val}"

                # add / update entity node (track first_seen)
                if ent not in G:
                    G.add_node(ent, node_type=col, label=val, first_seen=cdate)
                else:
                    # keep the earliest time the entity appeared
                    fs = G.nodes[ent].get("first_seen", cdate)
                    if fs is None or cdate < fs:
                        G.nodes[ent]["first_seen"] = cdate

                # SAFE DIRECTION: entity -> claim (no claim -> entity)
                G.add_edge(
                    ent, cid,
                    edge_type="entity-claim",
                    entity_type=col,
                    timestamp=cdate
                )

                # connect prior claims that shared this entity (older -> newer)
                prev = entity_index.get((col, val), [])
                for other_cid in prev:
                    if (other_cid, cid) not in seen_pairs:
                        G.add_edge(
                            other_cid, cid,
                            edge_type="claim-claim",
                            via=col,
                            shared_value=val,
                            timestamp=cdate  # time of the *new* link discovery
                        )
                        seen_pairs.add((other_cid, cid))

                # register this claim as having used the entity
                entity_index.setdefault((col, val), []).append(cid)

    return G

def main():
    os.makedirs("data", exist_ok=True)
    df = pd.read_csv("data/sy_dataset_1.csv", parse_dates=["claim_date"])
    G = build_temporal_graph_with_edge_attrs(df, ENTITY_COLUMNS)

    with open("data/temporal_graph_with_edge_attrs.gpickle", "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Saved: data/temporal_graph_with_edge_attrs.gpickle")
    print(f"nodes={G.number_of_nodes():,} edges={G.number_of_edges():,}")

if __name__ == "__main__":
    main()
