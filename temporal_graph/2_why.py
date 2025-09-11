# call this as:
# python why.py --claim-id 20000695 --cutoff 2024-01-08
import argparse
import pickle
from collections import defaultdict
from typing import Dict, Any, List, Optional

import pandas as pd
import networkx as nx


ENTITY_COLUMNS = [
    "insurer_license_plate", "insurer_phone_number", "insurer_email",
    "insurer_address", "repair_shop", "bank_account",
    "claim_location", "third_party_license_plate",
]


def explain_claim(
        claim_id: str,
        df: pd.DataFrame,
        G: nx.DiGraph,
        entity_cols: List[str],
        day_cutoff: Optional[pd.Timestamp] = None,
        recent_days: int = 30,
) -> Dict[str, Any]:
    """
    Explanation for why a claim looks risky, based on the graph + labels.

    Parameters
    ----------
    claim_id : str
        Claim ID to explain.
    df : DataFrame
        Must contain: 'claim_id', 'claim_date', 'is_fraud', and the entity columns.
    G : nx.DiGraph
        Temporal graph. Edges have: edge_type ('claim-claim' / 'claim-entity'),
        via, shared_value, timestamp (datetime).
    entity_cols : list of str
        Entity columns from your schema.
    day_cutoff : pd.Timestamp or None
        Consider only edges/evidence with timestamp <= cutoff. If None, uses claim date.
    recent_days : int
        Window (days) for "recent burst" counts.

    Returns
    -------
    dict with metrics + bullet-point reasons.
    """
    claim_id = str(claim_id)
    row = df.loc[df["claim_id"] == claim_id]
    if row.empty:
        return {"error": f"claim_id {claim_id} not found in df"}

    claim_dt = row["claim_date"].iloc[0]
    label = int(row["is_fraud"].iloc[0])

    # Entities on the row
    entity_vals: Dict[str, Any] = {}
    for col in entity_cols:
        if col in row.columns:
            entity_vals[col] = row[col].iloc[0]

    # Helper: respect cutoff
    def _ts_ok(ts):
        if day_cutoff is None:
            return True
        try:
            return pd.to_datetime(ts) <= day_cutoff
        except Exception:
            return True

    # Collect neighborhood evidence (<= cutoff)
    used_entities = defaultdict(set)  # col -> {entity nodes}
    cc_neighbors = set()
    cc_edges = []

    for u, v, d in G.edges(data=True):
        if not _ts_ok(d.get("timestamp")):
            continue

        et = d.get("edge_type")
        if et == "claim-entity" and str(u) == claim_id:
            col = d.get("entity_type")
            # destination node encodes "col:value", good to keep
            if col is not None:
                used_entities[col].add(v)

        if et == "claim-claim" and (str(u) == claim_id or str(v) == claim_id):
            other = str(v) if str(u) == claim_id else str(u)
            cc_neighbors.add(other)
            cc_edges.append((u, v, d))

    # Per-entity stats (reuse, rate, recent burst)
    base_rate = float(df["is_fraud"].mean())
    entity_explanations: List[Dict[str, Any]] = []

    now_cut = day_cutoff if day_cutoff is not None else claim_dt
    recent_start = now_cut - pd.Timedelta(days=recent_days)

    for col, val in entity_vals.items():
        if pd.isna(val):
            continue
        # All claims sharing this entity value up to cutoff
        same = df[(df[col] == val) & (df["claim_date"] <= now_cut)]
        n_same = int(len(same))
        if n_same <= 1:
            continue

        rate = float(same["is_fraud"].mean())
        recent_n = int(same[(same["claim_date"] >= recent_start)]["claim_id"].nunique())
        lift = round(rate / base_rate, 2) if base_rate > 0 else None

        entity_explanations.append(
            {
                "entity_type": col,
                "value": val,
                "n_claims_sharing": n_same,
                "fraud_rate_sharing": round(rate, 4),
                f"recent_{recent_days}d_count": recent_n,
                "lift_vs_base": lift,
            }
        )

    # Component-level stats
    try:
        und = G.to_undirected()
        comp = nx.node_connected_component(und, claim_id)
        comp_claims = [n for n in comp if G.nodes[n].get("node_type") == "claim"]
        comp_size = len(comp_claims)
        comp_df = df[(df["claim_id"].isin(comp_claims)) & (df["claim_date"] <= now_cut)]
        comp_rate = float(comp_df["is_fraud"].mean()) if not comp_df.empty else 0.0
    except Exception:
        comp_size, comp_rate = 1, 0.0

    # Reasons (bullets)
    bullets: List[str] = []
    rs = row["repair_shop"].iloc[0] if "repair_shop" in df.columns else None
    if rs is not None:
        bullets.append(f"Repair shop = **{rs}**.")

    bullets.append(
        f"Shares **{len(cc_neighbors)}** other claims via entities (claim–claim links)."
    )
    bullets.append(
        f"Connected component (claims only) size = **{comp_size}**, "
        f"component fraud rate (≤ cutoff) = **{comp_rate:.2f}** vs base **{base_rate:.2f}**."
    )

    # Sort entity reasons by (lift, volume)
    entity_explanations.sort(
        key=lambda x: ((x.get("lift_vs_base") or 0.0), x["n_claims_sharing"]),
        reverse=True,
    )
    for e in entity_explanations[:6]:
        bullets.append(
            f"- **{e['entity_type']}** = `{e['value']}` → shared by **{e['n_claims_sharing']}** claims (≤ cutoff), "
            f"fraud rate **{e['fraud_rate_sharing']:.2f}** (lift **{e['lift_vs_base']}x**), "
            f"**{e[f'recent_{recent_days}d_count']}** in last {recent_days}d."
        )

    return {
        "claim_id": claim_id,
        "label": label,
        "claim_date": str(pd.to_datetime(claim_dt).date()),
        "base_rate": round(base_rate, 3),
        "component_size": comp_size,
        "component_rate": round(comp_rate, 3),
        "neighbors_via_entities": len(cc_neighbors),
        "top_entity_signals": entity_explanations,
        "reasons": bullets,
    }


# ---------------------------- CLI helpers ----------------------------

def _load_inputs(
        csv_path: str = "data/sy_dataset_1.csv",
        gpickle_path: str = "data/temporal_graph_with_edge_attrs.gpickle",
) -> (pd.DataFrame, nx.DiGraph):
    """Load dataframe and graph from disk."""
    df = pd.read_csv(csv_path, parse_dates=["claim_date"])
    df["claim_id"] = df["claim_id"].astype(str)
    with open(gpickle_path, "rb") as f:
        G = pickle.load(f)
    return df, G


def _print_explanation(exp: Dict[str, Any]) -> None:
    if "error" in exp:
        print(exp["error"])
        return
    print(f"\n=== Explanation for claim_id={exp['claim_id']} ===")
    print(f"Date: {exp['claim_date']} | Label: {exp['label']} (1=fraud)")
    print(
        f"Base rate: {exp['base_rate']:.3f} | "
        f"Component size: {exp['component_size']} | "
        f"Component rate: {exp['component_rate']:.3f} | "
        f"Neighbors via entities: {exp['neighbors_via_entities']}"
    )
    print("\nWhy it looks risky:")
    for b in exp["reasons"]:
        print(" • " + b)
    print()


def main():
    parser = argparse.ArgumentParser(description="Explain a claim using the temporal graph.")
    parser.add_argument("--claim-id", required=True, help="Claim ID to explain (string).")
    parser.add_argument("--cutoff", default=None, help="Cutoff date (YYYY-MM-DD). If omitted, uses claim date.")
    parser.add_argument("--recent-days", type=int, default=30, help="Window for recent burst counts.")
    parser.add_argument("--csv", default="data/sy_dataset_1.csv", help="Path to CSV with labels/entities.")
    parser.add_argument("--gpickle", default="data/temporal_graph_with_edge_attrs.gpickle",
                        help="Path to pickled NetworkX graph.")
    args = parser.parse_args()

    df, G = _load_inputs(args.csv, args.gpickle)

    cutoff_ts = pd.to_datetime(args.cutoff) if args.cutoff else None

    exp = explain_claim(
        claim_id=str(args.claim_id),
        df=df,
        G=G,
        entity_cols=ENTITY_COLUMNS,
        day_cutoff=cutoff_ts,
        recent_days=args.recent_days,
    )
    _print_explanation(exp)


if __name__ == "__main__":
    main()


