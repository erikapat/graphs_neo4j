# 01_simulate_data.py
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

def simulate_claims_strong(n=4000, seed=42):
    rng = np.random.default_rng(seed)
    now = datetime(2024, 1, 1)

    plates = [f"PLT{10000+i}" for i in range(2500)]
    phones = [f"+34{600000000+i}" for i in range(2500)]
    emails = [f"user{i}@ex.com" for i in range(2500)]
    addrs  = [f"STREET_{i}" for i in range(1200)]
    shops  = [f"SHOP_{i}" for i in range(400)]
    banks  = [f"ES76{1000000000+i}" for i in range(1500)]
    locs   = [f"LOC_{i}" for i in range(500)]
    thirdp = [f"TP_{i}" for i in range(2500)]

    # 20% risky shops; risky appear ~6x more often; fraud: risky=0.85, safe=0.05
    risky_idx = set(rng.choice(len(shops), size=max(1, int(0.20 * len(shops))), replace=False))
    risky_shops = set(shops[i] for i in risky_idx)
    w_risky, w_safe = 0.6, 0.1
    weights = np.array([w_risky if s in risky_shops else w_safe for s in shops], dtype=float)
    weights /= weights.sum()

    rows = []
    for i in range(n):
        claim_id = str(10_000_000 + i)
        step_days = int(rng.exponential(scale=2.5))
        claim_date = now + timedelta(days=step_days)
        shop = shops[rng.choice(len(shops), p=weights)]
        p = 0.85 if shop in risky_shops else 0.05
        rows.append({
            "claim_id": claim_id,
            "claim_date": claim_date,
            "insurer_license_plate": plates[rng.integers(len(plates))],
            "insurer_phone_number":  phones[rng.integers(len(phones))],
            "insurer_email":         emails[rng.integers(len(emails))],
            "insurer_address":       addrs[rng.integers(len(addrs))],
            "repair_shop":           shop,
            "bank_account":          banks[rng.integers(len(banks))],
            "claim_location":        locs[rng.integers(len(locs))],
            "third_party_license_plate": thirdp[rng.integers(len(thirdp))],
            "is_fraud": int(rng.random() < p),
        })

    df = pd.DataFrame(rows).sort_values("claim_date").reset_index(drop=True)
    return df

def main():
    os.makedirs("../data", exist_ok=True)
    df = simulate_claims_strong(n=4000, seed=42)
    df.to_csv("data/sy_dataset_1.csv", index=False)
    print("Saved: data/sy_dataset_1.csv")

if __name__ == "__main__":
    main()
