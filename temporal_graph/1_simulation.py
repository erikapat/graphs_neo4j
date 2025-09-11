# 01_simulation.py
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


def simulate_multi_entity(
        n_days=60,
        target_base_rate=0.12,  # desired average fraud rate (10–15%)
        seed=42,
        # steadier daily counts (slight front-load)
        daily_lambda_start=90,
        daily_lambda_end=70,
        # risky entity prevalence
        risky_shop_frac=0.10,  # 10% shops are risky
        risky_phone_frac=0.06,  # 6% phones are risky
        risky_plate_ring_count=30,  # number of collusive plate rings
        ring_size=12,  # plates per ring
        # sampling weights (risky entities appear more often)
        w_risky_shop=5.0,
        w_risky_phone=3.0,
        # logit contributions (how much each risky factor increases odds)
        contrib_shop_risky=2.0,
        contrib_phone_risky=1.2,
        contrib_plate_ring=1.0,
        # noise controls separation between classes (higher = harder; lower = easier)
        noise_sigma=0.40
):
    """
    Simulates claims with multi-entity fraud drivers:
      - risky repair shops (more frequent + higher fraud)
      - risky phone numbers (shared across many claims)
      - collusive license-plate rings (plates reused within small rings)
    Timeline: Poisson daily counts with mild decay (steadier than exponential).
    Fraud prob = sigmoid(intercept + sum(entity contributions) + Gaussian noise).
    The intercept is shifted to exactly match target_base_rate without destroying separation.
    """
    rng = np.random.default_rng(seed)
    now = datetime(2024, 1, 1)

    # --- entity pools ---
    n_plates = 8000
    n_phones = 4000
    n_emails = 5000
    n_addrs = 3000
    n_shops = 1000
    n_banks = 4000
    n_locs = 1200
    n_thirdp = 8000

    plates = np.array([f"PLT{100000 + i}" for i in range(n_plates)])
    phones = np.array([f"+34{600000000 + i}" for i in range(n_phones)])
    emails = np.array([f"user{i}@ex.com" for i in range(n_emails)])
    addrs = np.array([f"STREET_{i}" for i in range(n_addrs)])
    shops = np.array([f"SHOP_{i}" for i in range(n_shops)])
    banks = np.array([f"ES76{1000000000 + i}" for i in range(n_banks)])
    locs = np.array([f"LOC_{i}" for i in range(n_locs)])
    thirdp = np.array([f"TP_{i}" for i in range(n_thirdp)])

    # --- risky sets & rings ---
    risky_shop_mask = np.zeros(n_shops, dtype=bool)
    risky_shop_mask[rng.choice(n_shops, size=int(risky_shop_frac * n_shops), replace=False)] = True
    risky_phone_mask = np.zeros(n_phones, dtype=bool)
    risky_phone_mask[rng.choice(n_phones, size=int(risky_phone_frac * n_phones), replace=False)] = True

    # Collusive plate rings (small groups reused together)
    rings = []
    pool_idx = rng.permutation(n_plates)
    ptr = 0
    for _ in range(risky_plate_ring_count):
        ring_idx = pool_idx[ptr:ptr + ring_size]
        if len(ring_idx) < ring_size: break
        rings.append(plates[ring_idx])
        ptr += ring_size

    # Sampling weights -> risky entities appear more often
    shop_weights = np.where(risky_shop_mask, w_risky_shop, 1.0).astype(float)
    shop_weights /= shop_weights.sum()
    phone_weights = np.where(risky_phone_mask, w_risky_phone, 1.0).astype(float)
    phone_weights /= phone_weights.sum()

    # Daily volumes (Poisson with linear interpolation of lambda)
    lambdas = np.linspace(daily_lambda_start, daily_lambda_end, n_days)
    n_per_day = rng.poisson(lam=lambdas)
    total_n = int(n_per_day.sum())

    # Build claims and structural "risk score" (without intercept)
    base_logit = 0.0  # we'll choose intercept later to hit the target base rate
    raw_linear = []  # contribution sum before intercept
    rows = []
    claim_counter = 0

    for day in range(n_days):
        date = now + timedelta(days=day)
        k = int(n_per_day[day])
        for _ in range(k):
            claim_id = str(20_000_000 + claim_counter)
            claim_counter += 1

            # Sample entities (shop/phone risk-biased)
            shop_idx = rng.choice(n_shops, p=shop_weights)
            phone_idx = rng.choice(n_phones, p=phone_weights)

            # Plates: 70% from a ring (collusion), else random
            if rings and rng.random() < 0.70:
                ring = rings[rng.integers(len(rings))]
                plate = ring[rng.integers(len(ring))]
                in_ring = True
            else:
                plate = plates[rng.integers(n_plates)]
                in_ring = False

            row = {
                "claim_id": claim_id,
                "claim_date": date,
                "insurer_license_plate": plate,
                "insurer_phone_number": phones[phone_idx],
                "insurer_email": emails[rng.integers(n_emails)],
                "insurer_address": addrs[rng.integers(n_addrs)],
                "repair_shop": shops[shop_idx],
                "bank_account": banks[rng.integers(n_banks)],
                "claim_location": locs[rng.integers(n_locs)],
                "third_party_license_plate": thirdp[rng.integers(n_thirdp)],
            }

            # Linear contribution from risky entities
            lin = base_logit
            if risky_shop_mask[shop_idx]:
                lin += contrib_shop_risky
            if risky_phone_mask[phone_idx]:
                lin += contrib_phone_risky
            if in_ring:
                lin += contrib_plate_ring

            # Add Gaussian noise to control overlap
            lin += rng.normal(0.0, noise_sigma)

            raw_linear.append(lin)
            rows.append(row)

    raw_linear = np.array(raw_linear)

    # Calibrate intercept so mean(sigmoid(linear + intercept)) == target_base_rate
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    lo, hi = -8.0, 8.0  # intercept search bounds
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        avg = sigmoid(raw_linear + mid).mean()
        if avg < target_base_rate:
            lo = mid
        else:
            hi = mid
    intercept = 0.5 * (lo + hi)

    # Final draw of labels using calibrated probabilities
    rng2 = np.random.default_rng(seed + 7)
    probs = sigmoid(raw_linear + intercept)
    is_fraud = (rng2.random(len(probs)) < probs).astype(int)

    # Assemble final DataFrame
    for i, row in enumerate(rows):
        row["is_fraud"] = int(is_fraud[i])
    df = pd.DataFrame(rows).sort_values("claim_date").reset_index(drop=True)

    # Console summary to sanity-check separation
    print(f"Simulated rows: {len(df):,}")
    print(f"Achieved base_rate: {df['is_fraud'].mean():.3f} (target {target_base_rate:.3f})")
    # crude separability check: correlation between raw score and label
    corr = np.corrcoef(probs, is_fraud)[0, 1]
    print(f"Score/label correlation (higher is easier): {corr:.3f}")

    return df


def main():
    os.makedirs("data", exist_ok=True)
    df = simulate_multi_entity()
    df.to_csv("data/sy_dataset_1.csv", index=False)
    print("Saved: data/sy_dataset_1.csv")


if __name__ == "__main__":
    main()
