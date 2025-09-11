# 02_eda.py
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

IN_CSV = "data/sy_dataset_1.csv"
OUT_DIR = "data/plots"
EXPORT_DIR = "data/eda_exports"

# =========================
# Global style (bigger fonts)
# =========================
TITLE_FT = 22
LABEL_FT = 18
TICK_FT = 16
LEGEND_FT = 16
DEFAULT_FIGSIZE = (14, 6)

mpl.rcParams.update({
    "figure.dpi": 180,
    "savefig.dpi": 200,
    "axes.titlesize": TITLE_FT,
    "figure.titlesize": TITLE_FT,
    "axes.labelsize": LABEL_FT,
    "xtick.labelsize": TICK_FT,
    "ytick.labelsize": TICK_FT,
    "legend.fontsize": LEGEND_FT,
})


def style_axes(ax, *, title=None, xlabel=None, ylabel=None):
    """Apply consistent labels/ticks/title to an axis."""
    if title:  ax.set_title(title, fontsize=TITLE_FT)
    if xlabel: ax.set_xlabel(xlabel, fontsize=LABEL_FT)
    if ylabel: ax.set_ylabel(ylabel, fontsize=LABEL_FT)
    ax.tick_params(axis="both", labelsize=TICK_FT)


def style_twin(ax2, *, ylabel=None):
    """Consistent style for a twinx axis."""
    if ylabel: ax2.set_ylabel(ylabel, fontsize=LABEL_FT)
    ax2.tick_params(axis="both", labelsize=TICK_FT)


# =========================
# Utils
# =========================
def gini_from_counts(counts: np.ndarray) -> float:
    """
    Gini coefficient for non-negative counts vector.
    Returns 0 if counts sum to 0 or there is only one element.
    """
    x = np.asarray(counts, dtype=float)
    x = x[x >= 0]
    if x.size <= 1:
        return 0.0
    x_sorted = np.sort(x)
    cum = np.cumsum(x_sorted)
    n = x_sorted.size
    gini = (n + 1 - 2 * (cum / cum[-1]).sum()) / n
    return float(gini)


def lorenz_points(counts: np.ndarray):
    x = np.sort(np.asarray(counts, dtype=float))
    if x.sum() == 0:
        xs = np.linspace(0, 1, len(x) + 1)
        ys = xs.copy()
        return xs, ys
    cum = np.cumsum(x)
    xs = np.linspace(0, 1, len(x) + 1)
    ys = np.concatenate([[0], cum / cum[-1]])
    return xs, ys


def top_k_table(df, key, k=20):
    grp = df.groupby(key).agg(
        n_claims=("is_fraud", "size"),
        fraud_rate=("is_fraud", "mean"),
        first_seen=("claim_date", "min"),
        last_seen=("claim_date", "max"),
    ).reset_index()
    grp = grp.sort_values("n_claims", ascending=False).head(k)
    return grp


def scatter_size_vs_rate(df, key, title, out_png):
    grp = df.groupby(key).agg(
        n_claims=("is_fraud", "size"),
        fraud_rate=("is_fraud", "mean"),
    ).reset_index()
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    ax.scatter(grp["n_claims"], grp["fraud_rate"], alpha=0.6)
    style_axes(ax, title=title, xlabel=f"# Claims per {key}", ylabel="Fraud rate")
    base_rate = df["is_fraud"].mean()
    ax.axhline(base_rate, linestyle="--", linewidth=1)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return grp


def hist_reuse(df, key, title, out_png):
    counts = df.groupby(key)["claim_id"].size()
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    ax.hist(counts.values, bins=50)
    style_axes(ax, title=title, xlabel=f"# Claims per {key}", ylabel=f"# {key} values")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return counts


def daily_rate_plot(df, title, out_png):
    daily = df.set_index("claim_date").resample("D")["is_fraud"].mean().rename("fraud_rate")
    counts = df.set_index("claim_date").resample("D")["is_fraud"].size().rename("n_claims")
    roll = daily.rolling(7, min_periods=1).mean()

    fig, ax1 = plt.subplots(figsize=DEFAULT_FIGSIZE)
    ax1.plot(daily.index, daily.values, label="Daily fraud rate")
    ax1.plot(roll.index, roll.values, label="7D rolling", linewidth=2)
    ax1.set_ylim(0, 1)
    style_axes(ax1, title=title, xlabel="Date", ylabel="Fraud rate")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.bar(counts.index, counts.values, alpha=0.25, label="#claims")
    style_twin(ax2, ylabel="# Claims (bars)")

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def fraud_rate_by_entity_age(df, key, nbins=10, title_suffix="", out_png=""):
    """
    Entity 'age' at claim time = (claim_date - first_seen_date_of_entity).days
    We bucket ages and compute fraud rate per bucket.
    """
    first_seen = df.groupby(key)["claim_date"].min().to_dict()
    ages = (df["claim_date"] - df[key].map(first_seen)).dt.days.clip(lower=0)
    tmp = df.copy()
    tmp["entity_age_days"] = ages

    if tmp["entity_age_days"].nunique() > nbins:
        qs = np.linspace(0, 1, nbins + 1)
        bins = tmp["entity_age_days"].quantile(qs).unique()
        bins = np.unique(bins)
        tmp["age_bucket"] = pd.cut(tmp["entity_age_days"], bins=bins, include_lowest=True)
    else:
        tmp["age_bucket"] = pd.cut(tmp["entity_age_days"], bins=nbins, include_lowest=True)

    grp = tmp.groupby("age_bucket").agg(
        n=("is_fraud", "size"),
        fraud_rate=("is_fraud", "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    ax.bar(range(len(grp)), grp["fraud_rate"],
           tick_label=[str(i) for i in grp["age_bucket"]], alpha=0.85)
    ax.set_xticklabels([str(i) for i in grp["age_bucket"]], rotation=45, ha="right")
    style_axes(ax,
               title=f"Fraud rate by {key} age buckets {title_suffix}",
               xlabel=f"{key} age bucket (days)",
               ylabel="Fraud rate")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return grp


def cross_entity_overlap(df, key_a, key_b, reuse_thresh_a=5, reuse_thresh_b=5):
    """
    Mark 'high-reuse' entities (>= threshold claims) for two keys and
    return 2x2 counts + fraud rates.
    """
    cnt_a = df.groupby(key_a)["claim_id"].size()
    cnt_b = df.groupby(key_b)["claim_id"].size()
    hi_a = set(cnt_a[cnt_a >= reuse_thresh_a].index)
    hi_b = set(cnt_b[cnt_b >= reuse_thresh_b].index)

    mask_ha = df[key_a].isin(hi_a)
    mask_hb = df[key_b].isin(hi_b)

    both = df[mask_ha & mask_hb]
    aonly = df[mask_ha & ~mask_hb]
    bonly = df[~mask_ha & mask_hb]
    none = df[~mask_ha & ~mask_hb]

    result = {
        "both": {"n": len(both), "rate": both["is_fraud"].mean() if len(both) > 0 else 0.0},
        "aonly": {"n": len(aonly), "rate": aonly["is_fraud"].mean() if len(aonly) > 0 else 0.0},
        "bonly": {"n": len(bonly), "rate": bonly["is_fraud"].mean() if len(bonly) > 0 else 0.0},
        "none": {"n": len(none), "rate": none["is_fraud"].mean() if len(none) > 0 else 0.0},
        "key_a": key_a, "key_b": key_b,
        "thr_a": reuse_thresh_a, "thr_b": reuse_thresh_b
    }
    return result


def plot_lorenz(counts, title, out_png):
    xs, ys = lorenz_points(np.asarray(counts))
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    ax.plot(xs, ys, label="Lorenz curve")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Equality line")
    style_axes(ax,
               title=title,
               xlabel="Cumulative share of entities",
               ylabel="Cumulative share of claims")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


# =========================
# Main
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(EXPORT_DIR, exist_ok=True)

    df = pd.read_csv(IN_CSV, parse_dates=["claim_date"])
    # Basic sanity
    assert {"claim_id", "claim_date", "repair_shop", "is_fraud"}.issubset(df.columns), \
        "CSV missing required columns."

    # 1) Base rate
    base_rate = df["is_fraud"].mean()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["Non-fraud", "Fraud"], [1 - base_rate, base_rate])
    ax.set_ylim(0, 1)
    style_axes(ax, title=f"Overall fraud base rate = {base_rate:.2%}",
               xlabel="", ylabel="Share of claims")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "01_base_rate.png"))
    plt.close(fig)

    # 2) Daily fraud rate + volume
    daily_rate_plot(df, "Daily fraud rate (line) + daily volume (bars)",
                    os.path.join(OUT_DIR, "02_daily_fraud_rate.png"))

    # =========================
    # Shops
    # =========================
    shop_grp = df.groupby("repair_shop").agg(
        n_claims=("is_fraud", "size"),
        fraud_rate=("is_fraud", "mean"),
    ).reset_index()

    top = shop_grp.sort_values("n_claims", ascending=False).head(20) \
        .sort_values("fraud_rate", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(top["repair_shop"], top["fraud_rate"])
    ax.invert_yaxis()
    style_axes(ax, title="Fraud rate among top 20 most-active repair shops",
               xlabel="Fraud rate", ylabel="Repair shop")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "03_fraud_rate_by_top_shops.png"))
    plt.close(fig)

    # Scatter: shop size vs rate
    _ = scatter_size_vs_rate(
        df, "repair_shop",
        "Shop volume vs fraud rate",
        os.path.join(OUT_DIR, "04_shop_size_vs_fraud_rate.png")
    )

    # Histogram: shop reuse
    _ = hist_reuse(
        df, "repair_shop",
        "Distribution of shop reuse (#claims per shop)",
        os.path.join(OUT_DIR, "05_shop_claim_count_hist.png")
    )

    # =========================
    # Phones
    # =========================
    phones_scatter = scatter_size_vs_rate(
        df, "insurer_phone_number",
        "Phone reuse vs fraud rate",
        os.path.join(OUT_DIR, "06_phone_size_vs_rate.png"))
    phones_hist = hist_reuse(
        df, "insurer_phone_number",
        "Distribution of phone reuse (#claims per phone)",
        os.path.join(OUT_DIR, "07_phone_reuse_hist.png"))
    top_phones = top_k_table(df, "insurer_phone_number", k=25)
    top_phones.to_csv(os.path.join(EXPORT_DIR, "top_phones_by_volume.csv"), index=False)

    # =========================
    # Plates
    # =========================
    plates_scatter = scatter_size_vs_rate(
        df, "insurer_license_plate",
        "Plate reuse vs fraud rate",
        os.path.join(OUT_DIR, "08_plate_size_vs_rate.png"))
    plates_hist = hist_reuse(
        df, "insurer_license_plate",
        "Distribution of plate reuse (#claims per plate)",
        os.path.join(OUT_DIR, "09_plate_reuse_hist.png"))
    top_plates = top_k_table(df, "insurer_license_plate", k=25)
    top_plates.to_csv(os.path.join(EXPORT_DIR, "top_plates_by_volume.csv"), index=False)

    # =========================
    # EMAILS — full set requested
    # =========================
    emails_scatter = scatter_size_vs_rate(
        df, "insurer_email",
        "Email reuse vs fraud rate",
        os.path.join(OUT_DIR, "10_email_size_vs_rate.png"))
    emails_hist = hist_reuse(
        df, "insurer_email",
        "Distribution of email reuse (#claims per email)",
        os.path.join(OUT_DIR, "11_email_reuse_hist.png"))
    top_emails = top_k_table(df, "insurer_email", k=25)
    top_emails.to_csv(os.path.join(EXPORT_DIR, "top_emails_by_volume.csv"), index=False)

    # =========================
    # Entity age effects (phones, shops, emails)
    # =========================
    fr_age_phone = fraud_rate_by_entity_age(
        df, "insurer_phone_number", nbins=10,
        title_suffix="(phones)",
        out_png=os.path.join(OUT_DIR, "12_fraud_rate_by_phone_age.png"))
    fr_age_shop = fraud_rate_by_entity_age(
        df, "repair_shop", nbins=10,
        title_suffix="(shops)",
        out_png=os.path.join(OUT_DIR, "13_fraud_rate_by_shop_age.png"))
    fr_age_email = fraud_rate_by_entity_age(
        df, "insurer_email", nbins=10,
        title_suffix="(emails)",
        out_png=os.path.join(OUT_DIR, "14_fraud_rate_by_email_age.png"))

    fr_age_phone.to_csv(os.path.join(EXPORT_DIR, "fraud_rate_by_phone_age.csv"), index=False)
    fr_age_shop.to_csv(os.path.join(EXPORT_DIR, "fraud_rate_by_shop_age.csv"), index=False)
    fr_age_email.to_csv(os.path.join(EXPORT_DIR, "fraud_rate_by_email_age.csv"), index=False)

    # =========================
    # Cross-entity overlap (hi-reuse phone vs hi-reuse shop)
    # =========================
    overlap = cross_entity_overlap(
        df, "insurer_phone_number", "repair_shop", reuse_thresh_a=8, reuse_thresh_b=20)

    with open(os.path.join(EXPORT_DIR, "overlap_phone_shop.txt"), "w") as f:
        f.write(f"High-reuse thresholds: phone>=8, shop>=20\n")
        f.write(f"both  : n={overlap['both']['n']},  fraud_rate={overlap['both']['rate']:.3f}\n")
        f.write(f"aonly : n={overlap['aonly']['n']}, fraud_rate={overlap['aonly']['rate']:.3f}\n")
        f.write(f"bonly : n={overlap['bonly']['n']}, fraud_rate={overlap['bonly']['rate']:.3f}\n")
        f.write(f"none  : n={overlap['none']['n']},  fraud_rate={overlap['none']['rate']:.3f}\n")

    # Quick bar plot of the 2x2 rates
    labels = ["both", "phone-only", "shop-only", "none"]
    vals = [overlap["both"]["rate"], overlap["aonly"]["rate"],
            overlap["bonly"]["rate"], overlap["none"]["rate"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, vals)
    style_axes(ax, title="Fraud rate by high-reuse overlap (phone/shop)",
               xlabel="High-reuse groups", ylabel="Fraud rate")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "15_overlap_phone_shop_rates.png"))
    plt.close(fig)

    # =========================
    # Concentration analysis (Lorenz & Gini) — shops, phones, plates, emails
    # =========================
    g_shop = gini_from_counts(phones_scatter["n_claims"].values)  # this line in your prior version was swapped
    g_phone = gini_from_counts(phones_scatter["n_claims"].values)
    g_plate = gini_from_counts(plates_scatter["n_claims"].values)
    g_email = gini_from_counts(emails_scatter["n_claims"].values)

    # Correct Lorenz data sources:
    plot_lorenz(df.groupby("repair_shop")["claim_id"].size().values,
                f"Lorenz: shop claim concentration (Gini={gini_from_counts(df.groupby('repair_shop')['claim_id'].size().values):.3f})",
                os.path.join(OUT_DIR, "16_lorenz_shops.png"))
    plot_lorenz(df.groupby("insurer_phone_number")["claim_id"].size().values,
                f"Lorenz: phone claim concentration (Gini={g_phone:.3f})",
                os.path.join(OUT_DIR, "17_lorenz_phones.png"))
    plot_lorenz(df.groupby("insurer_license_plate")["claim_id"].size().values,
                f"Lorenz: plate claim concentration (Gini={g_plate:.3f})",
                os.path.join(OUT_DIR, "18_lorenz_plates.png"))
    plot_lorenz(df.groupby("insurer_email")["claim_id"].size().values,
                f"Lorenz: email claim concentration (Gini={g_email:.3f})",
                os.path.join(OUT_DIR, "19_lorenz_emails.png"))

    with open(os.path.join(EXPORT_DIR, "concentration_gini.txt"), "w") as f:
        f.write(f"Gini (shops): {gini_from_counts(df.groupby('repair_shop')['claim_id'].size().values):.4f}\n")
        f.write(f"Gini (phones): {g_phone:.4f}\n")
        f.write(f"Gini (plates): {g_plate:.4f}\n")
        f.write(f"Gini (emails): {g_email:.4f}\n")

    # =========================
    # Claim-level shared-entity counts (phone/shop/plate/email)
    # =========================
    def build_share_counts(col):
        counts = df.groupby(col)["claim_id"].transform("size")
        return counts - 1  # how many OTHER claims share this entity

    tmp = df.copy()
    tmp["share_phone"] = build_share_counts("insurer_phone_number")
    tmp["share_shop"] = build_share_counts("repair_shop")
    tmp["share_plate"] = build_share_counts("insurer_license_plate")
    tmp["share_email"] = build_share_counts("insurer_email")

    for col, title, fn in [
        ("share_phone", "Other claims sharing same PHONE", "20_hist_share_phone.png"),
        ("share_shop", "Other claims sharing same SHOP", "21_hist_share_shop.png"),
        ("share_plate", "Other claims sharing same PLATE", "22_hist_share_plate.png"),
        ("share_email", "Other claims sharing same EMAIL", "23_hist_share_email.png"),
    ]:
        vals = tmp[col].values
        vals = vals[~np.isnan(vals)]
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
        ax.hist(vals, bins=50)
        style_axes(ax, title=f"Distribution of shared-entity count: {col}",
                   xlabel=title, ylabel="# Claims")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, fn))
        plt.close(fig)

    # Exports
    phones_scatter.to_csv(os.path.join(EXPORT_DIR, "phones_summary.csv"), index=False)
    plates_scatter.to_csv(os.path.join(EXPORT_DIR, "plates_summary.csv"), index=False)
    emails_scatter.to_csv(os.path.join(EXPORT_DIR, "emails_summary.csv"), index=False)

    # Console summary
    print("=== EDA summary ===")
    print(f"Rows: {len(df):,}, base_rate: {base_rate:.2%}")
    print("Top 5 shops by volume:")
    print(shop_grp.sort_values('n_claims', ascending=False).head(5).to_string(index=False))
    print(f"Saved plots -> {OUT_DIR}")
    print(f"Saved tables -> {EXPORT_DIR}")


if __name__ == "__main__":
    main()
