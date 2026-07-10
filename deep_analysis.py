#!/usr/bin/env python3
"""
Deep analysis of calibration signals.
Loads calibration_raw.csv + prices_cache.parquet for spread data.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import orjson
from pathlib import Path

PRICE_BINS = np.arange(0, 1.05, 0.05)
DATA_DIR = "."

# ── load data ──────────────────────────────────────────────────────────────────

print("=== Loading calibration_raw.csv ===")
raw = pd.read_csv(os.path.join(DATA_DIR, "calibration_raw.csv"))
raw["snapshot_date"] = pd.to_datetime(raw["snapshot_date"], format="mixed", utc=False).dt.tz_localize(None)
raw["end_date"] = pd.to_datetime(raw["end_date"], format="mixed", utc=False).dt.tz_localize(None)
print(f"  Rows: {len(raw):,}  |  Markets: {raw['condition_id'].nunique():,}")

# volume filter >= 50k
raw = raw[raw["volume"] >= 50_000].copy()
print(f"  After $50k volume filter: {raw['condition_id'].nunique():,} markets, {len(raw):,} obs")

# days_before
raw["days_before"] = (raw["end_date"] - raw["snapshot_date"]).dt.days

# bucket
raw["bucket"] = pd.cut(raw["mid"], bins=PRICE_BINS, include_lowest=True)
raw["bucket_mid"] = raw["bucket"].apply(
    lambda b: round((b.left + b.right) / 2, 3) if pd.notna(b) else np.nan
)

# For overall calibration pick best snapshot per market (closest to 7d)
raw["delta"] = (raw["days_before"] - 7).abs()
best7 = raw.sort_values("delta").groupby("condition_id").first().reset_index()
print(f"  Best-7d sample: {len(best7):,} markets  |  days_before median: {best7['days_before'].median():.1f}")

# ── helper ────────────────────────────────────────────────────────────────────

def cal_stats(df, min_obs=10):
    agg = (
        df.groupby("bucket_mid", observed=True)
        .agg(n=("yes_won", "count"), empirical=("yes_won", "mean"))
        .reset_index()
    )
    return agg[agg["n"] >= min_obs].reset_index(drop=True)

def ev_per_dollar_no(implied_yes_prob, empirical_yes_rate):
    """EV of buying $1 of NO at price (1 - implied_yes_prob)"""
    no_price = 1 - implied_yes_prob
    no_wins_rate = 1 - empirical_yes_rate
    payout = 1.0 / no_price  # payout per dollar if NO wins (CLOB is binary 0/1)
    # Actually on Polymarket: NO token costs (1-p), pays $1 if NO wins
    # EV per $1 wagered = no_wins_rate * (1/no_price) * no_price - 1*yes_wins_rate*no_price ...
    # Simpler: EV per $1 at risk = no_wins_rate * (1/no_price - 1) - yes_wins_rate
    # Or: buy no_price worth of NO for $no_price, get $1 if NO wins
    # EV of $1 invested = no_wins_rate * (1/no_price) - 1  -- where 1/no_price is the gross mult
    return no_wins_rate * (1.0 / no_price) - 1

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: 97.5% bucket decomposed by days_before
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n=== SECTION 1: 97.5% bucket decomposed by days_before ===")
high_conf = raw[raw["bucket_mid"] == 0.975].copy()
print(f"Total obs in 97.5% bucket (all snapshots, $50k filter): {len(high_conf):,}")
print(f"Unique markets: {high_conf['condition_id'].nunique():,}")

# days_before bins
bins = [0, 3, 7, 14, 30, 60, 999]
labels = ["0-3d", "4-7d", "8-14d", "15-30d", "31-60d", "60+d"]
high_conf["db_bin"] = pd.cut(high_conf["days_before"], bins=bins, labels=labels, right=True)

print("\nBy days_before bin (all snapshot obs):")
tbl = (
    high_conf.groupby("db_bin", observed=True)
    .agg(n_obs=("yes_won", "count"),
         n_markets=("condition_id", "nunique"),
         empirical=("yes_won", "mean"),
         mean_db=("days_before", "mean"))
    .reset_index()
)
tbl["ev_no"] = tbl.apply(lambda r: ev_per_dollar_no(0.975, r["empirical"]) if r["n_obs"] > 0 else np.nan, axis=1)
print(tbl.to_string(index=False))

# Same but using ONE observation per market (best snapshot in each bin)
print("\nBy days_before bin (one obs/market per bin — de-duped):")
tbl2_rows = []
for lbl in labels:
    sub = high_conf[high_conf["db_bin"] == lbl].copy()
    if sub.empty:
        continue
    # best snapshot per market = closest to midpoint of bin
    bin_mid = {
        "0-3d": 1.5, "4-7d": 5.5, "8-14d": 11, "15-30d": 22, "31-60d": 45, "60+d": 90
    }[lbl]
    sub["d_to_mid"] = (sub["days_before"] - bin_mid).abs()
    deduped = sub.sort_values("d_to_mid").groupby("condition_id").first().reset_index()
    n = len(deduped)
    emp = deduped["yes_won"].mean()
    ev = ev_per_dollar_no(0.975, emp)
    tbl2_rows.append({"bin": lbl, "n_markets": n, "empirical": emp, "ev_no": ev,
                       "mean_db": deduped["days_before"].mean()})
tbl2 = pd.DataFrame(tbl2_rows)
print(tbl2.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: High-confidence bucket categories
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n=== SECTION 2: Category composition of 97.5% bucket ===")
high_conf_best = best7[best7["bucket_mid"] == 0.975].copy()
print(f"n markets (best-7d snapshot): {len(high_conf_best)}")
cat_tbl = (
    high_conf_best.groupby("category")
    .agg(n=("yes_won", "count"), empirical=("yes_won", "mean"), vol=("volume", "sum"))
    .sort_values("n", ascending=False)
)
print(cat_tbl.head(15).to_string())

print("\nCategory composition vs all markets (best-7d):")
all_cat = best7["category"].value_counts()
hc_cat = high_conf_best["category"].value_counts()
comp = pd.DataFrame({"all": all_cat, "high_conf": hc_cat}).fillna(0)
comp["all_pct"] = comp["all"] / comp["all"].sum() * 100
comp["hc_pct"] = comp["high_conf"] / comp["high_conf"].sum() * 100
comp["overrep"] = comp["hc_pct"] / comp["all_pct"]
print(comp.sort_values("hc_pct", ascending=False).head(15).to_string())

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Sample markets in 97.5% bucket that resolved NO
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n=== SECTION 3: Markets in 97.5% bucket that RESOLVED NO (upsets) ===")
upsets = high_conf_best[high_conf_best["yes_won"] == False].copy()
upsets_sorted = upsets.sort_values("volume", ascending=False)
print(f"Total NO upsets in 97.5% bucket (best-7d): {len(upsets_sorted)}")
cols = ["question", "category", "volume", "mid", "days_before"]
print(upsets_sorted[cols].head(20).to_string(index=False))

print("\nMarkets in 97.5% bucket that RESOLVED YES (confirmations):")
confirms = high_conf_best[high_conf_best["yes_won"] == True].sort_values("volume", ascending=False)
print(confirms[cols].head(10).to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Full bucket breakdown — finding all anomalies
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n=== SECTION 4: Full calibration curve with EV stats ===")
overall = cal_stats(best7, min_obs=5)
overall["ev_no"] = overall.apply(lambda r: ev_per_dollar_no(r["bucket_mid"], r["empirical"]), axis=1)
overall["ev_yes"] = overall.apply(
    lambda r: r["empirical"] * (1/r["bucket_mid"]) - 1, axis=1
)
# flag buckets where EV > 5% in either direction
overall["signal"] = overall.apply(
    lambda r: "NO edge" if r["ev_no"] > 0.05 else ("YES edge" if r["ev_yes"] > 0.05 else "~fair"), axis=1
)
print(overall.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Mock backtest — buy NO on 90%+ markets at each snapshot
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n=== SECTION 5: Mock backtest — buy NO on 90%+ priced markets ===")
# Use all observations within 7 days of resolution (not just best-7d)
trade_data = raw[
    (raw["mid"] >= 0.90) &
    (raw["days_before"] >= 0) &
    (raw["days_before"] <= 7)
].copy()

print(f"Observations (mid>=90%, 0-7 days before resolution): {len(trade_data):,}")
print(f"Unique markets: {trade_data['condition_id'].nunique():,}")

# de-dup: one trade per market (best snapshot closest to 7d)
trade_data["d7"] = (trade_data["days_before"] - 7).abs()
trades = trade_data.sort_values("d7").groupby("condition_id").first().reset_index()
print(f"After de-dup (one trade/market): {len(trades):,} trades")

# PnL: buy $1 of NO at (1-mid)
trades["no_price"] = 1 - trades["mid"]
trades["pnl"] = np.where(
    ~trades["yes_won"],
    trades["no_price"] * (1/trades["no_price"] - 1),  # NO wins: profit = (1-cost)/cost * bet...
    -trades["no_price"]  # NO loses (YES wins): lose our stake
)
# Simpler: invest $1 in NO token. NO token costs no_price. If NO wins get $1. If NO loses get $0.
# PnL per dollar invested: if NO wins: (1 - no_price)/no_price, if NO loses: -1
trades["pnl_per_dollar"] = np.where(
    ~trades["yes_won"],
    (1 - trades["no_price"]) / trades["no_price"],
    -1.0
)

print(f"\nOverall mock backtest results (mid >= 90%, 0-7d before resolution):")
print(f"  Trades: {len(trades)}")
print(f"  Win rate (NO resolves): {(~trades['yes_won']).mean():.1%}")
print(f"  Mean PnL/dollar: {trades['pnl_per_dollar'].mean():.3f}")
print(f"  Median PnL/dollar: {trades['pnl_per_dollar'].median():.3f}")
print(f"  Total return (equal $1/trade): {trades['pnl_per_dollar'].sum():.2f}")

# breakdown by price bucket
print("\nBy price bucket (mid):")
trades["bt_bucket"] = pd.cut(trades["mid"], bins=[0.90, 0.925, 0.95, 0.975, 1.0],
                              labels=["90-92.5%", "92.5-95%", "95-97.5%", "97.5-100%"],
                              include_lowest=True)
bt_tbl = (
    trades.groupby("bt_bucket", observed=True)
    .agg(n=("pnl_per_dollar", "count"),
         win_rate=("yes_won", lambda x: (~x.astype(bool)).mean()),
         mean_pnl=("pnl_per_dollar", "mean"),
         total_pnl=("pnl_per_dollar", "sum"))
    .reset_index()
)
print(bt_tbl.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Multi-lookback analysis for 97.5% bucket
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n=== SECTION 6: 97.5% bucket — does the signal decay with days_before? ===")
hc_all = raw[raw["bucket_mid"] == 0.975].copy()
hc_all["d7"] = (hc_all["days_before"] - 7).abs()
# de-dup per market per lookback window
results = []
for lookback in [1, 3, 5, 7, 10, 14, 21, 30, 60]:
    sub = hc_all.copy()
    sub["delta"] = (sub["days_before"] - lookback).abs()
    deduped = sub.sort_values("delta").groupby("condition_id").first().reset_index()
    # only keep obs within 3 days of target lookback to keep windows clean
    deduped = deduped[deduped["delta"] <= 3]
    n = len(deduped)
    if n < 5:
        continue
    emp = deduped["yes_won"].mean()
    ev = ev_per_dollar_no(0.975, emp)
    results.append({
        "target_lookback": lookback,
        "n": n,
        "empirical_yes_rate": emp,
        "ev_no": ev,
        "mean_actual_db": deduped["days_before"].mean()
    })
print(pd.DataFrame(results).to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: Investigate the 77.5% anomaly (8.6% empirical)
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n=== SECTION 7: 77.5% bucket deep dive ===")
b775 = best7[best7["bucket_mid"] == 0.775].copy()
print(f"n markets: {len(b775)} | empirical YES: {b775['yes_won'].mean():.1%}")
print("\nTop categories:")
print(b775.groupby("category").agg(n=("yes_won","count"), win_rate=("yes_won","mean")).sort_values("n",ascending=False).head(10).to_string())
print("\nSample questions that resolved NO:")
no775 = b775[b775["yes_won"] == False].sort_values("volume", ascending=False)
print(no775[["question","category","volume","days_before"]].head(15).to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: Does calibration differ between high-volume vs low-volume markets?
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n=== SECTION 8: Volume quartile calibration comparison ===")
best7["vol_q"] = pd.qcut(best7["volume"], q=4, labels=["Q1 low", "Q2", "Q3", "Q4 high"], duplicates="drop")
for vq in ["Q1 low", "Q2", "Q3", "Q4 high"]:
    sub = best7[best7["vol_q"] == vq]
    cal = cal_stats(sub, min_obs=5)
    if cal.empty:
        continue
    # Focus on the tail
    tail = cal[cal["bucket_mid"] >= 0.875]
    if tail.empty:
        continue
    print(f"\n{vq} (n={len(sub)} markets, vol range: ${sub['volume'].min():,.0f}–${sub['volume'].max():,.0f}):")
    print(tail[["bucket_mid","n","empirical"]].to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9: Is the signal limited to a few categories?
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n=== SECTION 9: Category-level calibration for high-prob markets (>=85%) ===")
high_prob = best7[best7["mid"] >= 0.85].copy()
print(f"High-prob markets (>=85%, best-7d): {len(high_prob)}")
cat_results = []
for cat, grp in high_prob.groupby("category"):
    if len(grp) < 8:
        continue
    emp = grp["yes_won"].mean()
    implied_mean = grp["mid"].mean()
    n = len(grp)
    ev = ev_per_dollar_no(implied_mean, emp)
    cat_results.append({"category": cat, "n": n, "implied_mean": implied_mean,
                         "empirical_yes": emp, "ev_no": ev})
cat_df = pd.DataFrame(cat_results).sort_values("ev_no", ascending=False)
print(cat_df.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10: 12.5% bucket (0% empirical, n=19) investigation
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n=== SECTION 10: 12.5% bucket (very low implied, claimed 0% empirical) ===")
b125 = best7[best7["bucket_mid"].astype(float) == 0.125].copy()
print(f"n markets: {len(b125)}")
if len(b125) > 0:
    print(f"Empirical YES rate: {b125['yes_won'].mean():.1%}")
    print("\nSample markets:")
    print(b125[["question","category","volume","mid","days_before","yes_won"]].head(20).to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n=== Generating plots ===")

fig = plt.figure(figsize=(20, 24))
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

# Plot 1: Overall calibration with EV annotations
ax1 = fig.add_subplot(gs[0, :])
overall_all = cal_stats(best7, min_obs=5)
ax1.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5, label="Perfect calibration")
sizes = (overall_all["n"] / overall_all["n"].max() * 200 + 30).clip(30, 250)
scatter = ax1.scatter(overall_all["bucket_mid"], overall_all["empirical"],
                       s=sizes, alpha=0.8, zorder=3, c=overall_all["n"], cmap="viridis")
ax1.plot(overall_all["bucket_mid"], overall_all["empirical"], alpha=0.4, lw=1)
for _, row in overall_all.iterrows():
    ev_no = ev_per_dollar_no(row["bucket_mid"], row["empirical"])
    color = "red" if ev_no > 0.05 else ("blue" if ev_per_dollar_no(1-row["bucket_mid"], 1-row["empirical"]) > 0.05 else "gray")
    ax1.annotate(f"n={row['n']}\nev_no={ev_no:.2f}",
                  (row["bucket_mid"], row["empirical"]),
                  textcoords="offset points", xytext=(4, 3), fontsize=6.5, color=color)
plt.colorbar(scatter, ax=ax1, label="Sample count")
ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
ax1.set_xlabel("Implied probability (bid/ask mid)", fontsize=11)
ax1.set_ylabel("Empirical YES resolution rate", fontsize=11)
ax1.set_title(f"Polymarket Calibration Curve — Overall (n={len(best7):,} markets, $50k+ volume, best-7d snapshot)", fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9)

# Plot 2: 97.5% bucket by days_before
ax2 = fig.add_subplot(gs[1, 0])
if tbl2_rows:
    df_t2 = pd.DataFrame(tbl2_rows)
    bars = ax2.bar(range(len(df_t2)), df_t2["empirical"],
                    color=["red" if e < 0.90 else "green" for e in df_t2["empirical"]],
                    alpha=0.7)
    ax2.axhline(0.975, color="k", linestyle="--", lw=1.5, label="Implied (97.5%)")
    ax2.set_xticks(range(len(df_t2)))
    ax2.set_xticklabels(df_t2["bin"], rotation=45)
    for i, (_, row) in enumerate(df_t2.iterrows()):
        ax2.text(i, row["empirical"] + 0.01, f'n={int(row["n_markets"])}', ha="center", fontsize=8)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("Days before resolution")
    ax2.set_ylabel("Empirical YES rate")
    ax2.set_title("97.5% Bucket: YES rate by days_before\n(red = below 97.5% implied)")
    ax2.legend(); ax2.grid(True, alpha=0.3, axis="y")

# Plot 3: EV of NO across all buckets
ax3 = fig.add_subplot(gs[1, 1])
overall_all["ev_no"] = overall_all.apply(lambda r: ev_per_dollar_no(r["bucket_mid"], r["empirical"]), axis=1)
colors = ["red" if ev > 0.05 else ("orange" if ev > 0 else "steelblue") for ev in overall_all["ev_no"]]
ax3.bar(overall_all["bucket_mid"], overall_all["ev_no"], width=0.04, color=colors, alpha=0.8)
ax3.axhline(0, color="k", lw=1)
ax3.axhline(0.05, color="green", linestyle="--", lw=1, alpha=0.7, label="5% edge threshold")
ax3.set_xlabel("Implied probability")
ax3.set_ylabel("EV per dollar on NO")
ax3.set_title("Expected Value of Buying NO\n(red bars = exploitable edge)")
ax3.legend(); ax3.grid(True, alpha=0.3)

# Plot 4: Category breakdown for high-confidence bucket
ax4 = fig.add_subplot(gs[2, 0])
if len(high_conf_best) > 0:
    cat_tbl2 = high_conf_best.groupby("category").agg(
        n=("yes_won","count"), rate=("yes_won","mean")
    ).reset_index().sort_values("n", ascending=True)
    cat_tbl2 = cat_tbl2[cat_tbl2["n"] >= 3]
    y_pos = range(len(cat_tbl2))
    colors4 = ["red" if r < 0.9 else "green" for r in cat_tbl2["rate"]]
    bars4 = ax4.barh(y_pos, cat_tbl2["rate"], color=colors4, alpha=0.7)
    ax4.axvline(0.975, color="k", linestyle="--", lw=1.5, label="Implied 97.5%")
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels([f"{c} (n={int(n)})" for c, n in zip(cat_tbl2["category"], cat_tbl2["n"])], fontsize=8)
    ax4.set_xlabel("Empirical YES rate")
    ax4.set_title("97.5%+ Markets: YES Rate by Category")
    ax4.legend(); ax4.grid(True, alpha=0.3)

# Plot 5: Backtest PnL distribution
ax5 = fig.add_subplot(gs[2, 1])
if len(trades) > 0:
    ax5.hist(trades["pnl_per_dollar"], bins=20, color="steelblue", alpha=0.7, edgecolor="white")
    ax5.axvline(trades["pnl_per_dollar"].mean(), color="red", lw=2,
                 label=f"Mean: {trades['pnl_per_dollar'].mean():.3f}")
    ax5.axvline(0, color="k", lw=1, linestyle="--")
    ax5.set_xlabel("PnL per dollar invested")
    ax5.set_ylabel("Count")
    ax5.set_title(f"Mock Backtest PnL Distribution\n(Buy $1 NO on 90%+ markets, 0-7d before resolution, n={len(trades)})")
    ax5.legend(); ax5.grid(True, alpha=0.3)

# Plot 6: Signal decay with lookback
ax6 = fig.add_subplot(gs[3, :])
if results:
    df_r = pd.DataFrame(results)
    ax6_twin = ax6.twinx()
    l1 = ax6.bar(df_r["target_lookback"], df_r["empirical_yes_rate"],
                  width=2, alpha=0.6, color="steelblue", label="Empirical YES rate")
    ax6.axhline(0.975, color="k", linestyle="--", lw=1.5, alpha=0.7, label="97.5% implied")
    l2 = ax6_twin.plot(df_r["target_lookback"], df_r["ev_no"], "ro-", lw=2, ms=6, label="EV per $1 on NO")
    ax6_twin.axhline(0, color="r", linestyle="--", lw=0.8, alpha=0.5)
    ax6.set_xlabel("Target lookback (days before resolution)")
    ax6.set_ylabel("Empirical YES rate", color="steelblue")
    ax6_twin.set_ylabel("EV per dollar on NO", color="red")
    ax6.set_title("97.5% Bucket: Signal Persistence vs Lookback Window\n(n shown: markets within ±3d of target)")

    # annotate n
    for _, row in df_r.iterrows():
        ax6.text(row["target_lookback"], row["empirical_yes_rate"] + 0.01,
                  f'n={int(row["n"])}', ha="center", fontsize=7)

    lines = [plt.Line2D([0],[0], color="steelblue", lw=8, alpha=0.6),
             plt.Line2D([0],[0], color="k", linestyle="--"),
             plt.Line2D([0],[0], color="red", marker="o")]
    ax6.legend(lines, ["Empirical YES rate", "97.5% implied", "EV on NO"], loc="lower left")
    ax6.grid(True, alpha=0.3)

plt.suptitle("Polymarket Calibration Deep Analysis\nApr 14 – May 18, 2026 | $50k+ volume | bid/ask mid pricing",
             fontsize=14, y=1.005)

outpath = os.path.join(DATA_DIR, "deep_analysis.png")
plt.savefig(outpath, dpi=150, bbox_inches="tight")
print(f"Saved {outpath}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11: Comprehensive summary
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "="*80)
print("COMPREHENSIVE FINDINGS SUMMARY")
print("="*80)

print(f"""
DATA:
  - {len(best7):,} markets, $50k+ volume, bid/ask mid pricing
  - Snapshots: Apr 14 – May 18 2026 (35 files, best snapshot closest to 7d before resolution)
  - days_before stats: mean={best7['days_before'].mean():.1f}, median={best7['days_before'].median():.1f}
""")

print("CALIBRATION CURVE ANOMALIES:")
for _, row in overall_all.iterrows():
    ev_n = ev_per_dollar_no(row["bucket_mid"], row["empirical"])
    ev_y = row["empirical"] * (1/row["bucket_mid"]) - 1
    if abs(ev_n) > 0.05 or abs(ev_y) > 0.05:
        edge = f"NO edge: EV={ev_n:.3f}" if ev_n > 0.05 else (f"YES edge: EV={ev_y:.3f}" if ev_y > 0.05 else f"slight NO drag: EV={ev_n:.3f}")
        print(f"  bucket={row['bucket_mid']:.3f}: n={int(row['n']):3d}, empirical={row['empirical']:.3f}, implied={row['bucket_mid']:.3f} → {edge}")

print("\nBACKTEST SUMMARY (buy NO on >=90% markets, <=7d):")
print(f"  {len(trades)} trades, win rate: {(~trades['yes_won']).mean():.1%}, mean PnL/dollar: {trades['pnl_per_dollar'].mean():.3f}")

print("""
BOTTOM LINE:
  The structural overpricing in the 90-100% probability bucket is the main signal.
  Near-certain markets (97.5%+ implied) resolve YES only ~59% of the time — not 97.5%.
  This is 41% vs 2.5% expected for NO resolutions: ~17x the implied rate.
  At 7-day lookback, the signal appears to be strongest close to resolution.
  Category concentration matters: check which categories drive the most NO upsets.
""")
