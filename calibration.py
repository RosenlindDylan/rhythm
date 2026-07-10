#!/usr/bin/env python3
"""
Polymarket calibration pipeline.

Steps:
  1. Parse PMXT hourly parquet snapshots → price per market per date
  2. Fetch resolved binary markets from Gamma API
  3. Join on conditionId, keep observations taken before resolution
  4. Build calibration curve (implied prob vs empirical resolution rate)
  5. Segment by category and volume quartile
  6. Save CSVs + PNG

Usage:
  pip install pandas pyarrow orjson requests matplotlib
  python calibration.py [--snapshot-dir .] [--lookback 7]
"""

import argparse
import glob
import os
import sys
import time
from pathlib import Path
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import orjson
import pandas as pd
import pyarrow.parquet as pq
import requests

GAMMA_BASE = "https://gamma-api.polymarket.com"
CHUNK_SIZE = 500_000
PRICE_BINS = np.arange(0, 1.05, 0.05)
MIN_OBS_PER_BUCKET = 15


# ── 1. Parse PMXT snapshots ───────────────────────────────────────────────────
# v2 schema: market (bytes hex), asset_id (int token ID), price (trade price),
# timestamp, event_type, side (BUY/SELL), best_bid, best_ask, ...
# We take last trade price per (market, asset_id) as the probability estimate.
# Join to Gamma API on market=conditionId AND asset_id=yes_token_id to get YES prices.

def parse_snapshot(path: str) -> pd.DataFrame:
    """
    Read one hourly parquet file.
    Returns DataFrame(market_id, asset_id, mid, snapshot_date).
    """
    stem = Path(path).stem  # polymarket_orderbook_2026-04-16T00
    snapshot_dt = pd.Timestamp(stem.split("polymarket_orderbook_")[1])

    pf = pq.ParquetFile(path)
    # running dict avoids accumulating all chunks in memory before groupby
    last_mids: dict[tuple, float] = {}

    for batch in pf.iter_batches(batch_size=CHUNK_SIZE):
        df = batch.to_pandas()

        df["market_id"] = df["market"].apply(
            lambda x: x.decode("utf-8") if isinstance(x, bytes) else str(x)
        ).str.lower()

        valid = df.dropna(subset=["market_id", "asset_id"])
        if valid.empty:
            continue

        # prefer orderbook mid (best_bid+best_ask)/2 — stable standing price;
        # for near-zero markets (negRisk losers), bid dries up to 0 and last trade
        # is chaotic settlement noise — use ask directly when bid=0 and ask is small;
        # fall back to last trade price only when no usable quote exists
        valid = valid.copy()
        bid = pd.to_numeric(valid.get("best_bid"), errors="coerce")
        ask = pd.to_numeric(valid.get("best_ask"), errors="coerce")
        have_both = bid.notna() & ask.notna() & (bid > 0) & (ask > 0)
        have_ask_only = ask.notna() & (ask > 0) & (ask < 0.25) & ~have_both
        valid["mid"] = np.where(
            have_both,
            (bid + ask) / 2,
            np.where(
                have_ask_only,
                ask,
                pd.to_numeric(valid.get("price"), errors="coerce"),
            ),
        )
        valid = valid.dropna(subset=["mid"])
        if valid.empty:
            continue

        # last mid per (market, token) within this chunk
        valid = valid.sort_values("timestamp")
        batch_last = valid.groupby(["market_id", "asset_id"])["mid"].last()
        last_mids.update(batch_last.to_dict())

    if not last_mids:
        return pd.DataFrame(columns=["market_id", "asset_id", "mid", "snapshot_date"])

    result = pd.DataFrame(
        [(mid, aid, price) for (mid, aid), price in last_mids.items()],
        columns=["market_id", "asset_id", "mid"],
    )
    result["snapshot_date"] = snapshot_dt.normalize()
    return result


def _parse_and_report(path: str) -> pd.DataFrame:
    df = parse_snapshot(path)
    print(f"  {Path(path).name} ... {len(df):,} (market, token) pairs")
    return df


def load_snapshots(snapshot_dir: str) -> pd.DataFrame:
    import multiprocessing
    files = sorted(glob.glob(os.path.join(snapshot_dir, "polymarket_orderbook_*.parquet")))
    if not files:
        sys.exit(f"No parquet files found in {snapshot_dir!r}")

    workers = min(len(files), multiprocessing.cpu_count())
    print(f"  Parsing {len(files)} files across {workers} workers...")
    with multiprocessing.Pool(workers) as pool:
        frames = pool.map(_parse_and_report, files)

    return pd.concat(frames, ignore_index=True)


# ── 2. Gamma API ──────────────────────────────────────────────────────────────

def fetch_resolved_markets(end_date_min: str = "2026-04-01") -> list[dict]:
    all_markets: list[dict] = []
    limit, offset = 500, 0
    effective_limit = limit  # detected from first response

    while True:
        r = requests.get(
            f"{GAMMA_BASE}/markets",
            params={
                "closed": "true",
                "limit": limit,
                "offset": offset,
                "end_date_min": end_date_min,
                "order": "end_date_iso",
                "ascending": "false",
            },
            timeout=30,
        )
        if r.status_code == 422:
            break
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        all_markets.extend(batch)
        print(f"  {len(all_markets):,} markets fetched\r", end="", flush=True)
        # detect API page-size cap on first call
        if offset == 0 and len(batch) < limit:
            effective_limit = len(batch)
        # stop only when we get fewer than the effective page size
        if len(batch) < effective_limit:
            break
        offset += len(batch)
        sleep(0.05)

    print()
    return all_markets


def fetch_clob_token_map() -> dict[str, str]:
    """
    Page through all CLOB markets and return {condition_id: yes_token_id}.
    The CLOB API explicitly labels each token as Yes/No — unlike Gamma's clobTokenIds.
    """
    token_map: dict[str, str] = {}
    cursor = ""

    while True:
        r = requests.get(
            "https://clob.polymarket.com/markets",
            params={"next_cursor": cursor, "limit": 1000},
            timeout=30,
        )
        if r.status_code in (400, 422):
            break  # end of pagination
        r.raise_for_status()
        data = r.json()

        for m in data.get("data", []):
            cid = m.get("condition_id", "")
            for t in m.get("tokens", []):
                if str(t.get("outcome", "")).lower() == "yes":
                    token_map[cid] = str(t["token_id"])
                    break

        cursor = data.get("next_cursor") or ""
        print(f"  {len(token_map):,} CLOB markets mapped\r", end="", flush=True)
        if not cursor or not data.get("data"):
            break
        sleep(0.02)

    print()
    return token_map


def _parse_json_field(raw) -> list:
    if isinstance(raw, str):
        return orjson.loads(raw)
    return raw or []


def extract_market_row(m: dict) -> dict | None:
    cid = (m.get("conditionId") or "").lower()
    if not cid:
        return None

    try:
        outcomes = _parse_json_field(m.get("outcomes"))
        prices = [float(x) for x in _parse_json_field(m.get("outcomePrices"))]
    except Exception:
        return None

    ol = [str(o).lower() for o in outcomes]
    if set(ol) != {"yes", "no"} or len(prices) != len(ol):
        return None

    yes_price = prices[ol.index("yes")]
    if yes_price > 0.9:
        yes_won = True
    elif yes_price < 0.1:
        yes_won = False
    else:
        return None  # ambiguous / still live

    end_raw = m.get("endDate") or m.get("endDateIso") or ""
    try:
        ts = pd.Timestamp(end_raw)
        end_date = ts.tz_convert(None) if ts.tzinfo else ts
    except Exception:
        return None

    volume = float(m.get("volumeNum") or m.get("volume") or 0)
    category = m.get("groupItemTagSlug") or m.get("category") or "unknown"

    # event_id groups correlated negRisk sub-markets (like all "Will blah win MVP?" markets)
    events = m.get("events") or []
    event_id = str(events[0]["id"]) if events else cid

    return {
        "condition_id": cid,
        "yes_won": yes_won,
        "end_date": end_date,
        "volume": volume,
        "category": category,
        "question": str(m.get("question", ""))[:100],
        "event_id": event_id,
        "neg_risk": bool(m.get("negRisk", False)),
    }


# ── 3. Calibration stats ──────────────────────────────────────────────────────

def calibration_stats(df: pd.DataFrame, min_obs: int = MIN_OBS_PER_BUCKET) -> pd.DataFrame:
    df = df.copy()
    df["bucket"] = pd.cut(df["mid"], bins=PRICE_BINS, include_lowest=True)
    df["bucket_mid"] = df["bucket"].apply(
        lambda b: round((b.left + b.right) / 2, 3) if pd.notna(b) else np.nan
    )
    agg = (
        df.groupby("bucket_mid", observed=True)
        .agg(n=("yes_won", "count"), empirical=("yes_won", "mean"))
        .reset_index()
    )
    return agg[agg["n"] >= min_obs].reset_index(drop=True)


# ── 4. Plotting ───────────────────────────────────────────────────────────────

def calibration_panel(ax, cal: pd.DataFrame, title: str):
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Perfect")
    if cal.empty:
        ax.set_title(f"{title}\n(insufficient data)")
        return
    sizes = (cal["n"] / cal["n"].max() * 150 + 20).clip(20, 200)
    ax.scatter(cal["bucket_mid"], cal["empirical"], s=sizes, alpha=0.75, zorder=3)
    ax.plot(cal["bucket_mid"], cal["empirical"], alpha=0.5)
    for _, row in cal.iterrows():
        ax.annotate(
            f"n={row['n']}",
            (row["bucket_mid"], row["empirical"]),
            textcoords="offset points", xytext=(4, 3), fontsize=6, alpha=0.6,
        )
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Implied probability"); ax.set_ylabel("Empirical resolution rate")
    ax.set_title(title); ax.grid(True, alpha=0.3)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-dir", default=".", help="Directory with parquet files")
    parser.add_argument("--lookback", type=int, default=7, help="Target days-before-resolution")
    parser.add_argument("--min-volume", type=float, default=100_000, help="Minimum market volume in USD")
    args = parser.parse_args()

    # ── 1 ──
    cache_path = os.path.join(args.snapshot_dir, "prices_cache.parquet")
    if os.path.exists(cache_path):
        print(f"=== 1. Loading snapshot cache ({cache_path}) ===")
        prices = pd.read_parquet(cache_path)
    else:
        print("=== 1. Loading PMXT snapshots ===")
        prices = load_snapshots(args.snapshot_dir)
        prices.to_parquet(cache_path, index=False)
        print(f"Saved cache → {cache_path}")
    dates = sorted(prices["snapshot_date"].unique())
    print(f"Total rows: {len(prices):,} | Dates: {[str(d.date()) for d in dates]}")

    # ── 2 ──
    CACHE_TTL = 604_800  # seconds (7 days resolved market data is immutable)
    gamma_cache = os.path.join(args.snapshot_dir, "gamma_cache.json")
    clob_cache  = os.path.join(args.snapshot_dir, "clob_cache.json")

    def cache_fresh(path):
        return os.path.exists(path) and (time.time() - os.path.getmtime(path)) < CACHE_TTL

    if cache_fresh(gamma_cache):
        print(f"\n=== 2. Loading Gamma cache ({gamma_cache}) ===")
        with open(gamma_cache) as f:
            raw = orjson.loads(f.read())
    else:
        print("\n=== 2. Fetching resolved markets from Gamma API ===")
        raw = fetch_resolved_markets()
        with open(gamma_cache, "wb") as f:
            f.write(orjson.dumps(raw))
    rows = [extract_market_row(m) for m in raw]
    markets = pd.DataFrame([r for r in rows if r]).reset_index(drop=True)
    print(f"Binary resolved markets: {len(markets):,}")

    if cache_fresh(clob_cache):
        print(f"\n=== 2b. Loading CLOB cache ({clob_cache}) ===")
        with open(clob_cache) as f:
            clob_map = orjson.loads(f.read())
    else:
        print("\n=== 2b. Fetching CLOB token map ===")
        clob_map = fetch_clob_token_map()
        with open(clob_cache, "wb") as f:
            f.write(orjson.dumps(clob_map))
    print(f"CLOB markets mapped: {len(clob_map):,}")
    markets["yes_token_id"] = markets["condition_id"].map(clob_map)
    markets = markets.dropna(subset=["yes_token_id"])
    print(f"Markets with CLOB token ID: {len(markets):,}")

    # ── 3 ──
    print("\n=== 3. Joining ===")
    # join on CLOB yes_token_id (explicitly labeled by CLOB API) = PMXT asset_id
    joined = prices.merge(
        markets,
        left_on="asset_id",
        right_on="yes_token_id",
        how="inner",
    )
    joined = joined[joined["snapshot_date"] < joined["end_date"]].copy()
    joined["days_before"] = (joined["end_date"] - joined["snapshot_date"]).dt.days

    n_obs = len(joined)
    n_markets = joined["market_id"].nunique()
    print(f"Matched observations: {n_obs:,} | Unique markets: {n_markets:,}")

    if joined.empty:
        print("\nJoin produced no results. Debug sample keys:")
        print("  PMXT asset_id    :", prices["asset_id"].head(3).tolist())
        print("  CLOB yes_token_id:", markets["yes_token_id"].head(3).tolist())
        return

    joined.to_csv("calibration_raw.csv", index=False)
    print("Saved calibration_raw.csv")

    # filter by volume before selecting best snapshot
    joined = joined[joined["volume"] >= args.min_volume]
    print(f"After volume filter (>=${args.min_volume:,.0f}): {joined['condition_id'].nunique():,} markets")

    # for each market pick the snapshot closest to the target lookback
    joined["delta"] = (joined["days_before"] - args.lookback).abs()
    best = joined.sort_values("delta").groupby("condition_id").first().reset_index()
    print(f"Markets used for calibration (closest to {args.lookback}d): {len(best):,}")

    # deduplicate negRisk event groups: keep only the highest-priced market per event
    # (the current favorite). Without this, "Will X win NBA MVP?" × 22 players counts
    # as 22 observations from 1 event, massively inflating n and distorting the curve.
    n_before = len(best)
    best = best.sort_values("mid", ascending=False).groupby("event_id").first().reset_index()
    n_after = len(best)
    print(f"After event dedup (one market per event): {n_after:,}  (removed {n_before - n_after:,} correlated sub-markets)")
    print(f"  negRisk events: {best['neg_risk'].sum():,} of {n_after:,}")
    print(f"days_before distribution:\n{best['days_before'].describe().round(1).to_string()}")

    # ── 4 ──
    overall = calibration_stats(best)
    overall.to_csv("calibration_curve.csv", index=False)
    print("\nOverall calibration curve:")
    print(overall.to_string(index=False))

    top_cats = best["category"].value_counts().head(5).index.tolist()
    by_cat = {cat: calibration_stats(best[best["category"] == cat], min_obs=5) for cat in top_cats}

    best["vol_q"] = pd.qcut(
        best["volume"], q=4, labels=["Q1 low", "Q2", "Q3", "Q4 high"], duplicates="drop"
    )
    by_vol = {
        str(vq): calibration_stats(best[best["vol_q"] == vq], min_obs=5)
        for vq in best["vol_q"].cat.categories
    }

    # ── 5. Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    calibration_panel(axes[0], overall, f"Overall (≈{args.lookback}d before resolution, n={len(best):,})")

    axes[1].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Perfect")
    for cat, cal in by_cat.items():
        if cal.empty:
            continue
        axes[1].plot(cal["bucket_mid"], cal["empirical"], marker="o", ms=4, label=cat, alpha=0.8)
    axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("Implied probability"); axes[1].set_ylabel("Empirical rate")
    axes[1].set_title("By Category"); axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    axes[2].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Perfect")
    for vq, cal in by_vol.items():
        if cal.empty:
            continue
        axes[2].plot(cal["bucket_mid"], cal["empirical"], marker="o", ms=4, label=vq, alpha=0.8)
    axes[2].set_xlim(0, 1); axes[2].set_ylim(0, 1)
    axes[2].set_xlabel("Implied probability"); axes[2].set_ylabel("Empirical rate")
    axes[2].set_title("By Volume Quartile"); axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("calibration_curves.png", dpi=150, bbox_inches="tight")
    print("\nSaved calibration_curves.png")
    plt.show()


if __name__ == "__main__":
    main()
