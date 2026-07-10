#!/usr/bin/env python3
"""
Realistic backtest: buy NO on 97.5%+ markets at ~7d before resolution.

Entry price = 1 - YES_bid  (taker buys NO at the ask, which = 1 - YES_bid)
Fee model   = 2% of gross payout (Polymarket taker fee)
Position    = $100 flat per trade (to show dollar PnL and slippage context)
"""

import glob
import os
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import orjson
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = "."
PRICE_BINS = np.arange(0, 1.05, 0.05)
POLYMARKET_FEE = 0.02   # 2% of gross payout on winning side
POSITION_SIZE  = 100.0  # dollars per trade

# ── 1. Load calibration_raw + event dedup ─────────────────────────────────────

print("=== Loading calibration data ===")
raw = pd.read_csv(os.path.join(DATA_DIR, "calibration_raw.csv"))
raw["snapshot_date"] = pd.to_datetime(raw["snapshot_date"], format="mixed").dt.tz_localize(None)
raw["end_date"]      = pd.to_datetime(raw["end_date"],      format="mixed").dt.tz_localize(None)
raw["days_before"]   = (raw["end_date"] - raw["snapshot_date"]).dt.days
raw["delta"]         = (raw["days_before"] - 7).abs()

# volume filter + best snapshot per market
raw = raw[raw["volume"] >= 50_000]
best = raw.sort_values("delta").groupby("condition_id").first().reset_index()

# event dedup: keep highest-mid market per event
best = best.sort_values("mid", ascending=False).groupby("event_id").first().reset_index()
print(f"  Independent events after dedup: {len(best):,}")
print(f"  days_before: median={best['days_before'].median():.0f}, mean={best['days_before'].mean():.1f}")

# ── 2. Identify trades: mid >= 0.90 (buy NO) ──────────────────────────────────

trades = best[best["mid"] >= 0.90].copy()
print(f"\n  Markets with mid >= 90%: {len(trades):,}")
print(f"  Bucket breakdown:")
trades["bucket"] = pd.cut(trades["mid"], bins=[0.90,0.925,0.95,0.975,1.0],
                           labels=["90-92.5%","92.5-95%","95-97.5%","97.5-100%"],
                           include_lowest=True)
print(trades.groupby("bucket", observed=True).agg(
    n=("yes_won","count"), yes_rate=("yes_won","mean")
).to_string())

# ── 3. Pull actual bid/ask from raw parquet for each trade ────────────────────
# Group by snapshot_date so each parquet is read once for all trades on that date.

print("\n=== Pulling actual bid/ask from parquet snapshots ===")

parquet_files = glob.glob(os.path.join(DATA_DIR, "polymarket_orderbook_*.parquet"))
date_to_file  = {}
for f in parquet_files:
    stem = Path(f).stem
    dt   = pd.Timestamp(stem.split("polymarket_orderbook_")[1]).normalize()
    date_to_file[dt] = f

# build lookup: condition_id -> asset_id (as str)
cid_to_token = dict(zip(trades["condition_id"], trades["asset_id"].astype(str)))
# results accumulate here
bid_ask_results: dict[str, tuple] = {}  # condition_id -> (yes_bid, yes_ask)

for snap_date, group in trades.groupby("snapshot_date"):
    pq_path = date_to_file.get(snap_date)
    if pq_path is None:
        print(f"  {snap_date.date()}: no parquet found")
        continue

    token_ids = set(group["asset_id"].astype(str).tolist())
    cid_for_token = {str(r["asset_id"]): r["condition_id"] for _, r in group.iterrows()}

    print(f"  {snap_date.date()}: reading {Path(pq_path).name} for {len(token_ids)} tokens ...", end=" ", flush=True)

    pf = pq.ParquetFile(pq_path)
    found: dict[str, dict] = {}  # token_id -> {bid, ask}

    for batch in pf.iter_batches(batch_size=500_000):
        df = batch.to_pandas()
        df["_tok"] = df["asset_id"].astype(str)
        sub = df[df["_tok"].isin(token_ids)]
        if sub.empty:
            continue
        sub = sub.sort_values("timestamp")
        for tok, grp in sub.groupby("_tok"):
            last = grp.iloc[-1]
            bid = pd.to_numeric(last.get("best_bid"), errors="coerce")
            ask = pd.to_numeric(last.get("best_ask"), errors="coerce")
            found[tok] = {"bid": float(bid) if pd.notna(bid) else np.nan,
                          "ask": float(ask) if pd.notna(ask) else np.nan}
        if set(found.keys()) >= token_ids:
            break  # all tokens found

    for tok, cid in cid_for_token.items():
        info = found.get(tok, {})
        bid_ask_results[cid] = (info.get("bid", np.nan), info.get("ask", np.nan))

    print(f"found {len(found)}/{len(token_ids)}")

spreads_df = pd.DataFrame([
    {"condition_id": cid, "yes_bid": v[0], "yes_ask": v[1]}
    for cid, v in bid_ask_results.items()
])
trades = trades.merge(spreads_df, on="condition_id", how="left")

# ── 4. Compute realistic entry prices ─────────────────────────────────────────
#
# In a binary market: YES + NO = $1
# To BUY NO as a taker:   pay (1 - YES_bid)  [you lift the YES bid = sell YES = buy NO at ask]
# Polymarket fee on WIN:  2% of gross payout ($1 per token)
# Net payout if NO wins:  $1 * (1 - fee) = $0.98 per token
#
# If yes_bid is 0 or missing, we fall back to (1 - yes_ask) as a pessimistic estimate,
# or flag as untradeable.

def no_ask_price(yes_bid, yes_ask, mid):
    """Cost to buy $1 of NO payout (taker order)."""
    if pd.notna(yes_bid) and yes_bid > 0:
        return 1.0 - yes_bid          # buy NO = sell YES at YES bid
    elif pd.notna(yes_ask) and yes_ask > 0 and yes_ask < 1.0:
        return 1.0 - yes_ask          # worst case: NO ask from YES ask
    else:
        return 1.0 - mid              # fallback: use mid (optimistic)

trades["no_ask"]     = trades.apply(lambda r: no_ask_price(r["yes_bid"], r["yes_ask"], r["mid"]), axis=1)
trades["mid_no"]     = 1.0 - trades["mid"]            # naive mid-price NO cost
trades["spread_pct"] = (trades["no_ask"] - trades["mid_no"]) / trades["mid_no"] * 100
trades["tradeable"]  = trades["no_ask"] < 0.50        # sanity filter: don't trade if NO > 50¢

print(f"\n=== Spread analysis ===")
print(f"  Markets where bid was found: {trades['yes_bid'].notna().sum()} / {len(trades)}")
print(f"  NO ask prices (what you actually pay):")
print(trades[["question","mid","mid_no","no_ask","spread_pct","yes_bid","yes_ask"]].to_string(index=False))
print(f"\n  Mean NO mid: {trades['mid_no'].mean():.4f}")
print(f"  Mean NO ask (entry): {trades['no_ask'].mean():.4f}")
print(f"  Mean spread over mid: {trades['spread_pct'].mean():.1f}%")

# ── 5. Backtest at realistic prices ───────────────────────────────────────────

print(f"\n=== Backtest (position = ${POSITION_SIZE:.0f}/trade, fee = {POLYMARKET_FEE*100:.0f}% of payout) ===")

def run_backtest(df, entry_col, label):
    rows = []
    for _, t in df.iterrows():
        entry = t[entry_col]
        if entry <= 0 or entry >= 1:
            continue
        tokens      = POSITION_SIZE / entry           # tokens bought
        yes_won     = bool(t["yes_won"])
        if yes_won:
            pnl = -POSITION_SIZE                      # NO loses: lose full stake
        else:
            gross   = tokens * 1.0                    # $1 per token
            fee     = gross * POLYMARKET_FEE
            pnl     = gross - fee - POSITION_SIZE     # profit after fee and cost

        rows.append({
            "condition_id": t["condition_id"],
            "question":     t["question"][:60],
            "entry":        entry,
            "tokens":       tokens,
            "yes_won":      yes_won,
            "pnl":          pnl,
            "bucket":       t.get("bucket", "?"),
        })
    df_out = pd.DataFrame(rows)
    if df_out.empty:
        print(f"  [{label}] No trades.")
        return df_out

    wins = (~df_out["yes_won"]).sum()
    n    = len(df_out)
    total_invested = n * POSITION_SIZE
    total_pnl      = df_out["pnl"].sum()
    roi            = total_pnl / total_invested * 100

    print(f"\n  [{label}]")
    print(f"    Trades: {n}  |  NO wins: {wins} ({wins/n:.1%})")
    print(f"    Total invested: ${total_invested:,.0f}")
    print(f"    Total PnL:      ${total_pnl:,.2f}")
    print(f"    ROI:            {roi:.1f}%")
    print(f"    Mean PnL/trade: ${df_out['pnl'].mean():,.2f}")
    print(f"    Median PnL/trade: ${df_out['pnl'].median():,.2f}")
    print(f"    Best trade:  ${df_out['pnl'].max():,.2f}  ({df_out.loc[df_out['pnl'].idxmax(),'question']})")
    print(f"    Worst trade: ${df_out['pnl'].min():,.2f}  ({df_out.loc[df_out['pnl'].idxmin(),'question']})")

    # by bucket
    print(f"\n    By bucket:")
    bkt = df_out.groupby("bucket", observed=True).agg(
        n=("pnl","count"),
        no_win_rate=("yes_won", lambda x: (~x.astype(bool)).mean()),
        total_pnl=("pnl","sum"),
        mean_pnl=("pnl","mean"),
    )
    print(bkt.to_string())
    return df_out

# Run at three price assumptions
bt_mid      = run_backtest(trades, "mid_no",  "Naive mid-price (optimistic)")
bt_realistic = run_backtest(trades[trades["tradeable"]], "no_ask", "Realistic ask price (taker)")

# ── 6. Break-even analysis ────────────────────────────────────────────────────

print(f"\n=== Break-even NO win rate by entry price ===")
print(f"{'Entry (NO ask)':>16} {'Breakeven NO win%':>18} {'Observed NO win%':>18} {'EV/dollar':>12}")
for entry in [0.01, 0.02, 0.025, 0.03, 0.04, 0.05, 0.07, 0.10]:
    observed_no_win = 1 - trades["yes_won"].mean()   # overall for the bucket
    breakeven = entry / (1 - POLYMARKET_FEE)         # need this win rate to break even
    ev = observed_no_win * (1 - POLYMARKET_FEE) / entry - 1
    marker = " ◄ " if abs(entry - trades["no_ask"].mean()) < 0.005 else ""
    print(f"  {entry:>14.3f}¢    {breakeven:>16.1%}    {observed_no_win:>16.1%}    {ev:>10.3f}{marker}")

# ── 7. Plots ──────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1: PnL distribution
ax = axes[0]
if not bt_realistic.empty:
    colors = ["green" if p > 0 else "red" for p in bt_realistic["pnl"]]
    ax.bar(range(len(bt_realistic)), sorted(bt_realistic["pnl"]), color=sorted(colors, reverse=True), alpha=0.7)
    ax.axhline(0, color="k", lw=1)
    ax.axhline(bt_realistic["pnl"].mean(), color="blue", lw=1.5, linestyle="--",
               label=f"Mean ${bt_realistic['pnl'].mean():.0f}")
    ax.set_xlabel("Trade (sorted by PnL)")
    ax.set_ylabel("PnL ($)")
    ax.set_title(f"Trade PnL — Realistic Entry\n(${POSITION_SIZE:.0f}/trade, {POLYMARKET_FEE*100:.0f}% fee)")
    ax.legend()
    ax.grid(True, alpha=0.3)

# Panel 2: NO ask vs mid for each trade
ax = axes[1]
scatter_data = trades[trades["yes_bid"].notna() & (trades["yes_bid"] > 0)]
if not scatter_data.empty:
    colors2 = ["green" if not yw else "red" for yw in scatter_data["yes_won"]]
    ax.scatter(scatter_data["mid"], scatter_data["no_ask"], c=colors2, alpha=0.7, s=60)
    xs = np.linspace(0.90, 1.0, 100)
    ax.plot(xs, 1 - xs, "k--", lw=1, label="No spread (ask=mid)")
    ax.set_xlabel("YES mid price")
    ax.set_ylabel("NO ask (your entry price)")
    ax.set_title("Spread: NO ask vs implied mid\n(green=NO won, red=YES won)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Panel 3: cumulative PnL over trades
ax = axes[2]
if not bt_realistic.empty:
    # sort by entry price so we see effect of tightening spread filter
    bt_sorted = bt_realistic.sort_values("entry")
    cumulative = bt_sorted["pnl"].cumsum().values
    ax.plot(cumulative, lw=2, label="Cumulative PnL (sorted by entry price)")
    ax.axhline(0, color="k", lw=1)
    ax.fill_between(range(len(cumulative)), cumulative, 0,
                    where=cumulative > 0, alpha=0.3, color="green")
    ax.fill_between(range(len(cumulative)), cumulative, 0,
                    where=cumulative < 0, alpha=0.3, color="red")
    ax.set_xlabel("Trades (sorted cheapest NO first)")
    ax.set_ylabel("Cumulative PnL ($)")
    ax.set_title("Cumulative PnL\n(equal $100/trade)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle(
    f"Polymarket NO-token Backtest — 90%+ Implied Markets, ~7d Lookback\n"
    f"Apr 14–May 18 2026 | $50k+ volume | Event-deduped | n={len(trades)} trades",
    fontsize=12
)
plt.tight_layout()
outpath = os.path.join(DATA_DIR, "backtest.png")
plt.savefig(outpath, dpi=150, bbox_inches="tight")
print(f"\nSaved {outpath}")
