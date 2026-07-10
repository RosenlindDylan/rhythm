#!/usr/bin/env python3
"""
Post-process arb_monitor.py output.

arb_monitor.py's raw MECE-deviation detector is deliberately naive (sum of a
group's YES prices vs 1.00) and picks up a lot of markets that *look*
mutually-exclusive-and-exhaustive but structurally aren't: crypto
price-threshold brackets (multiple thresholds can all resolve YES), exact
score markets, "top N" finish markets, and esports bracket-progression
markets. This script filters those known false-positive structures out,
then ranks what's left by how many consecutive cycles it persisted and its
estimated edge after fees — a deviation seen once is noise, a deviation
still there after an hour of polling is a real candidate.

Usage:
  python analyze_results.py [--summary arb_summary.csv] [--min-cycles 3]
"""

import argparse
import re

import pandas as pd

# (regex, label) — any question matching one of these is a structurally
# unsound MECE candidate, not a real arbitrage.
FALSE_POSITIVE_PATTERNS = [
    (r"\$[\d,]+[kKmM]?\b.{0,20}\b(reach|hit|exceed|above|below|over|under)\b", "price threshold bracket"),
    (r"\btop[\s-]?\d+\b", "top-N finish"),
    (r"\b\d{1,3}\s*[-–]\s*\d{1,3}\b", "exact score"),
    (r"\b(round of|advance to|make the|reach the (?:semis|finals|final))\b", "bracket progression"),
    (r"\bover\s*/\s*under\b|\bo/u\b", "over/under line"),
]


def false_positive_reason(question: str, sample_legs: str = "") -> str | None:
    # the price/score/round keywords that give away a structurally-unsound MECE
    # group usually live in the *sub-market* text, not the event-level title
    text = f"{question or ''} {sample_legs or ''}".lower()
    for pattern, label in FALSE_POSITIVE_PATTERNS:
        if re.search(pattern, text):
            return label
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="arb_summary.csv")
    parser.add_argument("--min-cycles", type=int, default=3, help="Minimum persistence to keep a finding")
    parser.add_argument("--out", default="arb_interesting.csv")
    args = parser.parse_args()

    print(f"=== Loading {args.summary} ===")
    df = pd.read_csv(args.summary)
    print(f"  {len(df):,} raw findings")

    if df.empty:
        print("  Nothing to analyze.")
        pd.DataFrame().to_csv(args.out, index=False)
        return

    # ── filter known false-positive market structures ──
    if "sample_legs" not in df.columns:
        df["sample_legs"] = ""
    df["false_positive_reason"] = df.apply(
        lambda r: false_positive_reason(r["question"], r.get("sample_legs")), axis=1
    )
    fp_counts = df["false_positive_reason"].value_counts()
    if not fp_counts.empty:
        print("\n  False-positive structures filtered:")
        print(fp_counts.to_string())

    filtered = df[df["false_positive_reason"].isna()].copy()
    print(f"\n  {len(df) - len(filtered):,} filtered out, {len(filtered):,} remaining")

    # ── require persistence ──
    filtered = filtered[filtered["seen_cycles"] >= args.min_cycles]
    print(f"  {len(filtered):,} remaining after requiring seen_cycles >= {args.min_cycles}")

    if filtered.empty:
        print("\nNo candidates survived filtering.")
        filtered.to_csv(args.out, index=False)
        return

    # ── one row per event: the best (most persistent, highest-edge) finding ──
    ranked = (
        filtered.sort_values(["seen_cycles", "net_edge_after_fees"], ascending=[False, False])
        .groupby("event_id", as_index=False)
        .first()
        .sort_values(["seen_cycles", "net_edge_after_fees"], ascending=[False, False])
    )

    print(f"\n=== Top candidates ({len(ranked)} unique events) ===")
    cols = [c for c in ["finding_type", "platform", "question", "deviation",
                        "net_edge_after_fees", "seen_cycles"] if c in ranked.columns]
    print(ranked[cols].head(20).to_string(index=False))

    ranked.to_csv(args.out, index=False)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
