#!/usr/bin/env python3
"""
Cross-platform + within-platform arbitrage monitor.

Polls Polymarket (Gamma API, no auth) and Kalshi (RSA-PSS signed requests)
on a fixed interval and looks for three kinds of structural mispricing:

  1. MECE deviation within Polymarket  — an event's markets should sum their
     YES prices to $1.00; a meaningful deviation is a guaranteed-profit basket.
  2. MECE deviation within Kalshi      — same idea, Kalshi side.
  3. Cross-platform divergence         — the same real-world event priced
     differently on Polymarket vs Kalshi (matched by fuzzy title similarity,
     since the two platforms don't share IDs).

Every cycle's raw findings are appended to arb_log.jsonl. A running
persistence counter (how many consecutive cycles each finding has shown up)
is maintained in memory and flushed to arb_summary.csv at the end, since a
mispricing that only appears once is almost always noise, not a real edge.

Usage:
  pip install pandas requests cryptography
  export KALSHI_KEY_ID=...
  export KALSHI_PRIVATE_KEY_PATH=/path/to/kalshi_private_key.pem
  python arb_monitor.py [--cycles 6] [--interval-min 10] [--min-volume 5000]
"""

import argparse
import base64
import difflib
import json
import os
import sys
import time
from datetime import datetime, timezone
from time import sleep

import requests
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.serialization import load_pem_private_key

GAMMA_BASE = "https://gamma-api.polymarket.com"
KALSHI_BASE = "https://api.elections.kalshi.com"

MECE_DEVIATION_THRESHOLD = 0.03     # sum of YES prices must be off from 1.00 by more than this
CROSS_PLATFORM_THRESHOLD = 0.04     # abs price difference on matched events
FUZZY_MATCH_THRESHOLD = 0.62        # difflib ratio to consider two titles "the same event"
ASSUMED_ROUND_TRIP_FEE = 0.035      # rough combined taker-fee estimate, cross-platform trades


# ── Kalshi auth ────────────────────────────────────────────────────────────────

def load_kalshi_private_key(path: str):
    with open(path, "rb") as f:
        return load_pem_private_key(f.read(), password=None)


def kalshi_signed_headers(method: str, path: str, key_id: str, private_key) -> dict:
    timestamp_ms = str(int(time.time() * 1000))
    message = (timestamp_ms + method.upper() + path).encode("utf-8")
    signature = private_key.sign(
        message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("utf-8"),
        "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
    }


def kalshi_get(path: str, params: dict, key_id: str, private_key) -> dict:
    headers = kalshi_signed_headers("GET", path, key_id, private_key)
    r = requests.get(KALSHI_BASE + path, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


# ── Polymarket fetch ───────────────────────────────────────────────────────────

def fetch_polymarket_events(min_volume: float = 0.0) -> list[dict]:
    """Active (unresolved) events, each with its nested markets list."""
    events, limit, offset = [], 500, 0
    effective_limit = limit  # Gamma silently caps page size below `limit`; detect it from page 1
    while True:
        r = requests.get(
            f"{GAMMA_BASE}/events",
            params={"closed": "false", "limit": limit, "offset": offset,
                     "order": "volume", "ascending": "false"},
            timeout=30,
        )
        if r.status_code == 422:
            break
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        events.extend(batch)
        if offset == 0 and len(batch) < limit:
            effective_limit = len(batch)
        if len(batch) < effective_limit:
            break
        offset += len(batch)
        sleep(0.05)
    if min_volume:
        events = [e for e in events if float(e.get("volume") or 0) >= min_volume]
    return events


def _parse_json_field(raw):
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return []
    return raw or []


def polymarket_event_legs(event: dict) -> list[tuple[str, float]]:
    """
    Return [(question, yes_price), ...] for each binary market in the event,
    restricted to markets Polymarket itself flags `negRisk` (i.e. the platform
    guarantees this group is mutually exclusive). Without this filter, MECE
    detection floods on groups that merely share an event tag but aren't
    actually exclusive — player-prop markets (multiple players can score),
    "which states will X visit" (multiple can happen), etc. — where the YES
    prices summing above 1.0 is expected, not a mispricing.
    """
    legs = []
    for m in event.get("markets") or []:
        if not m.get("negRisk"):
            continue
        outcomes = _parse_json_field(m.get("outcomes"))
        prices_raw = _parse_json_field(m.get("outcomePrices"))
        try:
            prices = [float(x) for x in prices_raw]
        except (TypeError, ValueError):
            continue
        ol = [str(o).lower() for o in outcomes]
        if "yes" not in ol or len(prices) != len(ol):
            continue
        legs.append((str(m.get("question", ""))[:120], prices[ol.index("yes")]))
    return legs


# ── Kalshi fetch ───────────────────────────────────────────────────────────────

def fetch_kalshi_events(key_id: str, private_key) -> list[dict]:
    events, cursor, path = [], None, "/trade-api/v2/events"
    while True:
        params = {"status": "open", "with_nested_markets": "true", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        data = kalshi_get(path, params, key_id, private_key)
        batch = data.get("events", [])
        events.extend(batch)
        cursor = data.get("cursor")
        if not cursor or not batch:
            break
        sleep(0.05)
    return events


def kalshi_event_legs(event: dict) -> list[tuple[str, float]]:
    legs = []
    for m in event.get("markets") or []:
        bid, ask = m.get("yes_bid"), m.get("yes_ask")
        if bid is None or ask is None:
            continue
        mid = (float(bid) + float(ask)) / 2.0 / 100.0   # Kalshi quotes in cents
        legs.append((str(m.get("title") or m.get("ticker") or "")[:120], mid))
    return legs


# ── MECE detection ─────────────────────────────────────────────────────────────

def find_mece_deviations(event_id: str, event_title: str, legs: list[tuple[str, float]], platform: str) -> dict | None:
    if len(legs) < 2:
        return None
    total = sum(p for _, p in legs)
    deviation = total - 1.0
    if abs(deviation) < MECE_DEVIATION_THRESHOLD:
        return None
    return {
        "finding_type": f"mece_{platform}",
        "platform": platform,
        "event_id": event_id,
        "question": event_title,
        "sample_legs": " | ".join(q for q, _ in legs[:5]),
        "n_legs": len(legs),
        "price_sum": round(total, 4),
        "deviation": round(deviation, 4),
        "net_edge_after_fees": round(abs(deviation) - ASSUMED_ROUND_TRIP_FEE, 4),
    }


# ── Cross-platform matching ────────────────────────────────────────────────────

def match_cross_platform(poly_events: list[dict], kalshi_events: list[dict]) -> list[dict]:
    findings = []
    kalshi_titled = [(e, str(e.get("title") or "")) for e in kalshi_events if e.get("title")]

    for pe in poly_events:
        p_title = str(pe.get("title") or "")
        if not p_title:
            continue
        p_legs = polymarket_event_legs(pe)
        if not p_legs:
            continue
        p_lead_q, p_lead_price = max(p_legs, key=lambda x: x[1])

        best_match, best_score = None, 0.0
        for ke, k_title in kalshi_titled:
            score = difflib.SequenceMatcher(None, p_title.lower(), k_title.lower()).ratio()
            if score > best_score:
                best_score, best_match = score, ke
        if best_match is None or best_score < FUZZY_MATCH_THRESHOLD:
            continue

        k_legs = kalshi_event_legs(best_match)
        if not k_legs:
            continue
        k_lead_q, k_lead_price = max(k_legs, key=lambda x: x[1])

        divergence = p_lead_price - k_lead_price
        if abs(divergence) < CROSS_PLATFORM_THRESHOLD:
            continue

        findings.append({
            "finding_type": "cross_platform",
            "platform": "polymarket+kalshi",
            "event_id": pe.get("id") or p_title,
            "question": f"{p_title}  [match_score={best_score:.2f}]",
            "sample_legs": f"{p_lead_q} | {k_lead_q}",
            "n_legs": None,
            "price_sum": None,
            "deviation": round(divergence, 4),
            "net_edge_after_fees": round(abs(divergence) - ASSUMED_ROUND_TRIP_FEE, 4),
        })
    return findings


# ── main loop ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=6, help="Number of polling cycles")
    parser.add_argument("--interval-min", type=float, default=10.0, help="Minutes between cycles")
    parser.add_argument("--min-volume", type=float, default=0.0, help="Skip Polymarket events below this volume")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--skip-kalshi", action="store_true", help="Polymarket-only run (no Kalshi credentials)")
    args = parser.parse_args()

    key_id = os.environ.get("KALSHI_KEY_ID")
    key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH")
    private_key = None
    if not args.skip_kalshi:
        if not key_id or not key_path:
            sys.exit("Set KALSHI_KEY_ID and KALSHI_PRIVATE_KEY_PATH, or pass --skip-kalshi")
        private_key = load_kalshi_private_key(key_path)

    log_path = os.path.join(args.output_dir, "arb_log.jsonl")
    summary_path = os.path.join(args.output_dir, "arb_summary.csv")

    # key = (finding_type, event_id) -> running record
    persistence: dict[tuple, dict] = {}

    with open(log_path, "a") as log_f:
        for cycle in range(args.cycles):
            now = datetime.now(timezone.utc).isoformat()
            print(f"\n=== Cycle {cycle + 1}/{args.cycles} — {now} ===")

            print("  Fetching Polymarket events...", end=" ", flush=True)
            poly_events = fetch_polymarket_events(min_volume=args.min_volume)
            print(f"{len(poly_events):,} events")

            kalshi_events = []
            if not args.skip_kalshi:
                print("  Fetching Kalshi events...", end=" ", flush=True)
                kalshi_events = fetch_kalshi_events(key_id, private_key)
                print(f"{len(kalshi_events):,} events")

            cycle_findings = []

            for pe in poly_events:
                legs = polymarket_event_legs(pe)
                hit = find_mece_deviations(pe.get("id") or pe.get("title", ""), str(pe.get("title", ""))[:120], legs, "polymarket")
                if hit:
                    cycle_findings.append(hit)

            for ke in kalshi_events:
                legs = kalshi_event_legs(ke)
                hit = find_mece_deviations(ke.get("event_ticker") or ke.get("title", ""), str(ke.get("title", ""))[:120], legs, "kalshi")
                if hit:
                    cycle_findings.append(hit)

            if not args.skip_kalshi:
                cycle_findings.extend(match_cross_platform(poly_events, kalshi_events))

            print(f"  Findings this cycle: {len(cycle_findings)}")

            for f in cycle_findings:
                record = dict(f, timestamp=now, cycle=cycle)
                log_f.write(json.dumps(record) + "\n")

                key = (f["finding_type"], f["event_id"])
                if key not in persistence:
                    persistence[key] = dict(f, first_seen=now, last_seen=now, seen_cycles=1)
                else:
                    persistence[key].update(f)
                    persistence[key]["last_seen"] = now
                    persistence[key]["seen_cycles"] += 1
            log_f.flush()

            if cycle < args.cycles - 1:
                sleep(args.interval_min * 60)

    print(f"\nSaved raw log -> {log_path}")

    import pandas as pd
    if persistence:
        summary_df = pd.DataFrame(list(persistence.values()))
        summary_df = summary_df.sort_values(["seen_cycles", "net_edge_after_fees"], ascending=[False, False])
    else:
        summary_df = pd.DataFrame(columns=[
            "finding_type", "platform", "event_id", "question", "sample_legs", "n_legs",
            "price_sum", "deviation", "net_edge_after_fees", "first_seen", "last_seen", "seen_cycles",
        ])
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary ({len(summary_df)} unique findings) -> {summary_path}")


if __name__ == "__main__":
    main()
