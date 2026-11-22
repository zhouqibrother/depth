#!/usr/bin/env python3
"""Refresh only the Bitcoin intraday block. Official daily fields stay as they are.

  python rebuild/update_intraday.py --dashboard dashboard_data.json

One Yahoo daily-history request, one alternative.me request, and the Coinbase
daily candle (Kraken if Coinbase is down). The full nine-market rebuild is left
to update.yml. This script refuses to write if any field other than
btc_intraday would change.
"""
import argparse
import json
import os
import sys
import tempfile
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
import intraday  # noqa: E402


def atomic_json(path, obj):
    path = os.path.abspath(path)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=os.path.dirname(path) or ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(obj, handle, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def without(doc):
    return {k: v for k, v in doc.items() if k != "btc_intraday"}


def load_fng(previous):
    try:
        return intraday.fetch_fng(), "alternative.me"
    except Exception as e:
        intraday.log(f"! alternative.me: {e}")
    if previous and previous.get("fng_value") is not None and previous.get("fng_date"):
        import pandas as pd

        intraday.log("  alternative.me: reusing the previous print")
        series = pd.Series(
            [float(previous["fng_value"])],
            index=[pd.Timestamp(previous["fng_date"])],
        )
        return series, "previous"
    return None, "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    ap.add_argument("--dashboard", default="dashboard_data.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        import yaml

        cfg = yaml.safe_load(handle)
    with open(args.dashboard, encoding="utf-8") as handle:
        dashboard = json.load(handle)
    if not isinstance(dashboard, dict):
        intraday.log("dashboard_data.json is not an object")
        sys.exit(1)
    close = intraday.fetch_yahoo_daily()
    previous = dashboard.get("btc_intraday") if isinstance(dashboard.get("btc_intraday"), dict) else None
    fng, fng_source = load_fng(previous)
    block = intraday.compute_block(
        close,
        fng,
        intraday.tiers_from(dashboard),
        intraday.official_from_payload(dashboard),
        intraday.blend_weight(cfg),
        cfg.get("min_history_days_for_scoring", 1260),
        datetime.now(intraday.CST),
    )
    block["sources"]["fng"] = fng_source
    bad = intraday.problems(block, intraday.tiers_from(dashboard))
    if bad:
        intraday.log("refusing to write intraday block")
        for item in bad:
            intraday.log(f"  - {item}")
        sys.exit(1)
    updated = dict(dashboard)
    updated["btc_intraday"] = block
    if without(updated) != without(dashboard):
        intraday.log("refusing to write: official dashboard fields would change")
        sys.exit(1)
    view = intraday.summary(block)
    intraday.log(json.dumps(view, ensure_ascii=False))
    if args.dry_run:
        intraday.log("dry run, dashboard not written")
        return
    atomic_json(args.dashboard, updated)
    intraday.log(f"updated {args.dashboard} btc_intraday only")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        intraday.log(f"! intraday update failed: {e}")
        sys.exit(1)
