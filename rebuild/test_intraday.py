#!/usr/bin/env python3
"""The intraday scorer must match score_asset, and the fear thresholds stay put."""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from build_dashboard import blend_external, score_asset  # noqa: E402
import intraday  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIN_HIST = 1260
TIERS = [
    {"key": "L1", "name": "观察", "max": 18, "ratio": 0.0},
    {"key": "L2", "name": "建仓", "max": 8, "ratio": 0.2},
    {"key": "L3", "name": "加码", "max": 5, "ratio": 0.3},
    {"key": "L4", "name": "重仓", "max": 2, "ratio": 0.5},
]


def fail(msg):
    print("FAIL", msg, file=sys.stderr)
    sys.exit(1)


def assert_close(got, exp, tol, label):
    if abs(got - exp) > tol:
        fail(f"{label}: got {got} expected {exp}")


def walk(n=1700, seed=7):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.02, n)
    px = 100 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2018-01-01", periods=n, freq="D")
    return pd.Series(px, index=idx)


def official_fp(close, price):
    series = close.copy()
    day = close.index[-1] + pd.Timedelta(days=1)
    series.loc[day] = price
    return float(score_asset(series, MIN_HIST).iloc[-1])


def test_thresholds_unchanged():
    with open(os.path.join(ROOT, "rebuild", "config.yaml"), encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    got = [
        {k: t[k] for k in ("key", "name", "max", "ratio", "consecutive_days")}
        for t in cfg["tiers"]
    ]
    expect = [
        {"key": "L1", "name": "观察", "max": 18, "ratio": 0.0, "consecutive_days": 2},
        {"key": "L2", "name": "建仓", "max": 8, "ratio": 0.2, "consecutive_days": 3},
        {"key": "L3", "name": "加码", "max": 5, "ratio": 0.3, "consecutive_days": 4},
        {"key": "L4", "name": "重仓", "max": 2, "ratio": 0.5, "consecutive_days": 3},
    ]
    if got != expect:
        fail(f"tiers changed: {got}")
    # The page's badge, and the intraday warning, use the same max comparison.
    cases = [(18, "L1"), (18.001, None), (8, "L2"), (8.001, "L1"), (5, "L3"), (5.001, "L2"), (2, "L4"), (2.001, "L3"), (0, "L4")]
    for value, key in cases:
        got_key = intraday.zone_key(value, TIERS)
        if got_key != key:
            fail(f"zone {value} -> {got_key}, expected {key}")


def test_score_matches_official_method():
    close = walk()
    history = close.iloc[:-1]
    model = intraday.build_model(history, MIN_HIST, intraday.ymd(history.index[-1]))
    prices = [
        float(history.iloc[-1]),
        float(history.iloc[-1]) * 0.92,
        float(history.iloc[-1]) * 0.8,
        float(history.iloc[-1]) * 1.05,
        float(close.iloc[-1]),
    ]
    cases = []
    for price in prices:
        got = intraday.score_from_model(model, price)
        exp = official_fp(history, price)
        assert_close(got, exp, 1e-8, f"score {price:.4f}")
        cases.append({"price": price, "fp": got})
    # Completing one more close, then scoring the next one, stays on the official path.
    folded = intraday.fold_model(model, float(close.iloc[-1]), intraday.ymd(close.index[-1]))
    nxt = float(close.iloc[-1]) * 0.85
    assert_close(intraday.score_from_model(folded, nxt), official_fp(close, nxt), 1e-8, "fold")
    # JSON must not move the score. The page reads these floats back.
    restored = json.loads(json.dumps(model))
    assert_close(
        intraday.score_from_model(restored, prices[1]),
        intraday.score_from_model(model, prices[1]),
        1e-9,
        "json roundtrip",
    )
    blended = intraday.blend(cases[1]["fp"], 0.30, 12)
    # Same blend the dashboard applies after the internal percentile.
    ext = pd.Series([12.0], index=[history.index[-1] + pd.Timedelta(days=1)])
    series = history.copy()
    day = history.index[-1] + pd.Timedelta(days=1)
    series.loc[day] = prices[1]
    fp = score_asset(series, MIN_HIST)
    exp_blend = float(blend_external(fp, ext, 0.30).iloc[-1])
    assert_close(blended, exp_blend, 1e-8, "blend")
    return model, cases, {
        "price": float(close.iloc[-1]),
        "day": intraday.ymd(close.index[-1]),
        "then": nxt,
        "expect": intraday.score_from_model(folded, nxt),
    }


def test_block_does_not_replace_official():
    close = walk(seed=11)
    history = close.iloc[:-1]
    model = intraday.build_model(history, MIN_HIST, "2026-11-20")
    official = {"date": "2026-11-20", "price": 59731.4805, "value": 10.428, "tier": "L1"}
    session = {"date": "2026-11-21", "low": float(history.iloc[-1]) * 0.7, "live": float(history.iloc[-1]), "open": True}
    block = intraday.build_block(
        model, session, {"value": 12, "date": "2026-11-21", "carried": False},
        TIERS, official, "2026-11-21T12:00:00", 0.30,
        {"history": "test", "session": "test", "gaps_filled": []},
    )
    bad = intraday.problems(block, TIERS)
    if bad:
        fail("; ".join(bad))
    if block["replaces_official"] or block["counts_consecutive_days"]:
        fail("block claims official authority")
    if block["official_value"] != 10.428 or block["official_tier"] != "L1":
        fail("official fields were rewritten")
    if block["symbol"] != "BTC-USD":
        fail("warning is not BTC-only")
    # Active follows the published thresholds only. L1 is not a warning.
    if block["active"] != (block["zone"] in ("L2", "L3", "L4")):
        fail("active flag does not match L2+")
    if not (block["value"] <= block["live_value"]):
        fail("basis should be the more fearful of the low and the live price")
    forged = dict(block)
    forged["replaces_official"] = True
    if not intraday.problems(forged, TIERS):
        fail("a block that replaces the official tier was accepted")


def test_javascript_matches(model, cases, fold):
    node = os.environ.get("NODE", "node")
    fixture = {"model": model, "cases": cases, "fold": fold}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(fixture, handle)
        path = handle.name
    script = os.path.join(os.path.dirname(__file__), "check_intraday.js")
    try:
        out = subprocess.check_output([node, script, path], stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as e:
        fail(e.output)
    finally:
        os.unlink(path)
    if "ok" not in out:
        fail(out)


def main():
    test_thresholds_unchanged()
    model, cases, fold = test_score_matches_official_method()
    test_block_does_not_replace_official()
    test_javascript_matches(model, cases, fold)
    print("ok")


if __name__ == "__main__":
    main()
