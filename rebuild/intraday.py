#!/usr/bin/env python3
"""Bitcoin-only intraday fear estimate.

The official daily depth in dashboard_data.json is unchanged: it still scores
the daily close, including the alternative.me blend, and the L1–L4 thresholds
and consecutive-day state machine are not touched.

This module answers a different question: if today's UTC close were the session
low, or the live price, what would that same depth be? The result is published
as btc_intraday. It is a warning, not a new official tier. A single wick never
overwrites latest, tier, or the close history.

The estimate is the five-component expanding-percentile score in score_asset,
then the same alternative.me blend the dashboard already applies to BTC-USD.
When today's alternative.me print is missing, the latest print on or before the
session is carried forward so the number stays on the official scale. That
carry is labeled fng_carried and is not used for the official close.

History for the model is every completed UTC daily close before the session.
The open session is never inserted into that history, so replacing its close
does not create a second bar. Recent holes in the Yahoo daily series (a missing
close, not an ancient gap) may be filled from Coinbase after a scale check.
Yahoo's chart API with range=max currently returns monthly bars, so the daily
history is requested with explicit period1/period2.
"""
import json
import math
import os
import sys
import time
from bisect import bisect_left, bisect_right, insort
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
import yaml

CST = timezone(timedelta(hours=8))
UA = {"User-Agent": "Mozilla/5.0 (onebitking-dashboard)"}
DEEP_ZONES = ("L2", "L3", "L4")
YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/"
COINBASE_CANDLES = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
COINBASE_SPOT = "https://api.coinbase.com/v2/prices/BTC-USD/spot"
KRAKEN_OHLC = "https://api.kraken.com/0/public/OHLC"
FNG_URL = "https://api.alternative.me/fng/?limit=0&format=json"
NOTE = (
    "盘中估算：把这个价格当作今日收盘，用与官方相同的五分量展开百分位，"
    "并按同样的 alternative.me 权重混合。不是收盘确认档，不改连续天数，不改配额。"
)


def log(*args):
    print(*args, file=sys.stderr, flush=True)


def ymd(ts):
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def http_get(url, params=None, timeout=30):
    last = None
    for i in range(3):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers=UA)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            last = e
            log(f"  GET {url} try {i + 1}: {e}")
            if i < 2:
                time.sleep(1.5 * (i + 1))
    raise last


def pct_including(sorted_vals, x):
    """Percentile of x inside sorted_vals plus x itself. Ties count half.

    Same rule as expanding_pct in build_dashboard.py, applied to one new point
    so the open session does not have to be rescored from the first bar.
    """
    lo = bisect_left(sorted_vals, x)
    hi = bisect_right(sorted_vals, x)
    n = len(sorted_vals) + 1
    return (lo + 0.5 * ((hi + 1) - lo)) / n * 100.0


def _finite_sorted(series):
    vals = series.to_numpy(dtype=float)
    out = [float(x) for x in vals if np.isfinite(x)]
    out.sort()
    return out


def _welford(values):
    n = 0
    mean = 0.0
    m2 = 0.0
    for x in values:
        if not np.isfinite(x):
            continue
        n += 1
        delta = x - mean
        mean += delta / n
        m2 += delta * (x - mean)
    return n, mean, m2


def build_model(close, min_hist, history_through):
    """State needed to score one hypothetical next close.

    `close` is completed daily closes only, oldest to newest, one row per day.
    The distributions match score_asset: RSI is used as itself, the other four
    components are expanding percentiles, and the fear depth is the expanding
    percentile of their average.
    """
    c = close.astype(float).sort_index()
    c = c[~c.index.duplicated(keep="last")]
    if len(c) < max(252, int(min_hist)) or float(c.iloc[-1]) <= 0:
        raise RuntimeError(f"need at least {max(252, int(min_hist))} completed closes, have {len(c)}")
    n = 14
    diff = c.diff()
    up = diff.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-diff.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    if not np.isfinite(up.iloc[-1]) or not np.isfinite(dn.iloc[-1]):
        raise RuntimeError("RSI state is not finite")
    ratio = (c - c.rolling(200).mean()) / c.rolling(200).mean()
    exp_mean = ratio.expanding().mean()
    exp_std = ratio.expanding().std()
    z = (ratio - exp_mean) / exp_std
    dd = c / c.rolling(252, min_periods=252).max() - 1
    vol = c.pct_change().rolling(20).std()
    mom = c.pct_change(20)
    # Component A is RSI itself (already 0–100), matching score_asset.
    a = rsi(c).reindex(c.index)
    b = pd.Series(_expanding_pct_array(z.to_numpy(dtype=float)), index=c.index)
    cc = pd.Series(_expanding_pct_array(dd.to_numpy(dtype=float)), index=c.index)
    dv = 100.0 - pd.Series(_expanding_pct_array(vol.to_numpy(dtype=float)), index=c.index)
    e = pd.Series(_expanding_pct_array(mom.to_numpy(dtype=float)), index=c.index)
    raw = pd.concat([a, b, cc, dv, e], axis=1).mean(axis=1, skipna=False)
    ratio_n, ratio_mean, ratio_m2 = _welford(ratio.to_numpy(dtype=float))
    pandas_n = int(np.isfinite(ratio.to_numpy(dtype=float)).sum())
    if ratio_n != pandas_n:
        raise RuntimeError("expanding-ratio count does not match")
    model = {
        "history_through": history_through,
        "prev_close": float(c.iloc[-1]),
        "closes": [float(x) for x in c.iloc[-252:].tolist()],
        "rsi_up": float(up.iloc[-1]),
        "rsi_dn": float(dn.iloc[-1]),
        "ratio_n": int(ratio_n),
        "ratio_mean": float(ratio_mean),
        "ratio_m2": float(ratio_m2),
        "z": _finite_sorted(z),
        "dd": _finite_sorted(dd),
        "vol": _finite_sorted(vol),
        "mom": _finite_sorted(mom),
        "raw": _finite_sorted(raw),
        "min_history": int(min_hist),
    }
    if len(model["raw"]) + 1 < int(min_hist):
        raise RuntimeError("not enough scored days to leave the warmup window")
    return model


def rsi(close, n=14):
    """Wilder RSI. Same construction as build_dashboard.score_asset."""
    diff = close.diff()
    up = diff.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-diff.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn)


def _expanding_pct_array(values):
    """Same walk as build_dashboard.expanding_pct, local copy so this module stands alone."""
    seen, out = [], np.full(len(values), np.nan)
    for i, x in enumerate(values):
        if x is None or not np.isfinite(x):
            continue
        insort(seen, float(x))
        n = len(seen)
        lo, hi = bisect_left(seen, float(x)), bisect_right(seen, float(x))
        out[i] = (lo + 0.5 * (hi - lo)) / n * 100.0
    return out


def _components(model, price):
    price = float(price)
    if not math.isfinite(price) or price <= 0:
        raise RuntimeError("price must be a positive finite number")
    closes = model["closes"]
    if len(closes) < 252:
        raise RuntimeError("model is missing the 252-day window")
    alpha = 1.0 / 14.0
    delta = price - float(model["prev_close"])
    gain = delta if delta > 0 else 0.0
    loss = -delta if delta < 0 else 0.0
    rsi_up = (1.0 - alpha) * float(model["rsi_up"]) + alpha * gain
    rsi_dn = (1.0 - alpha) * float(model["rsi_dn"]) + alpha * loss
    rsi = 100.0 if rsi_dn == 0 else 100.0 - 100.0 / (1.0 + rsi_up / rsi_dn)
    ma = (sum(closes[-(200 - 1):]) + price) / 200.0
    ratio = (price - ma) / ma
    n = int(model["ratio_n"]) + 1
    mean = float(model["ratio_mean"]) + (ratio - float(model["ratio_mean"])) / n
    dlt = ratio - float(model["ratio_mean"])
    m2 = float(model["ratio_m2"]) + dlt * (ratio - mean)
    std = math.sqrt(m2 / (n - 1))
    if not math.isfinite(std) or std == 0:
        raise RuntimeError("MA200 z-score is not finite")
    z = (ratio - mean) / std
    dd = price / max(closes[-(252 - 1):] + [price]) - 1.0
    seq = list(closes[-20:]) + [price]
    rets = [seq[i] / seq[i - 1] - 1.0 for i in range(1, len(seq))]
    mu = sum(rets) / len(rets)
    var = sum((x - mu) ** 2 for x in rets) / (len(rets) - 1)
    vol = math.sqrt(var)
    mom = price / closes[-20] - 1.0
    zp = pct_including(model["z"], z)
    dp = pct_including(model["dd"], dd)
    vp = 100.0 - pct_including(model["vol"], vol)
    mp = pct_including(model["mom"], mom)
    raw = (rsi + zp + dp + vp + mp) / 5.0
    if len(model["raw"]) + 1 < int(model["min_history"]):
        raise RuntimeError("hypothetical close is still inside the warmup window")
    fp = pct_including(model["raw"], raw)
    return {
        "fp": fp,
        "rsi": rsi,
        "rsi_up": rsi_up,
        "rsi_dn": rsi_dn,
        "ratio": ratio,
        "z": z,
        "dd": dd,
        "vol": vol,
        "mom": mom,
        "raw": raw,
    }


def score_from_model(model, price):
    """Unblended fear percentile if `price` were the next close. Low means fear."""
    return _components(model, price)["fp"]


def fold_model(model, price, day):
    """Treat `price` as a completed close on `day` and return a new model."""
    comp = _components(model, price)
    out = dict(model)
    out["history_through"] = day
    out["prev_close"] = float(price)
    out["closes"] = (list(model["closes"]) + [float(price)])[-252:]
    out["rsi_up"] = comp["rsi_up"]
    out["rsi_dn"] = comp["rsi_dn"]
    n = int(model["ratio_n"]) + 1
    mean = float(model["ratio_mean"]) + (comp["ratio"] - float(model["ratio_mean"])) / n
    dlt = comp["ratio"] - float(model["ratio_mean"])
    out["ratio_n"] = n
    out["ratio_mean"] = mean
    out["ratio_m2"] = float(model["ratio_m2"]) + dlt * (comp["ratio"] - mean)
    for key in ("z", "dd", "vol", "mom", "raw"):
        arr = list(model[key])
        insort(arr, comp[key])
        out[key] = arr
    return out


def blend(fp, weight, fng_value):
    if fng_value is None or weight <= 0:
        return float(fp)
    return float(fp) * (1.0 - float(weight)) + float(fng_value) * float(weight)


def zone_of(value, tiers):
    """Deepest published threshold the reading is inside. Same walk as the page."""
    if value is None or not math.isfinite(value):
        return None
    hit = None
    for t in tiers:
        if value <= t["max"]:
            hit = t
    return hit


def zone_key(value, tiers):
    hit = zone_of(value, tiers)
    return hit["key"] if hit else None


def _round_value(value):
    if value is None:
        return None
    return round(float(value), 3)


def _round_px(value):
    if value is None:
        return None
    return round(float(value), 4)


def build_block(model, session, fng_choice, tiers, official, as_of, weight, sources):
    """Assemble the published btc_intraday object. Official fields are copied, not recomputed."""
    weight = float(weight or 0)
    fng_value = None if fng_choice is None else float(fng_choice["value"])
    low_px = session.get("low")
    live_px = session.get("live")
    low_internal = score_from_model(model, low_px) if low_px else None
    live_internal = score_from_model(model, live_px) if live_px else None
    # Zone off the published (rounded) reading, which is what the page compares
    # to the thresholds. Rounding happens before the L2/L3/L4 decision.
    low_value = None if low_internal is None else _round_value(blend(low_internal, weight, fng_value))
    live_value = None if live_internal is None else _round_value(blend(live_internal, weight, fng_value))
    basis = None
    basis_price = None
    value = None
    if low_value is not None and (live_value is None or low_value <= live_value):
        basis, basis_price, value = "session_low", low_px, low_value
    elif live_value is not None:
        basis, basis_price, value = "live", live_px, live_value
    zone = zone_key(value, tiers) if value is not None else None
    zone_name = None
    if zone:
        zone_name = next(t["name"] for t in tiers if t["key"] == zone)
    official_value = official.get("value")
    official_zone = zone_key(official_value, tiers) if official_value is not None else None
    session_open = bool(session.get("open")) and value is not None
    active = bool(session_open and zone in DEEP_ZONES)
    block = {
        "version": 1,
        "label": "盘中预警",
        "symbol": "BTC-USD",
        "as_of": as_of,
        "session_date": session["date"],
        "history_through": model["history_through"],
        "session_open": session_open,
        "basis": basis,
        "basis_price": _round_px(basis_price),
        "value": value,
        "zone": zone,
        "zone_name": zone_name,
        "active": active,
        "live_price": _round_px(live_px),
        "live_value": live_value,
        "live_zone": zone_key(live_value, tiers) if live_value is not None else None,
        "session_low": _round_px(low_px),
        "low_value": low_value,
        "low_zone": zone_key(low_value, tiers) if low_value is not None else None,
        "official_date": official.get("date"),
        "official_price": official.get("price"),
        "official_value": official.get("value"),
        "official_tier": official.get("tier"),
        "official_zone": official_zone,
        "fng_value": None if fng_value is None else round(fng_value, 3),
        "fng_date": None if fng_choice is None else fng_choice.get("date"),
        "fng_carried": False if fng_choice is None else bool(fng_choice.get("carried")),
        "blend_weight": weight,
        "counts_consecutive_days": False,
        "replaces_official": False,
        "sources": sources,
        "note": NOTE,
        "model": model,
    }
    return block


def problems(block, tiers):
    """Return a list of reasons the block must not be published. Empty means ok."""
    bad = []
    if not isinstance(block, dict):
        return ["btc_intraday is not an object"]
    for key in (
        "version", "label", "symbol", "as_of", "session_date", "history_through",
        "session_open", "active", "official_date", "blend_weight",
        "counts_consecutive_days", "replaces_official", "model", "note",
    ):
        if key not in block:
            bad.append(f"missing {key}")
    if bad:
        return bad
    if block["version"] != 1 or block["symbol"] != "BTC-USD" or block["label"] != "盘中预警":
        bad.append("identity fields drifted")
    if block["replaces_official"] is not False or block["counts_consecutive_days"] is not False:
        bad.append("intraday block claims to replace the official tier")
    if not isinstance(block["as_of"], str) or len(block["as_of"]) != 19:
        bad.append("as_of must be YYYY-MM-DDTHH:MM:SS in Asia/Shanghai")
    if not isinstance(block["session_open"], bool) or not isinstance(block["active"], bool):
        bad.append("session_open and active must be booleans")
    weight = block["blend_weight"]
    if not isinstance(weight, (int, float)) or isinstance(weight, bool) or not 0 <= float(weight) <= 1:
        bad.append("blend_weight out of range")
    value = block.get("value")
    if value is not None and (not isinstance(value, (int, float)) or not 0 <= float(value) <= 100):
        bad.append("value out of range")
    expect = zone_key(value, tiers) if value is not None else None
    if block.get("zone") != expect:
        bad.append("zone does not match the official thresholds")
    if block["active"] != bool(block["session_open"] and expect in DEEP_ZONES):
        bad.append("active does not match an open L2/L3/L4 reading")
    model = block["model"]
    if not isinstance(model, dict) or model.get("history_through") != block["history_through"]:
        bad.append("model history_through does not match")
    else:
        for key in ("z", "dd", "vol", "mom", "raw"):
            arr = model.get(key)
            if not isinstance(arr, list) or len(arr) < 100 or arr != sorted(arr):
                bad.append(f"model.{key} is missing or unsorted")
                break
        closes = model.get("closes")
        if not isinstance(closes, list) or len(closes) < 252:
            bad.append("model.closes is shorter than 252 days")
    return bad


def _clean_close(series):
    s = pd.Series(series).dropna().astype(float)
    if s.empty:
        return None
    idx = pd.to_datetime(s.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    s.index = idx.normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    vals = s.to_numpy(dtype=float)
    s = s[np.isfinite(vals) & (vals > 0)]
    return s if len(s) else None


def fetch_yahoo_daily(symbol="BTC-USD"):
    """Daily closes. period1/period2 stays on 1d; range=max is monthly for BTC."""
    start = int(datetime(2014, 9, 1, tzinfo=timezone.utc).timestamp())
    end = int(datetime.now(timezone.utc).timestamp()) + 86400
    r = http_get(YAHOO_CHART + quote(symbol, safe=""), {"period1": start, "period2": end, "interval": "1d"}, timeout=40)
    chart = r.json().get("chart") or {}
    if chart.get("error") or not chart.get("result"):
        raise RuntimeError(f"yahoo chart: {chart.get('error') or 'empty'}")
    res = chart["result"][0]
    stamps = res.get("timestamp") or []
    closes = (res.get("indicators") or {}).get("quote", [{}])[0].get("close") or []
    if not stamps or not closes:
        raise RuntimeError("yahoo chart returned no closes")
    idx = pd.to_datetime(stamps, unit="s", utc=True)
    series = _clean_close(pd.Series(closes, index=idx))
    if series is None or len(series) < 300:
        raise RuntimeError("yahoo daily history is too short")
    return series


def _frames_from_rows(rows, time_scale):
    if not rows:
        return None
    idx, low, close = [], [], []
    for row in rows:
        try:
            ts = pd.Timestamp(datetime.fromtimestamp(int(row[0]) / time_scale, timezone.utc).date())
            lo = float(row[1] if time_scale == 1 else row[3])
            cl = float(row[4] if time_scale == 1 else row[4])
        except (TypeError, ValueError, IndexError, OSError):
            continue
        if not math.isfinite(lo) or not math.isfinite(cl) or lo <= 0 or cl <= 0:
            continue
        idx.append(ts)
        low.append(lo)
        close.append(cl)
    if not idx:
        return None
    frame = pd.DataFrame({"low": low, "close": close}, index=pd.DatetimeIndex(idx))
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame


def fetch_coinbase_session(days=21):
    end = int(time.time())
    start = end - days * 86400
    r = http_get(COINBASE_CANDLES, {"granularity": 86400, "start": start, "end": end}, timeout=30)
    body = r.json()
    if not isinstance(body, list):
        raise RuntimeError(f"coinbase candles: {body!r}"[:200])
    # Coinbase rows are [time, low, high, open, close, volume], newest first.
    frame = _frames_from_rows([(row[0], row[1], row[2], row[3], row[4]) for row in body], time_scale=1)
    if frame is None or frame.empty:
        raise RuntimeError("coinbase candles were empty")
    return frame


def fetch_kraken_session():
    r = http_get(KRAKEN_OHLC, {"pair": "XBTUSD", "interval": 1440}, timeout=30)
    body = r.json()
    if body.get("error"):
        raise RuntimeError(f"kraken: {body['error']}")
    result = body.get("result") or {}
    key = next((k for k in result if k != "last"), None)
    if not key:
        raise RuntimeError("kraken returned no OHLC")
    # [time, open, high, low, close, vwap, volume, count]
    rows = []
    for row in result[key]:
        rows.append((row[0], row[3], row[2], row[1], row[4]))
    frame = _frames_from_rows(rows, time_scale=1)
    if frame is None or frame.empty:
        raise RuntimeError("kraken OHLC was empty")
    return frame.iloc[-21:]


def fetch_spot():
    r = http_get(COINBASE_SPOT, timeout=20)
    amount = float(((r.json() or {}).get("data") or {}).get("amount"))
    if not math.isfinite(amount) or amount <= 0:
        raise RuntimeError("coinbase spot was empty")
    return amount


def fetch_fng():
    r = http_get(FNG_URL, timeout=30)
    rows = r.json()["data"]
    idx = [pd.Timestamp(datetime.fromtimestamp(int(x["timestamp"]), timezone.utc).date()) for x in rows]
    s = pd.Series([float(x["value"]) for x in rows], index=idx).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s = s[(s >= 0) & (s <= 100)]
    if s.empty:
        raise RuntimeError("alternative.me returned no values")
    return s


def _same_scale(primary, backup):
    overlap = primary.index.intersection(backup.index)
    if len(overlap) < 5:
        return False
    ratio = (backup.loc[overlap].astype(float) / primary.loc[overlap].astype(float)).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if ratio.empty:
        return False
    med = float(ratio.median())
    return 0.97 <= med <= 1.03


def prepare_history(close, session_date, extra=None):
    """Completed closes before the session, with recent Yahoo holes filled."""
    session_ts = pd.Timestamp(session_date)
    history = close[close.index < session_ts]
    filled = []
    source = "yahoo-chart"
    if extra is not None and len(history):
        backup = extra[extra.index < session_ts]
        if _same_scale(history, backup):
            floor = history.index.max() - pd.Timedelta(days=14)
            missing = backup.index.difference(history.index)
            missing = missing[(missing > floor) & (missing < session_ts)]
            if len(missing):
                history = pd.concat([history, backup.loc[missing]]).sort_index()
                history = history[~history.index.duplicated(keep="last")]
                filled = [ymd(d) for d in missing]
                source = "yahoo-chart+coinbase-gap"
        else:
            log("  intraday: skipped gap fill, Coinbase scale does not match Yahoo")
    if history.empty or history.index.max() >= session_ts:
        raise RuntimeError("history leaked into the open session")
    return history, source, filled


def fng_for_session(fng, session_date):
    if fng is None or len(fng) == 0:
        return None
    day = pd.Timestamp(session_date)
    if day in fng.index:
        return {"value": float(fng.loc[day]), "date": ymd(day), "carried": False}
    prev = fng[fng.index <= day]
    if prev.empty:
        return None
    return {"value": float(prev.iloc[-1]), "date": ymd(prev.index[-1]), "carried": True}


def session_from_frame(frame, today, spot):
    today_ts = pd.Timestamp(today)
    low = live = None
    if frame is not None and today_ts in frame.index:
        low = float(frame.loc[today_ts, "low"])
        live = float(frame.loc[today_ts, "close"])
    if spot is not None:
        live = float(spot)
    return {"date": ymd(today_ts), "low": low, "live": live, "open": live is not None or low is not None}


def official_from_payload(payload):
    for asset in payload.get("assets") or []:
        if asset.get("symbol") == "BTC-USD":
            latest = asset.get("latest") or {}
            return {
                "date": latest.get("date"),
                "price": latest.get("price"),
                "value": latest.get("value"),
                "tier": asset.get("tier"),
            }
    raise RuntimeError("dashboard has no BTC-USD asset")


def tiers_from(payload_or_cfg):
    tiers = payload_or_cfg.get("tiers")
    if not tiers:
        raise RuntimeError("tiers missing")
    out = []
    for t in tiers:
        out.append({"key": t["key"], "name": t["name"], "max": float(t["max"]), "ratio": float(t["ratio"])})
    return out


def blend_weight(cfg):
    ext = (cfg or {}).get("external_index") or {}
    if not ext.get("enabled"):
        return 0.0
    return float(ext.get("blend_weight_external") or 0)


def _load_session_frame():
    try:
        frame = fetch_coinbase_session()
        return frame, "coinbase"
    except Exception as e:
        log(f"  coinbase candles: {e}")
    try:
        frame = fetch_kraken_session()
        return frame, "kraken"
    except Exception as e:
        log(f"  kraken candles: {e}")
    return None, None


def compute_block(close, fng, tiers, official, weight, min_hist, now=None):
    """Score the open UTC session. `close` may include today's partial bar; it is dropped."""
    # `now` is Asia/Shanghai and only stamps as_of. The candle boundary is UTC,
    # which is the date Yahoo and Coinbase put on a daily bar.
    now = now or datetime.now(CST)
    today = datetime.now(timezone.utc).date()
    frame, session_source = _load_session_frame()
    spot = None
    try:
        spot = fetch_spot()
    except Exception as e:
        log(f"  coinbase spot: {e}")
    extra = None if frame is None else frame["close"]
    history, history_source, filled = prepare_history(close, today, extra)
    model = build_model(history, min_hist, ymd(history.index[-1]))
    session = session_from_frame(frame, today, spot)
    if session_source is None and session.get("live") is None and session.get("low") is None:
        session_source = "none"
    choice = fng_for_session(fng, today)
    sources = {
        "history": history_source,
        "session": session_source or "none",
        "gaps_filled": filled,
    }
    block = build_block(
        model,
        session,
        choice,
        tiers,
        official,
        now.strftime("%Y-%m-%dT%H:%M:%S"),
        weight,
        sources,
    )
    bad = problems(block, tiers)
    if bad:
        raise RuntimeError("; ".join(bad))
    return block


def summary(block):
    """Loggable view without the scoring model."""
    if not block:
        return None
    return {k: v for k, v in block.items() if k != "model"}


def read_previous(path):
    try:
        with open(path, encoding="utf-8") as handle:
            doc = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    block = doc.get("btc_intraday") if isinstance(doc, dict) else None
    return block if isinstance(block, dict) else None


def replay(close, fng, day, price, weight):
    """Score one historical day as if `price` had been the close. For checks, not the site."""
    day_ts = pd.Timestamp(day)
    history = close[close.index < day_ts]
    model = build_model(history, 1260, ymd(history.index[-1]))
    internal = score_from_model(model, price)
    choice = fng_for_session(fng, day)
    blended = blend(internal, weight, None if choice is None else choice["value"])
    return {
        "day": day,
        "history_through": model["history_through"],
        "price": float(price),
        "internal": internal,
        "blended": blended,
        "fng": choice,
    }


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Replay a BTC close through the intraday scorer.")
    ap.add_argument("--replay", help="UTC session date, YYYY-MM-DD")
    ap.add_argument("--prices", help="Comma-separated hypothetical closes")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    args = ap.parse_args()
    if not args.replay or not args.prices:
        ap.error("--replay and --prices are required")
    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    weight = blend_weight(cfg)
    tiers = tiers_from(cfg)
    close = fetch_yahoo_daily()
    fng = fetch_fng()
    for raw in args.prices.split(","):
        price = float(raw)
        row = replay(close, fng, args.replay, price, weight)
        zone = zone_key(row["blended"], tiers)
        name = next((t["name"] for t in tiers if t["key"] == zone), "未入档")
        print(
            f"{row['day']} px={row['price']:.2f} internal={row['internal']:.3f} "
            f"blended={row['blended']:.3f} zone={zone or '-'} {name} "
            f"fng={row['fng']['value'] if row['fng'] else None} "
            f"history_through={row['history_through']}"
        )


if __name__ == "__main__":
    main()
