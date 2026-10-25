#!/usr/bin/env python3
"""
Rebuild onebitking.com's dashboard_data.json from public sources. No secrets.

  python rebuild/build_dashboard.py --config rebuild/config.yaml \
      --out dashboard_data.json --health rebuild/health.json

Prices: Yahoo via yfinance, then the Yahoo chart API, then Sina via akshare
for the indices that config.yaml names. Rates: FRED's public CSV (the same
file pandas-datareader reads, no API key). If that host does not answer, the
Fed H.15 constant-maturity series (what FRED republishes) is used, with the
latest Treasury.gov daily yield curve appended when it is newer. LPR comes
from Eastmoney via akshare. Bitcoin's external blend is alternative.me.
Bank deposit rates are the static block in config.yaml.

The output file is replaced only after it validates. A run with fewer than
6 of the 9 weighted assets, or with a required section missing, leaves the
existing dashboard_data.json untouched and exits non-zero.
"""
import argparse
import io
import json
import logging
import math
import os
import socket
import sys
import tempfile
import time
import warnings
from bisect import bisect_left, bisect_right, insort
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

os.environ.setdefault("TQDM_DISABLE", "1")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)
socket.setdefaulttimeout(45)

import numpy as np
import pandas as pd
import requests
import yaml

# pandas 3 registers Pandas4Warning on import, which would override an earlier
# ignore. yfinance 1.7 still passes inplace=True; silence that until it stops.
warnings.filterwarnings("ignore", message=".*inplace.*")
warnings.filterwarnings("ignore", category=FutureWarning)

# akshare calls requests.get with no timeout. Give every Session a default
# so a hung Sina or Eastmoney response cannot stall the job.
_orig_request = requests.sessions.Session.request


def _request_with_timeout(self, method, url, **kwargs):
    kwargs.setdefault("timeout", 30)
    return _orig_request(self, method, url, **kwargs)


requests.sessions.Session.request = _request_with_timeout

CST = timezone(timedelta(hours=8))
UA = {"User-Agent": "Mozilla/5.0 (onebitking-dashboard)"}
GREED_LEVELS = [("G1", 65), ("G2", 78), ("G3", 88), ("G4", 95)]
MIN_WEIGHTED = 6
# Fed H.15 constant-maturity package. Same history FRED's DGS*/T10Y2Y series use
# (3m max 17.01, 2y 16.95, 10y 15.84, 30y 15.21). No API key.
H15_URL = (
    "https://www.federalreserve.gov/datadownload/Output.aspx?rel=H15"
    "&series=bf17364827e38702b42a58cf8eaa3f78&lastobs=&from=1962-01-01"
    "&to=2099-12-31&filetype=csv&label=include&layout=seriescolumn"
)
H15_COLUMNS = {
    "RIFLGFCM03_N.B": "3m",
    "RIFLGFCY02_N.B": "2y",
    "RIFLGFCY10_N.B": "10y",
    "RIFLGFCY30_N.B": "30y",
}
TREASURY_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve"
    "&field_tdr_date_value={year}&page&_format=csv"
)
RATE_KEYS = ("yield_curve_2s10s", "yield_3m", "yield_2y", "yield_10y", "yield_30y")


def log(*args):
    print(*args, file=sys.stderr, flush=True)


def ymd(ts):
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def finite_series(values):
    s = pd.Series(values).astype(float)
    return s.where(np.isfinite(s))


def clean_close(series):
    """Daily closes, exchange-local dates, positive and finite, one row per day."""
    s = pd.Series(series).dropna().astype(float)
    if s.empty:
        return None
    idx = pd.to_datetime(s.index)
    if getattr(idx, "tz", None) is not None:
        # Keep the exchange calendar date. Converting to UTC first can roll
        # a midnight-local bar onto the previous UTC evening for Asian markets.
        idx = idx.tz_localize(None)
    s.index = idx.normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s = s[np.isfinite(s.to_numpy(dtype=float)) & (s.to_numpy(dtype=float) > 0)]
    if s.empty:
        return None
    return s


def http_get(url, params=None, timeout=30):
    r = requests.get(url, params=params, timeout=timeout, headers=UA)
    r.raise_for_status()
    return r


# ---------------- prices ----------------
def fetch_yahoo(symbol, retries=3):
    import yfinance as yf

    # yfinance 1.7 still sorts with inplace=True, which pandas 3 warns about.
    warnings.filterwarnings("ignore", category=pd.errors.Pandas4Warning)
    last_err = None
    for i in range(retries):
        try:
            df = yf.Ticker(symbol).history(period="max", auto_adjust=False)
            if df is not None and len(df) and "Close" in df.columns:
                s = clean_close(df["Close"])
                if s is not None and len(s):
                    return s
        except Exception as e:
            last_err = e
            log(f"  yahoo {symbol} try {i + 1}: {e}")
        if i + 1 < retries:
            time.sleep(2 * (i + 1))
    if last_err:
        log(f"  yahoo {symbol} gave up: {last_err}")
    return None


def fetch_yahoo_chart(symbol):
    """Public chart JSON. Used only when yfinance returns nothing."""
    try:
        url = "https://query1.finance.yahoo.com/v8/finance/chart/" + quote(symbol, safe="")
        r = http_get(url, {"range": "max", "interval": "1d"}, timeout=30)
        chart = r.json().get("chart") or {}
        if chart.get("error") or not chart.get("result"):
            log(f"  yahoo-chart {symbol}: {chart.get('error') or 'empty'}")
            return None
        res = chart["result"][0]
        closes = (res.get("indicators") or {}).get("quote", [{}])[0].get("close") or []
        stamps = res.get("timestamp") or []
        if not stamps or not closes:
            return None
        tzname = (res.get("meta") or {}).get("exchangeTimezoneName") or "UTC"
        idx = pd.to_datetime(stamps, unit="s", utc=True).tz_convert(tzname)
        return clean_close(pd.Series(closes, index=idx))
    except Exception as e:
        log(f"  yahoo-chart {symbol}: {e}")
        return None


def fetch_sina(asset):
    try:
        import akshare as ak

        if asset.get("sina_cn"):
            df = ak.stock_zh_index_daily(symbol=asset["sina_cn"])
        elif asset.get("sina_hk"):
            df = ak.stock_hk_index_daily_sina(symbol=asset["sina_hk"])
        elif asset.get("sina"):
            df = ak.index_global_hist_sina(symbol=asset["sina"])
        else:
            return None
        if df is None or df.empty or "close" not in df.columns or "date" not in df.columns:
            return None
        s = pd.Series(pd.to_numeric(df["close"], errors="coerce").values, index=pd.to_datetime(df["date"]))
        return clean_close(s)
    except Exception as e:
        log(f"  sina {asset['symbol']}: {e}")
        return None


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


def fetch_prices(asset):
    symbol = asset["symbol"]
    primary = fetch_yahoo(symbol)
    src = "yahoo"
    if primary is None:
        primary = fetch_yahoo_chart(symbol)
        src = "yahoo-chart"
    backup = None
    if asset.get("sina") or asset.get("sina_cn") or asset.get("sina_hk"):
        backup = fetch_sina(asset)
    if primary is None:
        return (backup, "sina") if backup is not None else (None, None)
    if backup is None or backup.empty:
        return primary, src
    # A short Yahoo reply should not crowd out a long Sina history.
    if len(primary) < 300 and len(backup) > len(primary):
        return backup, "sina"
    if backup.index[-1] > primary.index[-1] and _same_scale(primary, backup):
        extra = backup[backup.index > primary.index[-1]]
        if len(extra):
            merged = pd.concat([primary, extra])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            return merged, src + "+sina"
    return primary, src


# ---------------- rates ----------------
def fetch_fred(series_id, timeout=25):
    r = http_get(
        "https://fred.stlouisfed.org/graph/fredgraph.csv",
        {"id": series_id},
        timeout=timeout,
    )
    head = r.text[:80]
    if "date" not in head.lower():
        raise RuntimeError(f"unexpected FRED body for {series_id}: {head!r}")
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", "v"]
    df["v"] = pd.to_numeric(df["v"], errors="coerce")
    df = df.dropna()
    s = pd.Series(df["v"].to_numpy(dtype=float), index=pd.to_datetime(df["date"]))
    s = clean_level(s)
    if s is None or len(s) < 100:
        raise RuntimeError(f"FRED {series_id} returned too little data")
    return s


def clean_level(series):
    s = pd.Series(series).dropna().astype(float)
    if s.empty:
        return None
    idx = pd.to_datetime(s.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    s.index = idx.normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s = s[np.isfinite(s.to_numpy(dtype=float))]
    return s if len(s) else None


def load_fred(cfg):
    """Fetch every configured FRED series. The first id is the probe: if that
    host does not answer, the rest are skipped instead of waiting on each one."""
    ids = []
    for key in ("yield_10y",) + tuple(k for k in cfg["fred_series"] if k != "yield_10y"):
        sid = cfg["fred_series"][key]["id"]
        if sid not in ids:
            ids.append(sid)
    found, errors = {}, {}
    probe_timeout = 12
    for i, sid in enumerate(ids):
        try:
            found[sid] = fetch_fred(sid, timeout=probe_timeout if i == 0 else 25)
        except Exception as e:
            errors[sid] = str(e)
            if i == 0:
                for rest in ids[1:]:
                    errors.setdefault(rest, f"skipped after {sid} failed")
                break
    return found, errors


def fetch_h15():
    r = http_get(H15_URL, timeout=60)
    lines = r.text.splitlines()
    header_i = next(
        (i for i, line in enumerate(lines) if "Time Period" in line and "RIFLGFCY10_N.B" in line),
        None,
    )
    if header_i is None:
        raise RuntimeError("H.15 CSV has no constant-maturity header")
    df = pd.read_csv(io.StringIO("\n".join(lines[header_i:])))
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in H15_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"H.15 CSV missing {missing}")
    out = pd.DataFrame(index=pd.to_datetime(df["Time Period"]))
    for src, name in H15_COLUMNS.items():
        out[name] = pd.to_numeric(df[src], errors="coerce").to_numpy()
    out.index = pd.to_datetime(out.index).normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if out["10y"].notna().sum() < 1000:
        raise RuntimeError("H.15 10-year history is too short")
    return out


def _treasury_year(year):
    r = http_get(TREASURY_URL.format(year=year), timeout=30)
    if "Date" not in r.text[:200]:
        raise RuntimeError(f"treasury {year} did not return a CSV")
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [str(c).strip() for c in df.columns]
    if "Date" not in df.columns:
        raise RuntimeError(f"treasury {year} has columns {list(df.columns)}")
    frame = pd.DataFrame(index=pd.to_datetime(df["Date"], format="%m/%d/%Y"))
    wanted = {"3 Mo": "3m", "2 Yr": "2y", "10 Yr": "10y", "30 Yr": "30y"}
    for src, name in wanted.items():
        if src not in df.columns:
            frame[name] = np.nan
            continue
        frame[name] = pd.to_numeric(df[src], errors="coerce").to_numpy()
    frame.index = pd.to_datetime(frame.index).normalize()
    return frame[~frame.index.duplicated(keep="last")].sort_index()


def fetch_treasury(years):
    frames = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_treasury_year, year): year for year in years}
        for fut, year in futs.items():
            try:
                frames.append(fut.result())
            except Exception as e:
                log(f"  treasury {year}: {e}")
    if not frames:
        raise RuntimeError("Treasury.gov returned no yield-curve years")
    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def splice_newer(base, extra):
    if extra is None or extra.empty:
        return base
    if base is None or base.empty:
        return extra
    extra = extra[extra.index > base.index.max()]
    if extra.empty:
        return base
    out = pd.concat([base, extra.reindex(columns=base.columns)])
    return out[~out.index.duplicated(keep="last")].sort_index()


def curves_to_series(curves):
    out = {
        "yield_3m": clean_level(curves["3m"]),
        "yield_2y": clean_level(curves["2y"]),
        "yield_10y": clean_level(curves["10y"]),
        "yield_30y": clean_level(curves["30y"]),
    }
    spread = clean_level(curves["10y"] - curves["2y"])
    out["yield_curve_2s10s"] = spread
    missing = [k for k, s in out.items() if s is None or len(s) < 500]
    if missing:
        raise RuntimeError(f"yield history too short for {missing}")
    last_10y = float(out["yield_10y"].iloc[-1])
    if not 0.2 <= last_10y <= 25:
        raise RuntimeError(f"10-year yield {last_10y} is outside a plausible percent range")
    return out


def resolve_rates(cfg, fred_map):
    """Prefer FRED when the host answers; otherwise H.15, then Treasury.gov."""
    fred_series = {}
    for key, spec in cfg["fred_series"].items():
        sid = spec["id"]
        if sid in fred_map:
            fred_series[key] = fred_map[sid]
    if all(k in fred_series for k in RATE_KEYS):
        return fred_series, "FRED"
    year = datetime.now(CST).year
    errors = []
    try:
        curves = splice_newer(fetch_h15(), fetch_treasury([year - 1, year]))
        return curves_to_series(curves), "H.15+Treasury"
    except Exception as e:
        errors.append(f"H.15: {e}")
        log(f"  H.15 fallback failed: {e}")
    try:
        curves = fetch_treasury(list(range(1990, year + 1)))
        series = curves_to_series(curves)
        # credit spread has no Treasury equivalent; keep it if FRED happened to return it
        if "credit_spread" in fred_series:
            series["credit_spread"] = fred_series["credit_spread"]
        return series, "Treasury.gov"
    except Exception as e:
        errors.append(f"Treasury.gov: {e}")
        log(f"  Treasury.gov fallback failed: {e}")
    raise RuntimeError("; ".join(errors) or "no rate source")


# ---------------- scoring (same five components as the attached rebuild) ----------------
def expanding_pct(values):
    """Percentile (0-100) within values up to and including each point. Ties count half."""
    seen, out = [], np.full(len(values), np.nan)
    for i, x in enumerate(values):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            continue
        insort(seen, float(x))
        n = len(seen)
        lo, hi = bisect_left(seen, float(x)), bisect_right(seen, float(x))
        out[i] = (lo + 0.5 * (hi - lo)) / n * 100.0
    return out


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn
    return 100 - 100 / (1 + rs)


def score_asset(close, min_hist):
    """Five components -> raw score -> expanding fear percentile. Low = fear."""
    c = close.astype(float)
    a = finite_series(rsi(c).to_numpy())
    a.index = c.index
    ratio = (c - c.rolling(200).mean()) / c.rolling(200).mean()
    z = (ratio - ratio.expanding().mean()) / ratio.expanding().std()
    b = pd.Series(expanding_pct(finite_series(z).to_numpy()), index=c.index)
    dd = c / c.rolling(252, min_periods=252).max() - 1
    cc = pd.Series(expanding_pct(finite_series(dd).to_numpy()), index=c.index)
    vol = c.pct_change().rolling(20).std()
    d = 100 - pd.Series(expanding_pct(finite_series(vol).to_numpy()), index=c.index)
    mom = c.pct_change(20)
    e = pd.Series(expanding_pct(finite_series(mom).to_numpy()), index=c.index)
    comp = pd.concat([a, b, cc, d, e], axis=1)
    raw = comp.mean(axis=1, skipna=False)
    fp = pd.Series(expanding_pct(finite_series(raw).to_numpy()), index=c.index)
    valid = raw.notna().cumsum()
    fp[valid < min_hist] = np.nan
    return fp


def blend_external(fp, ext, weight):
    if ext is None or weight <= 0:
        return fp
    aligned = ext.reindex(fp.index)
    out = fp.copy()
    mask = aligned.notna() & fp.notna()
    out[mask] = fp[mask] * (1 - weight) + aligned[mask] * weight
    return out


def fetch_fng():
    r = http_get("https://api.alternative.me/fng/?limit=0&format=json", timeout=30)
    rows = r.json()["data"]
    idx = [pd.Timestamp(datetime.fromtimestamp(int(x["timestamp"]), timezone.utc).date()) for x in rows]
    s = pd.Series([float(x["value"]) for x in rows], index=idx).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s = s[(s >= 0) & (s <= 100)]
    if s.empty:
        raise RuntimeError("alternative.me returned no fear/greed values")
    return s


def ahr999(close):
    genesis = pd.Timestamp("2009-01-03")
    age = (close.index - genesis).days.to_numpy(dtype=float)
    gm200 = np.exp(np.log(close).rolling(200).mean())
    fit = 10 ** (5.84 * np.log10(np.maximum(age, 1)) - 17.01)
    return (close / gm200) * (close / fit)


def current_tier(fp, tiers, reset):
    """Walk the fear state forward. One level at a time; several levels may
    clear on the same day once each streak is long enough. Resets after the
    score stays above reset.above for reset.days days. The page itself does
    not read this field; it colors the latest score directly."""
    level, streak, calm = 0, [0] * len(tiers), 0
    above = reset["above"]
    need = reset["days"]
    for v in fp.dropna().to_numpy():
        calm = calm + 1 if v > above else 0
        if calm >= need:
            level, calm = 0, 0
        for k, t in enumerate(tiers):
            streak[k] = streak[k] + 1 if v <= t["max"] else 0
        while level < len(tiers) and streak[level] >= tiers[level].get("consecutive_days", 1):
            level += 1
    return tiers[level - 1]["key"] if level else None


def tier_profile(rows, tiers):
    n = len(rows)
    if n < 60:
        return {}
    hi_win, dq = [0.0] * n, deque()
    for i, (_, _, c) in enumerate(rows):
        while dq and dq[0] < i - 251:
            dq.popleft()
        while dq and rows[dq[-1]][2] <= c:
            dq.pop()
        dq.append(i)
        hi_win[i] = rows[dq[0]][2]

    def profile(hit):
        if not hit:
            return None
        episodes, prev = 1, hit[0]
        for i in hit[1:]:
            if (datetime.fromisoformat(rows[i][0]) - datetime.fromisoformat(rows[prev][0])).days > 30:
                episodes += 1
            prev = i
        dds = sorted((rows[i][2] / hi_win[i] - 1) * 100 for i in hit if hi_win[i])
        return {
            "episodes": episodes,
            "days": len(hit),
            "last": rows[hit[-1]][0],
            "dd": round(dds[len(dds) // 2], 1) if dds else None,
        }

    out = {}
    for t in tiers:
        p = profile([i for i, r in enumerate(rows) if r[1] <= t["max"]])
        if p:
            out[t["key"]] = p
    for key, lo in GREED_LEVELS:
        p = profile([i for i, r in enumerate(rows) if r[1] >= lo])
        if p:
            out[key] = p
    return out


def downsample(points, target):
    """Keep the most fearful point in each bucket so drawdowns are not averaged away."""
    if len(points) <= target:
        return points
    bucket = len(points) / target
    out, i = [], 0.0
    while i < len(points):
        chunk = points[int(i): max(int(i) + 1, int(i + bucket))]
        if chunk:
            out.append(min(chunk, key=lambda p: p[1]))
        i += bucket
    if out and out[-1] != points[-1]:
        out.append(points[-1])
    return out


def macro_item(series, name, note="", neutral=False, extra=None, invert=False):
    s = series.dropna()
    pct = expanding_pct(s.to_numpy(dtype=float))
    if invert:
        pct = 100 - pct
    item = {
        "name": name,
        "note": note or "",
        "neutral": bool(neutral),
        "date": ymd(s.index[-1]),
        "value": round(float(s.iloc[-1]), 4),
        "percentile": round(float(pct[-1]), 3),
    }
    if neutral:
        item["range_min"] = round(float(s.min()), 2)
        item["range_max"] = round(float(s.max()), 2)
    if extra:
        item.update(extra)
    return item


def price_on(close, day):
    if day in close.index:
        value = close.loc[day]
        if isinstance(value, pd.Series):
            value = value.iloc[-1]
        return float(value)
    return float(close.iloc[-1])


def build_asset(asset, close, src, fp, tiers, cfg, cutoff):
    sym = asset["symbol"]
    fpw = fp[fp.index >= cutoff]
    if len(fpw) < 30:
        raise RuntimeError(f"only {len(fpw)} scored days inside the export window")
    px = {ymd(d): float(v) for d, v in close.items()}
    pts = [(ymd(d), round(float(v), 3)) for d, v in fpw.items()]
    full = []
    for d, v in pts:
        p = px.get(d)
        if p is not None and math.isfinite(p):
            full.append((d, v, p))
    thin = downsample(pts, cfg.get("max_points", 900))
    series = []
    for d, v in thin:
        p = px.get(d)
        series.append({"d": d, "v": v, "p": round(p, 4) if p is not None and math.isfinite(p) else None})
    low_v, low_d = min((v, d) for d, v in pts)
    hi_slice = close.iloc[-252:]
    hi52 = float(hi_slice.max()) if len(hi_slice) else None
    disp = bool(asset.get("display_only"))
    last_day = fpw.index[-1]
    last_px = price_on(close, last_day)
    return {
        "symbol": sym,
        "name": asset["name"],
        "group": "display_only" if disp else "cyclical",
        "weight": asset.get("weight"),
        "tradeable": asset.get("tradeable", "") or "",
        "display_only": disp,
        "core": bool(asset.get("core")),
        "tier": None if disp else current_tier(fp, tiers, cfg["reset"]),
        "currency": asset.get("currency", "USD"),
        "latest": {"date": pts[-1][0], "value": pts[-1][1], "price": round(last_px, 4)},
        "low": {"date": low_d, "value": round(low_v, 3)},
        "note": asset.get("note", "") or "",
        "profile": tier_profile(full, tiers),
        "hi52": round(hi52, 4) if hi52 else None,
        "series": series,
    }


def build_composite(cfg, fps):
    weights = {a["symbol"]: a["weight"] for a in cfg["assets"] if a.get("weight") and a["symbol"] in fps}
    if len(weights) < MIN_WEIGHTED:
        return None, None
    # Holiday mismatch: carry the last score forward. index.html documents this
    # (missing days are not scored as 0, and weights are renormalized).
    df = pd.DataFrame({s: fps[s] for s in weights}).sort_index()
    df = df[~df.index.duplicated(keep="last")].ffill()
    w = pd.Series(weights, dtype=float)
    num = (df * w).sum(axis=1, min_count=1)
    den = df.notna().mul(w, axis=1).sum(axis=1)
    comp = (num / den).replace([np.inf, -np.inf], np.nan).dropna()
    if comp.empty:
        return None, None
    # Coverage is against the full configured book. The composite value itself
    # is renormalized over whichever of those assets have a score that day.
    configured = float(sum(a["weight"] for a in cfg["assets"] if a.get("weight")))
    last = comp.index[-1]
    covered = float(den.loc[last]) / configured if configured else 0.0
    composite = {
        "date": ymd(last),
        "value": round(float(comp.iloc[-1]), 3),
        "weight_covered": round(covered, 3),
        "series90": [{"d": ymd(d), "v": round(float(v), 3)} for d, v in comp.iloc[-90:].items()],
    }
    return composite, comp


def fear_episodes(comp, tlt, cutoff):
    joined = pd.concat([comp, tlt], axis=1, keys=["c", "t"]).dropna()
    joined = joined[joined.index >= cutoff]
    # Same rule the attached rebuild used to line up with the episodes the
    # live page was already showing (both in fear on the same day).
    hit = joined[(joined.c <= 30) & (joined.t <= 20)].index
    episodes = []
    for day in hit:
        if episodes and (day - pd.Timestamp(episodes[-1]["end"])).days <= 10:
            episodes[-1]["end"] = ymd(day)
        else:
            episodes.append({"start": ymd(day), "end": ymd(day)})
    return episodes


def build_bond(cfg, rate_series, rate_source, comp, fps, cutoff):
    macro = {}
    for key, spec in cfg["fred_series"].items():
        series = rate_series.get(key)
        if series is None or len(series) == 0:
            continue
        extra = None
        if key == "yield_curve_2s10s":
            extra = {"status": "inverted" if float(series.iloc[-1]) < 0 else "normal"}
        try:
            macro[key] = macro_item(
                series,
                spec["name"],
                spec.get("note", ""),
                spec.get("neutral", False),
                extra,
                invert=spec.get("invert", False),
            )
        except Exception as e:
            log(f"  {key}: {e}")
    order = ["yield_curve_2s10s", "credit_spread", "move_index", "yield_3m", "yield_2y", "yield_10y", "yield_30y"]
    bond = {"macro": {k: macro[k] for k in order if k in macro}}
    if "TLT" in fps and comp is not None and len(comp):
        tlt = fps["TLT"]
        asof = tlt.index[-1]
        history = comp.loc[:asof]
        if len(history):
            cv = float(history.iloc[-1])
            tv = float(tlt.iloc[-1])
            bond["divergence"] = {
                "date": ymd(asof),
                "composite": round(cv, 3),
                "tlt": round(tv, 3),
                "value": round(cv - tv, 3),
            }
            try:
                bond["simultaneous_fear_episodes"] = fear_episodes(comp, tlt, cutoff)
            except Exception as e:
                log(f"  fear episodes: {e}")
                bond["simultaneous_fear_episodes"] = []
    return bond, rate_source


def build_china(cfg):
    import akshare as ak

    last_err = None
    frame = None
    for i in range(3):
        try:
            frame = ak.macro_china_lpr()
            break
        except Exception as e:
            last_err = e
            log(f"  LPR try {i + 1}: {e}")
            time.sleep(2 * (i + 1))
    if frame is None:
        raise RuntimeError(last_err)
    for col in ("TRADE_DATE", "LPR1Y", "LPR5Y"):
        if col not in frame.columns:
            raise RuntimeError(f"LPR frame missing {col}; columns={list(frame.columns)}")
    frame = frame.copy()
    frame["TRADE_DATE"] = pd.to_datetime(frame["TRADE_DATE"])
    frame = frame.sort_values("TRADE_DATE").drop_duplicates("TRADE_DATE", keep="last")
    specs = [
        ("lpr_1y", "LPR1Y", "LPR 1年期", "≈企业贷款利率定价基准"),
        ("lpr_5y", "LPR5Y", "LPR 5年期以上", "≈房贷利率定价基准"),
    ]
    lpr = {}
    for key, col, name, note in specs:
        s = clean_level(pd.Series(pd.to_numeric(frame[col], errors="coerce").to_numpy(), index=frame["TRADE_DATE"]))
        if s is None or len(s) < 10:
            raise RuntimeError(f"{key} has too little history")
        item = macro_item(s, name, note, neutral=True)
        item["series"] = [{"d": ymd(d), "v": float(v)} for d, v in s.items()]
        lpr[key] = item
    return {"lpr": lpr, "bank_rates": cfg["bank_rates"]}


# ---------------- publish gate ----------------
def is_num(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate(payload, weighted_have, weighted_total):
    problems = []
    if weighted_have < MIN_WEIGHTED:
        problems.append(
            f"only {weighted_have} of {weighted_total} weighted assets have usable data "
            f"(need at least {MIN_WEIGHTED})"
        )
    gen = payload.get("generated_at")
    if not isinstance(gen, str) or len(gen) != 19 or gen[10] != "T":
        problems.append("generated_at must be YYYY-MM-DDTHH:MM:SS in Asia/Shanghai, with no timezone suffix")
    if not is_num(payload.get("budget")):
        problems.append("budget missing")
    tiers = payload.get("tiers")
    if not isinstance(tiers, list) or len(tiers) != 4:
        problems.append("tiers must be the four fear levels")
    else:
        for t in tiers:
            if not all(k in t for k in ("key", "name", "max", "ratio")) or not is_num(t["max"]) or not is_num(t["ratio"]):
                problems.append(f"tier {t!r} is incomplete")
    comp = payload.get("composite") or {}
    if not is_num(comp.get("value")) or not (0 <= comp.get("value", -1) <= 100):
        problems.append("composite.value missing or outside 0-100")
    if not isinstance(comp.get("date"), str):
        problems.append("composite.date missing")
    if not is_num(comp.get("weight_covered")) or not (0 < comp.get("weight_covered", 0) <= 1):
        problems.append("composite.weight_covered missing")
    spark = comp.get("series90")
    if not isinstance(spark, list) or len(spark) < 30:
        problems.append("composite.series90 is too short")
    assets = payload.get("assets")
    if not isinstance(assets, list) or not assets:
        problems.append("assets missing")
    else:
        for a in assets:
            sym = a.get("symbol")
            latest = a.get("latest") or {}
            low = a.get("low") or {}
            series = a.get("series")
            if not is_num(latest.get("value")) or not is_num(latest.get("price")) or latest.get("price", 0) <= 0:
                problems.append(f"{sym}: latest price/value missing")
            if not isinstance(latest.get("date"), str) or not isinstance(low.get("date"), str) or not is_num(low.get("value")):
                problems.append(f"{sym}: low/date missing")
            if not isinstance(series, list) or len(series) < 30:
                problems.append(f"{sym}: series too short")
            elif any("d" not in p or not is_num(p.get("v")) for p in series):
                problems.append(f"{sym}: a series point is missing d or v")
            if "display_only" not in a or "currency" not in a or "name" not in a:
                problems.append(f"{sym}: name/currency/display_only missing")
    bond = payload.get("bond_market") or {}
    macro = bond.get("macro") or {}
    curve = macro.get("yield_curve_2s10s") or {}
    if curve.get("status") not in ("normal", "inverted") or not is_num(curve.get("value")) or not is_num(curve.get("percentile")):
        problems.append("yield curve missing value, percentile, or status")
    for key in ("yield_3m", "yield_10y"):
        item = macro.get(key) or {}
        if not all(is_num(item.get(k)) for k in ("value", "percentile", "range_min", "range_max")):
            problems.append(f"{key} missing value, percentile, or range")
    div = bond.get("divergence") or {}
    if not all(is_num(div.get(k)) for k in ("composite", "tlt", "value")) or not isinstance(div.get("date"), str):
        problems.append("stock-bond divergence missing")
    episodes = bond.get("simultaneous_fear_episodes")
    if not isinstance(episodes, list) or any("start" not in e or "end" not in e for e in episodes):
        problems.append("simultaneous_fear_episodes missing")
    china = payload.get("china_rates") or {}
    lpr = china.get("lpr") or {}
    for key in ("lpr_1y", "lpr_5y"):
        item = lpr.get(key) or {}
        series = item.get("series")
        if not all(is_num(item.get(k)) for k in ("value", "range_min", "range_max")):
            problems.append(f"{key} missing value or range")
        if not isinstance(series, list) or len(series) < 10 or any("d" not in p or not is_num(p.get("v")) for p in series or []):
            problems.append(f"{key} series missing")
    banks = (china.get("bank_rates") or {}).get("banks")
    if not isinstance(banks, list) or len(banks) < 4:
        problems.append("bank deposit table missing")
    else:
        for b in banks:
            if not all(is_num(b.get(k)) for k in ("demand", "m3", "y1", "y3", "y5")) or not b.get("name"):
                problems.append(f"bank row incomplete: {b.get('name')}")
    if not isinstance((china.get("bank_rates") or {}).get("as_of"), str):
        problems.append("bank_rates.as_of missing")
    return problems


def atomic_json(path, obj, pretty=False):
    """Replace path only after the full document is on disk. A crash leaves the old file."""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=os.path.dirname(path) or ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            if pretty:
                json.dump(obj, handle, ensure_ascii=False, indent=1, allow_nan=False)
            else:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    ap.add_argument("--out", default="dashboard_data.json")
    ap.add_argument("--health", default="health.json")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    tiers = [
        {k: t[k] for k in ("key", "name", "max", "ratio")} | {"consecutive_days": t.get("consecutive_days", 1)}
        for t in cfg["tiers"]
    ]
    public_tiers = [{k: t[k] for k in ("key", "name", "max", "ratio")} for t in tiers]
    min_hist = cfg.get("min_history_days_for_scoring", 1260)
    cutoff = pd.Timestamp(datetime.now(CST).date() - timedelta(days=365 * cfg.get("years", 10)))
    cfg.setdefault("reset", {"above": 50, "days": 30})
    health = {}
    weighted_symbols = [a["symbol"] for a in cfg["assets"] if a.get("weight")]

    with ThreadPoolExecutor(max_workers=1) as pool:
        fred_future = pool.submit(load_fred, cfg)
        ext_cfg = cfg.get("external_index") or {}
        ext_w = ext_cfg.get("blend_weight_external", 0) if ext_cfg.get("enabled") else 0
        fng = None
        if ext_w:
            try:
                fng = fetch_fng()
                health["alternative.me"] = f"ok (last {ymd(fng.index[-1])})"
            except Exception as e:
                health["alternative.me"] = f"failed: {e}"
                log(f"! alternative.me: {e}")

        fps, out_assets, failures = {}, [], {}
        for asset in cfg["assets"]:
            sym = asset["symbol"]
            try:
                close, src = fetch_prices(asset)
                if close is None or len(close) < 300:
                    raise RuntimeError("no usable price history")
                fp = score_asset(close, min_hist)
                if sym == "BTC-USD" and fng is not None:
                    fp = blend_external(fp, fng, ext_w)
                fp = fp.dropna()
                if fp.empty:
                    raise RuntimeError("score never left the warmup window")
                built = build_asset(asset, close, src, fp, tiers, cfg, cutoff)
                fps[sym] = fp
                out_assets.append(built)
                age = (pd.Timestamp(datetime.now(CST).date()) - close.index[-1]).days
                health[sym] = f"ok ({src}, last {ymd(close.index[-1])}, {age}d old)"
                log(f"  {sym:<12} {src:<16} last {ymd(close.index[-1])}  fp={built['latest']['value']:7.2f}")
                if sym == "BTC-USD":
                    try:
                        ahr = float(ahr999(close).dropna().iloc[-1])
                        if math.isfinite(ahr):
                            health["ahr999"] = round(ahr, 3)
                    except Exception:
                        pass
            except Exception as e:
                failures[sym] = str(e)
                health[sym] = f"failed: {e}"
                log(f"! {sym}: {e}")

        try:
            fred_map, fred_errors = fred_future.result()
        except Exception as e:
            fred_map, fred_errors = {}, {"FRED": str(e)}

    for key, spec in cfg["fred_series"].items():
        sid = spec["id"]
        if sid in fred_map:
            health[f"FRED {sid}"] = f"ok (last {ymd(fred_map[sid].index[-1])})"
        elif sid in fred_errors:
            health[f"FRED {sid}"] = f"failed: {fred_errors[sid]}"

    rate_series, rate_source = {}, None
    try:
        rate_series, rate_source = resolve_rates(cfg, fred_map)
        health["rates"] = rate_source
    except Exception as e:
        health["rates"] = f"failed: {e}"
        log(f"! rates: {e}")

    move = fetch_yahoo("^MOVE")
    if move is not None:
        rate_series["move_index"] = move
        health["^MOVE"] = f"ok (yahoo, last {ymd(move.index[-1])})"
    else:
        health["^MOVE"] = "failed: no data"

    # move_index is not a FRED series; give it a label when present.
    if "move_index" in rate_series:
        cfg["fred_series"].setdefault("move_index", {"id": "^MOVE", "name": "MOVE指数", "invert": True})

    composite, comp = build_composite(cfg, fps)
    bond, _ = build_bond(cfg, rate_series, rate_source, comp if composite else None, fps, cutoff)
    if rate_source and bond.get("macro"):
        dated = {k: v.get("date") for k, v in bond["macro"].items()}
        health["rates"] = f"ok ({rate_source}, {json.dumps(dated, ensure_ascii=False)})"

    china = {"bank_rates": cfg["bank_rates"]}
    try:
        china = build_china(cfg)
        health["LPR (akshare/Eastmoney)"] = f"ok (last {china['lpr']['lpr_1y']['date']})"
    except Exception as e:
        health["LPR (akshare/Eastmoney)"] = f"failed: {e}"
        log(f"! LPR: {e}")

    weighted_have = sum(1 for s in weighted_symbols if s in fps)
    payload = {
        "generated_at": datetime.now(CST).strftime("%Y-%m-%dT%H:%M:%S"),
        "years": cfg.get("years", 10),
        "budget": cfg.get("budget"),
        "tiers": public_tiers,
        "composite": composite,
        "assets": out_assets,
        "bond_market": bond,
        "china_rates": china,
    }
    problems = validate(payload, weighted_have, len(weighted_symbols))
    if failures:
        detail = "; ".join(f"{sym}: {msg}" for sym, msg in failures.items())
        if weighted_have < MIN_WEIGHTED:
            problems.append("failed assets: " + detail)
        else:
            log("assets failed but enough remain to publish:", detail)
    if composite:
        health["composite"] = (
            f"{composite['date']} {composite['value']} coverage {composite['weight_covered']}"
        )
    log(json.dumps({**health, "published": not problems}, ensure_ascii=False, indent=1))
    if problems:
        health["published"] = False
        try:
            atomic_json(args.health, health, pretty=True)
        except Exception as e:
            log(f"! could not write {args.health}: {e}")
        log("REFUSING TO PUBLISH. Existing dashboard file was not modified.")
        for p in problems:
            log(f"  - {p}")
        sys.exit(1)
    atomic_json(args.out, payload)
    health["published"] = True
    atomic_json(args.health, health, pretty=True)
    log(f"PUBLISHED {args.out}")
    log(f"composite {health.get('composite')}")


if __name__ == "__main__":
    main()
