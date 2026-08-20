#!/usr/bin/env python3
"""
把 sentiment.db + config.yaml 导出成 dashboard_data.json，供 index.html 读取。

用法（在 ~/tether-monitor 下）：
    python3 export_dashboard.py
    python3 export_dashboard.py --years 8 --out docs/dashboard_data.json

脚本对 schema 做自动探测：不假设列名，先读 sqlite_master 找到存放
fear_percentile 的表和列，找不到会明确报错并列出候选，不会静默产出空文件。
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import date, datetime, timedelta

try:
    import yaml
except ImportError:
    print("需要 PyYAML：pip install pyyaml", file=sys.stderr)
    raise


# ---------- schema 探测 ----------

# 可能存放每日分数的表名，按优先级
SCORE_TABLE_HINTS = ["scores", "fear_scores", "daily_scores", "sentiment_scores"]
# 可能的百分位列名，按优先级
PCT_COL_HINTS = [
    "fear_percentile",
    "final_fear_percentile",
    "blended_fear_percentile",
    "self_fear_percentile",
]
DATE_COL_HINTS = ["date", "trade_date", "day", "dt"]
SYMBOL_COL_HINTS = ["symbol", "ticker", "asset"]


def list_tables(conn):
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return [r[0] for r in rows]


def columns_of(conn, table):
    return [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")').fetchall()]


def pick(candidates, available):
    for c in candidates:
        if c in available:
            return c
    return None


def find_score_source(conn):
    """返回 (table, date_col, symbol_col, pct_col)。找不到就抛错并报告实际 schema。"""
    tables = list_tables(conn)
    ordered = [t for t in SCORE_TABLE_HINTS if t in tables] + [
        t for t in tables if t not in SCORE_TABLE_HINTS
    ]
    report = []
    for t in ordered:
        cols = columns_of(conn, t)
        pct = pick(PCT_COL_HINTS, cols)
        dcol = pick(DATE_COL_HINTS, cols)
        scol = pick(SYMBOL_COL_HINTS, cols)
        if pct and dcol and scol:
            return t, dcol, scol, pct
        if pct or (dcol and scol):
            report.append(f"  {t}: {', '.join(cols)}")

    msg = ["在 sentiment.db 里找不到同时含 日期/标的/fear_percentile 的表。"]
    msg.append("下面是可能相关的表和它们的列，请把正确的表名列名告诉我：")
    msg.extend(report or [f"  {t}: {', '.join(columns_of(conn, t))}" for t in tables])
    raise SystemExit("\n".join(msg))


def find_state_table(conn):
    """symbol_state 表，用来读当前解锁档位。找不到返回 None，不是致命错误。"""
    for t in ("symbol_state", "states", "symbol_states"):
        if t in list_tables(conn):
            cols = columns_of(conn, t)
            scol = pick(SYMBOL_COL_HINTS, cols)
            tier = pick(["unlocked_tier", "tier", "current_tier"], cols)
            if scol and tier:
                return t, scol, tier
    return None


# ---------- config ----------


def load_assets(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assets = []
    raw = cfg.get("assets", {})
    if isinstance(raw, dict):
        for group, items in raw.items():
            for a in items or []:
                assets.append(
                    {
                        "symbol": a["symbol"],
                        "name": a.get("name", a["symbol"]),
                        "group": group,
                        "weight": a.get("weight"),
                        "tradeable": a.get("tradeable", ""),
                    }
                )
    else:  # 平铺列表的写法
        for a in raw or []:
            assets.append(
                {
                    "symbol": a["symbol"],
                    "name": a.get("name", a["symbol"]),
                    "group": a.get("group", ""),
                    "weight": a.get("weight"),
                    "tradeable": a.get("tradeable", ""),
                }
            )

    tiers = []
    tcfg = (cfg.get("triggers") or {}).get("tiers") or {}
    for key in ("L1", "L2", "L3", "L4"):
        t = tcfg.get(key) or {}
        tiers.append(
            {
                "key": key,
                "name": t.get("name", key),
                "max": t.get("fear_percentile_max"),
                "ratio": t.get("tier_ratio", 0),
            }
        )

    futures = {
        f["symbol"]: f.get("note", "")
        for f in (cfg.get("continuous_futures_symbols") or [])
        if isinstance(f, dict) and "symbol" in f
    }

    return assets, tiers, futures


# ---------- 抽稀 ----------


def downsample(points, target):
    """保留极值的抽稀：分桶，每桶取最低点（最恐慌那天不能被平滑掉）。"""
    if len(points) <= target:
        return points
    bucket = len(points) / target
    out, i = [], 0.0
    while i < len(points):
        chunk = points[int(i) : max(int(i) + 1, int(i + bucket))]
        if chunk:
            out.append(min(chunk, key=lambda p: p[1]))
        i += bucket
    if out and out[-1] != points[-1]:
        out.append(points[-1])
    return out


# ---------- 主流程 ----------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="sentiment.db")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="dashboard_data.json")
    ap.add_argument("--years", type=int, default=10, help="导出最近几年，默认10")
    ap.add_argument("--max-points", type=int, default=900, help="每标的最多点数")
    args = ap.parse_args()

    if not os.path.exists(args.db):
        raise SystemExit(f"找不到 {args.db}，请在项目根目录运行")
    if not os.path.exists(args.config):
        raise SystemExit(f"找不到 {args.config}")

    assets, tiers, futures = load_assets(args.config)
    conn = sqlite3.connect(args.db)

    table, dcol, scol, pcol = find_score_source(conn)
    print(f"分数来源：{table}.{pcol}（日期列 {dcol}，标的列 {scol}）")

    state = find_state_table(conn)
    tier_by_symbol = {}
    if state:
        st, ssym, stier = state
        for sym, tier in conn.execute(f'SELECT "{ssym}", "{stier}" FROM "{st}"'):
            tier_by_symbol[sym] = tier
        print(f"当前档位来源：{st}.{stier}")
    else:
        print("未找到 symbol_state 表，当前档位留空")

    cutoff = (date.today() - timedelta(days=365 * args.years)).isoformat()

    out_assets = []
    for a in assets:
        rows = conn.execute(
            f'SELECT "{dcol}", "{pcol}" FROM "{table}" '
            f'WHERE "{scol}" = ? AND "{pcol}" IS NOT NULL AND "{dcol}" >= ? '
            f'ORDER BY "{dcol}" ASC',
            (a["symbol"], cutoff),
        ).fetchall()

        pts = [(str(d)[:10], round(float(v), 3)) for d, v in rows]
        if not pts:
            print(f"  ! {a['symbol']}：{args.years}年内无分数，跳过")
            continue

        latest_date, latest_val = pts[-1]
        low_val, low_date = min((v, d) for d, v in pts)

        thin = downsample(pts, args.max_points)

        out_assets.append(
            {
                **a,
                "tier": tier_by_symbol.get(a["symbol"]) or None,
                "latest": {"date": latest_date, "value": latest_val},
                "low": {"date": low_date, "value": round(low_val, 3)},
                "note": futures.get(a["symbol"], ""),
                "series": [{"d": d, "v": v} for d, v in thin],
            }
        )
        print(f"  {a['symbol']:<11} {len(pts):>6} 行 → {len(thin):>4} 点  最新 {latest_val:.2f}")

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "years": args.years,
        "tiers": tiers,
        "assets": out_assets,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))

    size = os.path.getsize(args.out) / 1024
    print(f"\n已写出 {args.out}（{size:.0f} KB，{len(out_assets)} 个标的）")


if __name__ == "__main__":
    main()
