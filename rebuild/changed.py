"""Decide whether a rebuilt dashboard_data.json should be committed.

Exit codes (read by .github/workflows/update.yml):
  0  publish — data differs from HEAD beyond generated_at, or HEAD has no file yet
  2  unchanged — only generated_at moved, or the documents are equal
  1  error — new file missing or not valid JSON, or git itself failed

A missing HEAD copy is a successful first run, not an error: the new file
should be published. A broken new file is an error so CI does not treat it
as "no change" and exit green.
"""
import json
import os
import subprocess
import sys

REQUIRED = ("generated_at", "assets", "composite", "tiers", "bond_market", "china_rates")


def repo_root():
    out = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], stderr=subprocess.PIPE)
    return out.decode("utf-8").strip()


def load_new(path):
    if not os.path.exists(path):
        print(f"{path} is missing; not committing", file=sys.stderr)
        sys.exit(1)
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except json.JSONDecodeError as e:
        print(f"{path} is not valid JSON ({e}); not committing", file=sys.stderr)
        sys.exit(1)
    if not isinstance(doc, dict):
        print(f"{path} is not a JSON object; not committing", file=sys.stderr)
        sys.exit(1)
    missing = [k for k in REQUIRED if k not in doc]
    if missing:
        print(f"{path} is missing {', '.join(missing)}; not committing", file=sys.stderr)
        sys.exit(1)
    return doc


def load_head(relpath):
    try:
        raw = subprocess.check_output(["git", "show", f"HEAD:{relpath}"], stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return None
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        # Unreadable history should not block a good replacement.
        return None
    return doc if isinstance(doc, dict) else None


def main():
    try:
        root = repo_root()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"git rev-parse failed: {e}", file=sys.stderr)
        sys.exit(1)
    os.chdir(root)
    rel = "dashboard_data.json"
    new = load_new(os.path.join(root, rel))
    old = load_head(rel)
    if old is None:
        print("no dashboard_data.json in HEAD; treating this as the first publish")
        sys.exit(0)
    old.pop("generated_at", None)
    new.pop("generated_at", None)
    if old == new:
        print("no meaningful change beyond generated_at")
        sys.exit(2)
    print("dashboard_data.json changed")
    sys.exit(0)


if __name__ == "__main__":
    main()
