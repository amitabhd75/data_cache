import os
import time
import json
from datetime import date, datetime
from typing import List, Dict, Any, Optional

import pandas as pd
import requests


# -------------------------
# Config
# -------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_CACHE_DIR = os.path.join(REPO_ROOT, "data_cache")
FINNHUB_CACHE_DIR = os.path.join(DATA_CACHE_DIR, "finnhub_recs")
MACRO_DIR = os.path.join(DATA_CACHE_DIR, "macro_calendar")
EARN_DIR = os.path.join(DATA_CACHE_DIR, "earnings")
LOG_DIR = os.path.join(REPO_ROOT, "logs")

UNIVERSE_DIR = os.path.join(REPO_ROOT, "universes")
UNIVERSE_FILES = [
    "sp500.csv",
    "nasdaq100.csv",
    "etfs.csv",
    # add "russell1000.csv" later
]

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "").strip()

# Finnhub free tier says 60/min, but you already hit 429 at times.
# Start conservative: 45/min with backoff on 429.
MAX_CALLS_PER_MIN = int(os.getenv("FINNHUB_MAX_CALLS_PER_MIN", "45"))
SLEEP_ON_MINUTE = int(os.getenv("FINNHUB_SLEEP_SECONDS", "65"))  # sleep after bucket
REQ_TIMEOUT = 15


# -------------------------
# Cache helpers (same pattern you use locally)
# -------------------------
def _iso(d: date) -> str:
    return d.isoformat()

def _sym(symbol: str) -> str:
    return str(symbol).upper().strip()

def _cache_path(cache_dir: str, symbol: str, d: date) -> str:
    return os.path.join(cache_dir, _iso(d), f"{_sym(symbol)}.json")

def _latest_path(cache_dir: str, symbol: str) -> str:
    return os.path.join(cache_dir, "latest", f"{_sym(symbol)}.json")

def save_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

def save_finnhub_cached_recommendation(
    cache_dir: str,
    symbol: str,
    rec: Dict[str, Any],
    d: Optional[date] = None,
    also_update_latest: bool = True,
) -> None:
    d = d or date.today()
    day_path = _cache_path(cache_dir, symbol, d)
    save_json(day_path, rec)
    if also_update_latest:
        save_json(_latest_path(cache_dir, symbol), rec)


# -------------------------
# Finnhub fetch (recommendation trend)
# -------------------------
def fetch_finnhub_recommendation(symbol: str, api_key: str, session: Optional[requests.Session] = None) -> Dict[str, Any]:
    symbol = _sym(symbol)
    out = {"ok": False, "status": None, "data": None, "error": None}

    if not api_key:
        out["error"] = "missing api_key"
        return out

    s = session or requests.Session()
    try:
        url = "https://finnhub.io/api/v1/stock/recommendation"
        params = {"symbol": symbol, "token": api_key}
        r = s.get(url, params=params, timeout=REQ_TIMEOUT)
        out["status"] = r.status_code

        # handle rate limiting upstream
        if r.status_code == 429:
            out["error"] = r.text
            return out

        r.raise_for_status()
        js = r.json()

        if not isinstance(js, list) or len(js) == 0:
            out["error"] = "empty recommendation list"
            return out

        # pick most recent by 'period'
        def _period(x):
            return str(x.get("period", ""))

        js_sorted = sorted(js, key=_period, reverse=True)
        out["ok"] = True
        out["data"] = js_sorted[0]
        return out

    except Exception as e:
        out["error"] = str(e)
        return out


# -------------------------
# Macro calendar (BLS schedule scraper you already validated)
# Output must be: Date,Event,Impact
# -------------------------
def fetch_bls_schedule_csv(out_csv: str) -> pd.DataFrame:
    """
    Scrapes BLS "Release Calendar" style schedule page that lists CPI/JOBS etc.
    You already have a working scraper in your codebase; this is a placeholder hook.
    Replace the URL+parsing with your working version.
    """
    # IMPORTANT: plug your proven BLS schedule URL here (the one that gave you Jan rows).
    # Example patterns on bls.gov differ; keep what you already have.
    url = "https://www.bls.gov/schedule/news_release/bls.htm"  # <-- replace if you used a different one

    r = requests.get(url, timeout=REQ_TIMEOUT)
    r.raise_for_status()

    # Minimal: if your existing function already returns a dataframe, call it instead.
    # For now we just create an empty DF so pipeline doesn't break.
    df = pd.DataFrame(columns=["Date", "Event", "Impact"])

    # TODO: Replace this placeholder with your working parser
    # df = your_existing_bls_parser(r.text)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


# -------------------------
# Universe loader
# -------------------------
def load_symbols_from_universes() -> List[str]:
    syms = []
    for fn in UNIVERSE_FILES:
        path = os.path.join(UNIVERSE_DIR, fn)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if "Symbol" not in df.columns:
            continue
        syms.extend(df["Symbol"].astype(str).tolist())

    # normalize + unique
    syms = [_sym(s) for s in syms if str(s).strip()]
    syms = sorted(list(set(syms)))
    return syms


# -------------------------
# Main: rate-limited caching
# -------------------------
def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(FINNHUB_CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.join(FINNHUB_CACHE_DIR, "latest"), exist_ok=True)
    os.makedirs(MACRO_DIR, exist_ok=True)
    os.makedirs(EARN_DIR, exist_ok=True)

    run_day = date.today()
    run_ts = datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ")
    log_path = os.path.join(LOG_DIR, f"nightly_cache_{run_ts}.log")

    def log(msg: str):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log(f"Nightly cache start: {run_ts}")
    log(f"Symbols sources: {UNIVERSE_FILES}")
    log(f"FINNHUB key present: {bool(FINNHUB_API_KEY)} len={len(FINNHUB_API_KEY)}")
    log(f"Rate limit config: MAX_CALLS_PER_MIN={MAX_CALLS_PER_MIN} SLEEP_ON_MINUTE={SLEEP_ON_MINUTE}s")

    # 1) Macro calendar CSV (Date,Event,Impact)
    macro_csv = os.path.join(MACRO_DIR, "macro_calendar.csv")
    try:
        _ = fetch_bls_schedule_csv(macro_csv)
        log(f"Macro calendar saved: {macro_csv}")
    except Exception as e:
        log(f"Macro fetch failed: {e}")

    # 2) Finnhub recommendation caching
    symbols = load_symbols_from_universes()
    log(f"Total symbols to cache: {len(symbols)}")

    s = requests.Session()

    calls = 0
    bucket_start = time.time()
    ok_count = 0
    err_count = 0
    hit_429 = 0

    for i, sym in enumerate(symbols, 1):
        # bucket logic
        if calls >= MAX_CALLS_PER_MIN:
            elapsed = time.time() - bucket_start
            log(f"[bucket] calls={calls} elapsed={elapsed:.1f}s -> sleep {SLEEP_ON_MINUTE}s")
            time.sleep(SLEEP_ON_MINUTE)
            calls = 0
            bucket_start = time.time()

        rec = fetch_finnhub_recommendation(sym, FINNHUB_API_KEY, session=s)
        calls += 1

        if rec.get("status") == 429:
            hit_429 += 1
            log(f"[429] {sym}: {rec.get('error')}")
            # backoff hard
            time.sleep(SLEEP_ON_MINUTE)
            calls = 0
            bucket_start = time.time()
            # retry once after sleep
            rec = fetch_finnhub_recommendation(sym, FINNHUB_API_KEY, session=s)

        if rec.get("ok"):
            ok_count += 1
        else:
            err_count += 1

        save_finnhub_cached_recommendation(FINNHUB_CACHE_DIR, sym, rec, d=run_day, also_update_latest=True)

        if i % 50 == 0:
            log(f"[progress] {i}/{len(symbols)} ok={ok_count} err={err_count} 429={hit_429}")

    log(f"Nightly cache done. ok={ok_count} err={err_count} 429={hit_429}")
    log(f"Finnhub cached under: {FINNHUB_CACHE_DIR}/{run_day.isoformat()}/ and /latest/")
    log(f"Log saved: {log_path}")


if __name__ == "__main__":
    main()
