import os
import time
import json
from datetime import date, datetime
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import requests


# -------------------------
# Config
# -------------------------
REPO_ROOT = os.environ.get(
    "REPO_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)
DATA_CACHE_DIR = os.path.join(REPO_ROOT, "data_cache")
FINNHUB_CACHE_DIR = os.path.join(DATA_CACHE_DIR, "finnhub_recs")
MACRO_DIR = os.path.join(DATA_CACHE_DIR, "macro_calendar")
EARN_DIR = os.path.join(DATA_CACHE_DIR, "earnings")
LOG_DIR = os.path.join(REPO_ROOT, "logs")

UNIVERSE_DIR = os.path.join(REPO_ROOT, "universes")
UNIVERSE_FILES = [
    "sp500_constituents.csv",
    "nasdaq100_with_baseline.csv",
    "custom_universe_with_sector_baselines.csv",
    # add "russell1000.csv" later
]

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "").strip()

# Finnhub free tier says 60/min, but you already hit 429 at times.
# Start conservative: 45/min with backoff on 429.
MAX_CALLS_PER_MIN = int(os.getenv("FINNHUB_MAX_CALLS_PER_MIN", "45"))
SLEEP_ON_MINUTE = int(os.getenv("FINNHUB_SLEEP_SECONDS", "65"))  # sleep after bucket
REQ_TIMEOUT = 15

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; nightly_cache_job/1.0; +https://github.com/)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


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
# Macro helpers
# -------------------------
RISK_HIGH = "HIGH"
RISK_MED = "MED"
RISK_LOW = "LOW"

def _clean_spaces(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).replace("\xa0", " ").strip()
    s = " ".join(s.split())
    return s

def _to_date(x: Any) -> Optional[date]:
    if x is None:
        return None
    try:
        dt = pd.to_datetime(x, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.date()
    except Exception:
        return None

def _impact_from_text(s: str) -> str:
    s = (s or "").upper()
    if "HIGH" in s:
        return RISK_HIGH
    if "MED" in s or "MEDIUM" in s:
        return RISK_MED
    if "LOW" in s:
        return RISK_LOW
    # if no explicit impact, default MED for Fed/FOMC style, LOW for misc
    return ""


# -----------------------------
# BLS release schedule (CPI, Jobs, PPI, etc.)
# Produces Date,Event,Impact
# -----------------------------
def fetch_bls_schedule(year: int, session: Optional[requests.Session] = None) -> pd.DataFrame:
    """
    Pulls the BLS schedule for a given year from:
      https://www.bls.gov/schedule/{year}/home.htm

    Returns:
      Date, Event, Impact
    """
    url = f"https://www.bls.gov/schedule/{year}/home.htm"
    s = session or requests.Session()
    r = s.get(url, headers=DEFAULT_HEADERS, timeout=30)
    r.raise_for_status()

    tables = pd.read_html(r.text)
    if not tables:
        return pd.DataFrame(columns=["Date", "Event", "Impact"])

    # Find table with "Release" + a date column
    candidate = None
    for t in tables:
        cols = [str(c).strip().lower() for c in t.columns]
        if any(c == "release" for c in cols) and any("date" in c for c in cols):
            candidate = t
            break

    if candidate is None:
        candidate = pd.concat(tables, ignore_index=True)

    df = candidate.copy()
    df.columns = [str(c).strip() for c in df.columns]

    release_col = None
    date_col = None
    for c in df.columns:
        cl = c.lower()
        if release_col is None and cl == "release":
            release_col = c
        if date_col is None and "date" in cl:
            date_col = c

    if release_col is None or date_col is None:
        return pd.DataFrame(columns=["Date", "Event", "Impact"])

    rows = []
    for _, r0 in df.iterrows():
        name = _clean_spaces(r0.get(release_col))
        dt = _to_date(r0.get(date_col))
        if not name or not dt:
            continue

        # Categorize + impact (tune as you like)
        nm_u = name.upper()
        if "CONSUMER PRICE INDEX" in nm_u or nm_u.startswith("CPI"):
            imp = RISK_HIGH
        elif "EMPLOYMENT SITUATION" in nm_u or "NONFARM" in nm_u:
            imp = RISK_HIGH
        elif "PRODUCER PRICE" in nm_u or "PPI" in nm_u:
            imp = RISK_MED
        elif "RETAIL SALES" in nm_u:
            imp = RISK_MED
        elif "GDP" in nm_u:
            imp = RISK_HIGH
        else:
            imp = RISK_LOW

        rows.append({"Date": dt, "Event": name, "Impact": imp})

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("Date").reset_index(drop=True)
    return out


# -----------------------------
# FOMC calendar
# Produces Date,Event,Impact
# -----------------------------
def fetch_fomc_schedule(year: int, session: Optional[requests.Session] = None) -> pd.DataFrame:
    """
    Scrapes FOMC meeting calendar dates from the Fed page(s).
    Output:
      Date, Event, Impact
    """
    s = session or requests.Session()

    # The Fed maintains FOMC calendars on this page; formats sometimes change.
    # We'll parse all tables and extract rows that contain the requested year.
    url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
    r = s.get(url, headers=DEFAULT_HEADERS, timeout=30)
    r.raise_for_status()

    tables = pd.read_html(r.text)
    if not tables:
        return pd.DataFrame(columns=["Date", "Event", "Impact"])

    rows = []
    for t in tables:
        # Expect columns like: "Meeting date", "Minutes release", etc. (varies)
        df = t.copy()
        df.columns = [str(c).strip() for c in df.columns]
        for _, rr in df.iterrows():
            # Scan every cell for a date that belongs to the requested year
            for c in df.columns:
                val = rr.get(c)
                dt = _to_date(val)
                if dt and dt.year == year:
                    # If the table is listing multi-day meetings like "Jan 30-31"
                    # read_html sometimes returns a date-like string; if it becomes a single date, accept it.
                    event_name = "FOMC"
                    # Use column name to add detail
                    cl = str(c).lower()
                    if "minute" in cl:
                        event_name = "FOMC Minutes"
                    elif "meeting" in cl:
                        event_name = "FOMC Meeting"
                    elif "statement" in cl:
                        event_name = "FOMC Statement"
                    else:
                        event_name = "FOMC"

                    rows.append({"Date": dt, "Event": event_name, "Impact": RISK_HIGH})

    out = pd.DataFrame(rows)
    if out.empty:
        # fallback: still create something (non-breaking)
        return pd.DataFrame(columns=["Date", "Event", "Impact"])

    out = out.drop_duplicates(subset=["Date", "Event"]).sort_values(["Date", "Event"]).reset_index(drop=True)
    return out


# -----------------------------
# Build combined macro calendar CSV (Date,Event,Impact)
# -----------------------------
def build_macro_calendar_csv(out_csv: str, years: List[int], session: Optional[requests.Session] = None) -> pd.DataFrame:
    s = session or requests.Session()
    frames = []

    for y in years:
        try:
            frames.append(fetch_bls_schedule(y, session=s))
        except Exception as e:
            print(f"[macro] BLS failed for {y}: {e}")

        try:
            frames.append(fetch_fomc_schedule(y, session=s))
        except Exception as e:
            print(f"[macro] FOMC failed for {y}: {e}")

    cal = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["Date", "Event", "Impact"])

    if not cal.empty:
        cal["Date"] = pd.to_datetime(cal["Date"], errors="coerce").dt.date
        cal = cal.dropna(subset=["Date"])
        cal["Event"] = cal["Event"].astype(str).map(_clean_spaces)
        cal["Impact"] = cal["Impact"].astype(str).map(_clean_spaces)
        cal = cal.drop_duplicates(subset=["Date", "Event"]).sort_values(["Date", "Event"]).reset_index(drop=True)

    # Write ISO dates to be safe across machines
    out_df = cal.copy()
    out_df["Date"] = pd.to_datetime(out_df["Date"]).dt.strftime("%Y-%m-%d")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    return cal


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
    log(f"REPO_ROOT: {REPO_ROOT}")
    log(f"DATA_CACHE_DIR: {DATA_CACHE_DIR}")
    log(f"Symbols sources: {UNIVERSE_FILES}")
    log(f"FINNHUB key present: {bool(FINNHUB_API_KEY)} len={len(FINNHUB_API_KEY)}")
    log(f"Rate limit config: MAX_CALLS_PER_MIN={MAX_CALLS_PER_MIN} SLEEP_ON_MINUTE={SLEEP_ON_MINUTE}s")

    s = requests.Session()

    # 1) Macro calendar CSV (Date,Event,Impact)
    macro_csv = os.path.join(MACRO_DIR, "macro_calendar.csv")
    try:
        years = [run_day.year, run_day.year + 1]
        cal = build_macro_calendar_csv(macro_csv, years=years, session=s)
        log(f"Macro calendar saved: {macro_csv} rows={len(cal)} years={years}")
    except Exception as e:
        log(f"Macro fetch failed: {e}")

    # 2) Finnhub recommendation caching
    symbols = load_symbols_from_universes()
    log(f"Total symbols to cache: {len(symbols)}")

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
    log(f"Macro cached under: {macro_csv}")
    log(f"Log saved: {log_path}")


if __name__ == "__main__":
    main()
