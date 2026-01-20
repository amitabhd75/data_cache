import os
import time
import json
import re
from datetime import date, datetime
from typing import List, Dict, Any, Optional

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
EARN_DIR = os.path.join(DATA_CACHE_DIR, "earnings")

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
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

RISK_HIGH = "HIGH"
RISK_MED = "MED"
RISK_LOW = "LOW"

FED_FOMC_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"

from datetime import timedelta
import yfinance as yf

def save_earnings_cached(
    cache_dir: str,
    symbol: str,
    payload: Dict[str, Any],
    d: Optional[date] = None,
    also_update_latest: bool = True,
) -> None:
    d = d or date.today()
    day_path = _cache_path(cache_dir, symbol, d)
    save_json(day_path, payload)
    if also_update_latest:
        save_json(_latest_path(cache_dir, symbol), payload)

def fetch_earnings_date_yf(symbol: str) -> Optional[date]:
    """
    Best-effort next earnings date from yfinance.
    Returns a python date or None.
    """
    symbol = _sym(symbol)
    try:
        t = yf.Ticker(symbol)

        # Preferred (newer yfinance): get_earnings_dates may exist
        if hasattr(t, "get_earnings_dates"):
            df = t.get_earnings_dates(limit=8)
            if df is not None and len(df) > 0:
                # index is usually Timestamp
                idx = df.index
                # choose first date >= today (UTC-ish)
                today_dt = pd.Timestamp(date.today())
                future = [d for d in idx if pd.Timestamp(d).normalize() >= today_dt]
                if future:
                    return pd.Timestamp(sorted(future)[0]).date()

        # Fallback: calendar field (often has next earnings in it)
        cal = getattr(t, "calendar", None)
        if isinstance(cal, (pd.DataFrame, pd.Series)) and len(cal) > 0:
            # try common keys
            for key in ["Earnings Date", "EarningsDate", "Earnings Date(s)"]:
                if key in cal.index:
                    v = cal.loc[key]
                    # v may be Timestamp or list-like
                    if isinstance(v, (list, tuple, pd.Series)) and len(v) > 0:
                        return pd.Timestamp(v[0]).date()
                    return pd.Timestamp(v).date()

        # Fallback: t.calendar sometimes is DataFrame with first row as earnings date range
        if isinstance(cal, pd.DataFrame) and cal.shape[1] > 0:
            # try take first cell
            v = cal.iloc[0, 0]
            if pd.notna(v):
                return pd.Timestamp(v).date()

        return None
    except Exception:
        return None

def fetch_earnings_date_finnhub(
    symbol: str,
    api_key: str,
    session: Optional[requests.Session] = None,
    lookahead_days: int = 60,
) -> Optional[date]:
    """
    Uses Finnhub earnings calendar endpoint.
    Returns the nearest earnings date >= today if present.
    """
    symbol = _sym(symbol)
    if not api_key:
        return None

    s = session or requests.Session()
    try:
        url = "https://finnhub.io/api/v1/calendar/earnings"
        d0 = date.today()
        d1 = d0 + timedelta(days=lookahead_days)
        params = {
            "symbol": symbol,
            "from": d0.isoformat(),
            "to": d1.isoformat(),
            "token": api_key
        }
        r = s.get(url, params=params, timeout=REQ_TIMEOUT)

        # let caller handle 429 tracking via status_code
        if r.status_code == 429:
            return None

        r.raise_for_status()
        js = r.json() or {}
        items = js.get("earningsCalendar") or []
        if not items:
            return None

        # choose soonest >= today
        dates = []
        for it in items:
            dt = it.get("date")
            if dt:
                try:
                    dd = pd.to_datetime(dt).date()
                    if dd >= d0:
                        dates.append(dd)
                except Exception:
                    pass

        return min(dates) if dates else None
    except Exception:
        return None

# -------------------------
# Generic helpers
# -------------------------
def _iso(d: date) -> str:
    return d.isoformat()

def _sym(symbol: str) -> str:
    return str(symbol).upper().strip()

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


# -------------------------
# Cache helpers (same pattern you use locally)
# -------------------------
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

        if r.status_code == 429:
            out["error"] = r.text
            return out

        r.raise_for_status()
        js = r.json()

        if not isinstance(js, list) or len(js) == 0:
            out["error"] = "empty recommendation list"
            return out

        def _period(x):
            return str(x.get("period", ""))

        js_sorted = sorted(js, key=_period, reverse=True)
        out["ok"] = True
        out["data"] = js_sorted[0]
        return out

    except Exception as e:
        out["error"] = str(e)
        return out


# -----------------------------
# 1) FOMC calendar (Fed) - YOUR FUNCTION (unchanged)
# -----------------------------
def fetch_fomc_meetings(year: int, session: Optional[requests.Session] = None) -> pd.DataFrame:
    """
    Pulls meeting dates from the Fed FOMC calendar page.

    Returns columns:
      EventDate, EventName, Category, Impact, Source, Details
    """
    s = session or requests.Session()
    r = s.get(FED_FOMC_URL, headers=DEFAULT_HEADERS, timeout=30)
    r.raise_for_status()
    html = r.text

    start_pat = re.compile(rf"{year}\s+FOMC\s+Meetings", re.IGNORECASE)
    m = start_pat.search(html)
    if not m:
        return pd.DataFrame(columns=["EventDate", "EventName", "Category", "Impact", "Source", "Details"])

    window = html[m.start():]
    next_year_pat = re.compile(rf"{year+1}\s+FOMC\s+Meetings", re.IGNORECASE)
    n = next_year_pat.search(window)
    if n:
        window = window[:n.start()]

    month_names = (
        "January February March April May June July August September October November December"
    ).split()
    dayrange_pat = re.compile(r"(?P<d1>\d{1,2})\s*-\s*(?P<d2>\d{1,2})(?P<star>\*)?")

    text = re.sub(r"<[^>]+>", " ", window)
    text = _clean_spaces(text)

    tokens = text.split(" ")
    rows: List[Dict[str, Any]] = []
    i = 0
    current_month: Optional[str] = None

    while i < len(tokens):
        t = tokens[i]
        if t in month_names:
            current_month = t
            i += 1
            continue

        if current_month:
            mdr = dayrange_pat.match(t)
            if mdr:
                d1 = int(mdr.group("d1"))
                d2 = int(mdr.group("d2"))
                month_num = month_names.index(current_month) + 1
                start_dt = date(year, month_num, d1)

                details = f"{current_month} {d1}-{d2}" + (" (press conf*)" if mdr.group("star") else "")
                rows.append({
                    "EventDate": start_dt,
                    "EventName": "FOMC Meeting (Start)",
                    "Category": "FOMC",
                    "Impact": RISK_HIGH,
                    "Source": "Federal Reserve",
                    "Details": details,
                })
        i += 1

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("EventDate").reset_index(drop=True)
    return df


# -----------------------------
# 2) BLS release schedule (CPI, Jobs, PPI, etc.) - YOUR FUNCTION (as shared)
# -----------------------------
def fetch_bls_schedule(year: int, session: Optional[requests.Session] = None) -> pd.DataFrame:
    """
    Pulls the BLS schedule for a given year from:
      https://www.bls.gov/schedule/{year}/home.htm

    Returns a normalized table with:
      EventDate, EventName, Category, Impact, Source, Details
    """
    url = f"https://www.bls.gov/schedule/{year}/home.htm"
    s = session or requests.Session()
    r = s.get(url, headers=DEFAULT_HEADERS, timeout=30)
    r.raise_for_status()

    tables = pd.read_html(r.text)
    if not tables:
        return pd.DataFrame(columns=["EventDate", "EventName", "Category", "Impact", "Source", "Details"])

    candidate = None
    for t in tables:
        cols = [str(c).strip().lower() for c in t.columns]
        if any("release" == c for c in cols) and any("date" in c for c in cols):
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
        return pd.DataFrame(columns=["EventDate", "EventName", "Category", "Impact", "Source", "Details"])

    out_rows = []
    for _, r0 in df.iterrows():
        name = _clean_spaces(r0.get(release_col))
        dt = _to_date(r0.get(date_col))
        if not name or not dt:
            continue

        nm_u = name.upper()
        if "CONSUMER PRICE INDEX" in nm_u or nm_u.startswith("CPI"):
            cat, imp = "CPI", RISK_HIGH
        elif "EMPLOYMENT SITUATION" in nm_u or "NONFARM" in nm_u:
            cat, imp = "JOBS", RISK_HIGH
        elif "PRODUCER PRICE" in nm_u or "PPI" in nm_u:
            cat, imp = "PPI", RISK_MED
        elif "RETAIL SALES" in nm_u:
            cat, imp = "RETAIL", RISK_MED
        else:
            cat, imp = "BLS", RISK_LOW

        out_rows.append({
            "EventDate": dt,
            "EventName": name,
            "Category": cat,
            "Impact": imp,
            "Source": "BLS",
            "Details": f"BLS schedule {year}",
        })

    out = pd.DataFrame(out_rows)
    if not out.empty:
        out = out.sort_values("EventDate").reset_index(drop=True)
    return out


# -----------------------------
# Macro calendar writer (Date,Event,Impact only)
# -----------------------------
def build_macro_calendar_csv(out_csv: str, years: List[int], session: Optional[requests.Session] = None) -> pd.DataFrame:
    s = session or requests.Session()
    frames = []

    for y in years:
        # BLS
        try:
            bls = fetch_bls_schedule(y, session=s)
            if not bls.empty:
                frames.append(bls[["EventDate", "EventName", "Impact"]].rename(
                    columns={"EventDate": "Date", "EventName": "Event"}
                ))
        except Exception as e:
            print(f"[macro] BLS failed for {y}: {e}")

        # FOMC
        try:
            fomc = fetch_fomc_meetings(y, session=s)
            if not fomc.empty:
                # Keep EventName ("FOMC Meeting (Start)") as Event
                frames.append(fomc[["EventDate", "EventName", "Impact"]].rename(
                    columns={"EventDate": "Date", "EventName": "Event"}
                ))
        except Exception as e:
            print(f"[macro] FOMC failed for {y}: {e}")

    cal = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["Date", "Event", "Impact"])

    if not cal.empty:
        cal["Date"] = pd.to_datetime(cal["Date"], errors="coerce").dt.date
        cal = cal.dropna(subset=["Date"])
        cal["Event"] = cal["Event"].astype(str).map(_clean_spaces)
        cal["Impact"] = cal["Impact"].astype(str).map(_clean_spaces)
        cal = cal.drop_duplicates(subset=["Date", "Event"]).sort_values(["Date", "Event"]).reset_index(drop=True)

    # Write ISO dates for safety
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
        years = [run_day.year, run_day.year + 1]  # avoids "only Jan" syndrome
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
            time.sleep(SLEEP_ON_MINUTE)
            calls = 0
            bucket_start = time.time()
            rec = fetch_finnhub_recommendation(sym, FINNHUB_API_KEY, session=s)

        if rec.get("ok"):
            ok_count += 1
        else:
            err_count += 1

        save_finnhub_cached_recommendation(FINNHUB_CACHE_DIR, sym, rec, d=run_day, also_update_latest=True)

        if i % 50 == 0:
            log(f"[progress] {i}/{len(symbols)} ok={ok_count} err={err_count} 429={hit_429}")

            # -------------------------
        # Earnings caching (YF -> Finnhub fallback)
        # -------------------------
        # earn_payload = {
        #     "ok": False,
        #     "symbol": sym,
        #     "asof": run_day.isoformat(),
        #     "earnings_date": None,
        #     "source": None,
        #     "error": None,
        # }

        # ed = fetch_earnings_date_yf(sym)
        # if ed:
        #     earn_payload.update({"ok": True, "earnings_date": ed.isoformat(), "source": "YF"})
        # else:
        # Finnhub fallback only if key exists
        if FINNHUB_API_KEY:
            # NOTE: this is an extra Finnhub call -> counts toward rate limit
            # bucket logic: if you want strict counting, increment calls here too
            if calls >= MAX_CALLS_PER_MIN:
                elapsed = time.time() - bucket_start
                log(f"[bucket] calls={calls} elapsed={elapsed:.1f}s -> sleep {SLEEP_ON_MINUTE}s")
                time.sleep(SLEEP_ON_MINUTE)
                calls = 0
                bucket_start = time.time()

            ed2 = fetch_earnings_date_finnhub(sym, FINNHUB_API_KEY, session=s, lookahead_days=60)
            calls += 1  # counts this Finnhub earnings call

            if ed2:
                earn_payload.update({"ok": True, "earnings_date": ed2.isoformat(), "source": "FH"})
            else:
                earn_payload.update({"ok": False, "error": "no earnings date (YF+FH)"})
        else:
            earn_payload.update({"ok": False, "error": "no FINNHUB_API_KEY (YF only)"})

        save_earnings_cached(EARN_DIR, sym, earn_payload, d=run_day, also_update_latest=True)

    
    log(f"Nightly cache done. ok={ok_count} err={err_count} 429={hit_429}")
    log(f"Finnhub cached under: {FINNHUB_CACHE_DIR}/{run_day.isoformat()}/ and /latest/")
    log(f"Macro cached under: {macro_csv}")
    log(f"Log saved: {log_path}")



if __name__ == "__main__":
    main()
