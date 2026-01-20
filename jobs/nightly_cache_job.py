#!/usr/bin/env python3
"""
nightly_cache_job.py

Runs on GitHub Actions nightly to populate:
  1) Finnhub analyst recommendations cache (per symbol, per day + latest)
  2) Earnings cache (YF first, Finnhub fallback; per symbol, per day + latest)
  3) Macro calendar CSV (BLS + FOMC) -> data_cache/macro_calendar/macro_calendar.csv

Outputs:
  data_cache/
    finnhub_recs/
      YYYY-MM-DD/SYMBOL.json
      latest/SYMBOL.json
    earnings/
      YYYY-MM-DD/SYMBOL.json
      latest/SYMBOL.json
    macro_calendar/
      macro_calendar.csv
  logs/
    nightly_cache_<UTCSTAMP>.log

Env:
  REPO_ROOT                 (default: repo root inferred)
  FINNHUB_API_KEY           (required for Finnhub calls)
  FINNHUB_MAX_CALLS_PER_MIN (default: 45)
  FINNHUB_SLEEP_SECONDS     (default: 65)
  FINNHUB_EARN_LOOKAHEAD_DAYS (default: 60)
"""

import os
import time
import json
import re
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional

import pandas as pd
import requests

# yfinance is optional but recommended (used for earnings primary source)
try:
    import yfinance as yf
except Exception:
    yf = None


# -------------------------
# Config
# -------------------------
REPO_ROOT = os.environ.get(
    "REPO_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

DATA_CACHE_DIR = os.path.join(REPO_ROOT, "data_cache")
FINNHUB_CACHE_DIR = os.path.join(DATA_CACHE_DIR, "finnhub_recs")
EARN_DIR = os.path.join(DATA_CACHE_DIR, "earnings")
MACRO_DIR = os.path.join(DATA_CACHE_DIR, "macro_calendar")
LOG_DIR = os.path.join(REPO_ROOT, "logs")

UNIVERSE_DIR = os.path.join(REPO_ROOT, "universes")
UNIVERSE_FILES = [
    "sp500_constituents.csv",
    "nasdaq100_with_baseline.csv",
    "custom_universe_with_sector_baselines.csv",
    # "russell1000_constituents.csv",  # add later
]

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "").strip()

MAX_CALLS_PER_MIN = int(os.getenv("FINNHUB_MAX_CALLS_PER_MIN", "45"))
SLEEP_ON_MINUTE = int(os.getenv("FINNHUB_SLEEP_SECONDS", "65"))
REQ_TIMEOUT = int(os.getenv("REQ_TIMEOUT_SECONDS", "15"))
FINNHUB_EARN_LOOKAHEAD_DAYS = int(os.getenv("FINNHUB_EARN_LOOKAHEAD_DAYS", "60"))

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

RISK_HIGH = "HIGH"
RISK_MED = "MED"
RISK_LOW = "LOW"

FED_FOMC_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"


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
    return " ".join(s.split())

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
# Cache helpers (same pattern as your local)
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
    save_json(_cache_path(cache_dir, symbol, d), rec)
    if also_update_latest:
        save_json(_latest_path(cache_dir, symbol), rec)

def save_earnings_cached(
    cache_dir: str,
    symbol: str,
    payload: Dict[str, Any],
    d: Optional[date] = None,
    also_update_latest: bool = True,
) -> None:
    d = d or date.today()
    save_json(_cache_path(cache_dir, symbol, d), payload)
    if also_update_latest:
        save_json(_latest_path(cache_dir, symbol), payload)


# -------------------------
# Finnhub fetch: recommendation trend
# -------------------------
def fetch_finnhub_recommendation(
    symbol: str,
    api_key: str,
    session: Optional[requests.Session] = None
) -> Dict[str, Any]:
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

        js_sorted = sorted(js, key=lambda x: str(x.get("period", "")), reverse=True)
        out["ok"] = True
        out["data"] = js_sorted[0]
        return out
    except Exception as e:
        out["error"] = str(e)
        return out


# -------------------------
# Earnings: yfinance primary, finnhub fallback
# -------------------------
def fetch_earnings_date_yf(symbol: str) -> Optional[date]:
    """
    Best-effort next earnings date from yfinance.
    Returns a python date or None. ETFs usually None.
    """
    symbol = _sym(symbol)
    if yf is None:
        return None

    try:
        t = yf.Ticker(symbol)

        # Newer yfinance: get_earnings_dates()
        if hasattr(t, "get_earnings_dates"):
            df = t.get_earnings_dates(limit=8)
            if isinstance(df, pd.DataFrame) and not df.empty:
                today_dt = pd.Timestamp(date.today())
                future = [pd.Timestamp(x).normalize() for x in df.index]
                future = [d for d in future if d >= today_dt]
                if future:
                    return sorted(future)[0].date()

        # Fallback: calendar
        cal = getattr(t, "calendar", None)
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            for key in ["Earnings Date", "EarningsDate", "Earnings Date(s)"]:
                if key in cal.index:
                    val = cal.loc[key].dropna().values
                    if len(val) > 0:
                        return pd.to_datetime(val[0]).date()
            # sometimes first cell works
            v = cal.iloc[0, 0]
            if pd.notna(v):
                return pd.to_datetime(v).date()

        # Sometimes calendar is Series
        if isinstance(cal, pd.Series) and not cal.empty:
            for key in ["Earnings Date", "EarningsDate", "Earnings Date(s)"]:
                if key in cal.index:
                    v = cal.loc[key]
                    if isinstance(v, (list, tuple, pd.Series)) and len(v) > 0:
                        return pd.to_datetime(v[0]).date()
                    return pd.to_datetime(v).date()

        return None
    except Exception:
        return None


def fetch_earnings_date_finnhub(
    symbol: str,
    api_key: str,
    session: Optional[requests.Session] = None,
    lookahead_days: int = 60,
) -> Dict[str, Any]:
    """
    Finnhub earnings calendar fallback.
    Returns wrapper dict:
      {ok, status, earnings_date (iso or None), error}
    """
    symbol = _sym(symbol)
    out = {"ok": False, "status": None, "earnings_date": None, "error": None}

    if not api_key:
        out["error"] = "missing api_key"
        return out

    s = session or requests.Session()
    try:
        d0 = date.today()
        d1 = d0 + timedelta(days=int(lookahead_days))

        url = "https://finnhub.io/api/v1/calendar/earnings"
        params = {
            "symbol": symbol,
            "from": d0.isoformat(),
            "to": d1.isoformat(),
            "token": api_key,
        }
        r = s.get(url, params=params, timeout=REQ_TIMEOUT)
        out["status"] = r.status_code

        if r.status_code == 429:
            out["error"] = r.text
            return out

        r.raise_for_status()
        js = r.json() or {}
        items = js.get("earningsCalendar") or []
        if not items:
            out["error"] = "empty earningsCalendar"
            return out

        dates: List[date] = []
        for it in items:
            dt = it.get("date")
            if not dt:
                continue
            try:
                dd = pd.to_datetime(dt).date()
                if dd >= d0:
                    dates.append(dd)
            except Exception:
                continue

        if not dates:
            out["error"] = "no future earnings dates in window"
            return out

        out["ok"] = True
        out["earnings_date"] = min(dates).isoformat()
        return out

    except Exception as e:
        out["error"] = str(e)
        return out


# -----------------------------
# 1) FOMC calendar (Fed)
# -----------------------------
def fetch_fomc_meetings(year: int, session: Optional[requests.Session] = None) -> pd.DataFrame:
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

    month_names = "January February March April May June July August September October November December".split()
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
# 2) BLS release schedule
# -----------------------------
def fetch_bls_schedule(year: int, session: Optional[requests.Session] = None) -> pd.DataFrame:
    """
    Pulls BLS schedule for year from:
      https://www.bls.gov/schedule/{year}/home.htm
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
        if ("release" in cols) and any("date" in c for c in cols):
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
def build_macro_calendar_csv(
    out_csv: str,
    years: List[int],
    session: Optional[requests.Session] = None
) -> pd.DataFrame:
    s = session or requests.Session()
    frames = []

    for y in years:
        # BLS
        try:
            bls = fetch_bls_schedule(y, session=s)
            if not bls.empty:
                frames.append(
                    bls[["EventDate", "EventName", "Impact"]].rename(
                        columns={"EventDate": "Date", "EventName": "Event"}
                    )
                )
        except Exception as e:
            print(f"[macro] BLS failed for {y}: {e}")

        # FOMC
        try:
            fomc = fetch_fomc_meetings(y, session=s)
            if not fomc.empty:
                frames.append(
                    fomc[["EventDate", "EventName", "Impact"]].rename(
                        columns={"EventDate": "Date", "EventName": "Event"}
                    )
                )
        except Exception as e:
            print(f"[macro] FOMC failed for {y}: {e}")

    cal = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["Date", "Event", "Impact"])
    if not cal.empty:
        cal["Date"] = pd.to_datetime(cal["Date"], errors="coerce").dt.date
        cal = cal.dropna(subset=["Date"])
        cal["Event"] = cal["Event"].astype(str).map(_clean_spaces)
        cal["Impact"] = cal["Impact"].astype(str).map(_clean_spaces)
        cal = cal.drop_duplicates(subset=["Date", "Event"]).sort_values(["Date", "Event"]).reset_index(drop=True)

    out_df = cal.copy()
    out_df["Date"] = pd.to_datetime(out_df["Date"]).dt.strftime("%Y-%m-%d")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    return cal


# -------------------------
# Universe loader
# -------------------------
def load_symbols_from_universes() -> List[str]:
    syms: List[str] = []
    for fn in UNIVERSE_FILES:
        path = os.path.join(UNIVERSE_DIR, fn)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if "Symbol" not in df.columns:
            continue
        syms.extend(df["Symbol"].astype(str).tolist())

    syms = [_sym(s) for s in syms if str(s).strip()]
    return sorted(list(set(syms)))


# -------------------------
# Simple Finnhub rate limiter
# -------------------------
class FinnhubLimiter:
    def __init__(self, max_calls_per_min: int, sleep_seconds: int, logger):
        self.max_calls = max_calls_per_min
        self.sleep_seconds = sleep_seconds
        self.log = logger
        self.calls = 0
        self.bucket_start = time.time()

    def _sleep_bucket(self):
        elapsed = time.time() - self.bucket_start
        self.log(f"[bucket] calls={self.calls} elapsed={elapsed:.1f}s -> sleep {self.sleep_seconds}s")
        time.sleep(self.sleep_seconds)
        self.calls = 0
        self.bucket_start = time.time()

    def before_call(self):
        if self.calls >= self.max_calls:
            self._sleep_bucket()

    def after_call(self):
        self.calls += 1

    def backoff_on_429(self, context: str, symbol: str, err: str):
        self.log(f"[429]{context} {symbol}: {err}")
        self._sleep_bucket()


# -------------------------
# Main
# -------------------------
def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    os.makedirs(FINNHUB_CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.join(FINNHUB_CACHE_DIR, "latest"), exist_ok=True)

    os.makedirs(EARN_DIR, exist_ok=True)
    os.makedirs(os.path.join(EARN_DIR, "latest"), exist_ok=True)

    os.makedirs(MACRO_DIR, exist_ok=True)

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
    log(f"YFinance available: {yf is not None}")

    session = requests.Session()
    limiter = FinnhubLimiter(MAX_CALLS_PER_MIN, SLEEP_ON_MINUTE, logger=log)

    # 1) Macro calendar
    macro_csv = os.path.join(MACRO_DIR, "macro_calendar.csv")
    try:
        years = [run_day.year, run_day.year + 1]
        cal = build_macro_calendar_csv(macro_csv, years=years, session=session)
        log(f"Macro calendar saved: {macro_csv} rows={len(cal)} years={years}")
    except Exception as e:
        log(f"Macro fetch failed: {e}")

    # 2) Symbols
    symbols = load_symbols_from_universes()
    log(f"Total symbols to cache: {len(symbols)}")

    # Counters
    rec_ok = rec_err = rec_429 = 0
    earn_ok = earn_err = earn_429 = 0

    for i, sym in enumerate(symbols, 1):
        # -------------------------
        # A) Finnhub recommendation (1 Finnhub call)
        # -------------------------
        rec = {"ok": False, "status": None, "data": None, "error": None}
        if FINNHUB_API_KEY:
            limiter.before_call()
            rec = fetch_finnhub_recommendation(sym, FINNHUB_API_KEY, session=session)
            limiter.after_call()

            if rec.get("status") == 429:
                rec_429 += 1
                limiter.backoff_on_429("[recs]", sym, rec.get("error") or "")
                # retry once
                limiter.before_call()
                rec = fetch_finnhub_recommendation(sym, FINNHUB_API_KEY, session=session)
                limiter.after_call()

        if rec.get("ok"):
            rec_ok += 1
        else:
            rec_err += 1

        save_finnhub_cached_recommendation(FINNHUB_CACHE_DIR, sym, rec, d=run_day, also_update_latest=True)

        # -------------------------
        # B) Earnings (YF first; Finnhub fallback -> extra Finnhub call sometimes)
        # -------------------------
        earn_payload = {
            "ok": False,
            "symbol": _sym(sym),
            "asof": run_day.isoformat(),
            "earnings_date": None,
            "source": None,
            "status": None,
            "error": None,
        }

        ed = fetch_earnings_date_yf(sym)
        if ed:
            earn_payload.update({"ok": True, "earnings_date": ed.isoformat(), "source": "YF"})
            earn_ok += 1
        else:
            if FINNHUB_API_KEY:
                limiter.before_call()
                fh = fetch_earnings_date_finnhub(sym, FINNHUB_API_KEY, session=session, lookahead_days=FINNHUB_EARN_LOOKAHEAD_DAYS)
                limiter.after_call()

                if fh.get("status") == 429:
                    earn_429 += 1
                    limiter.backoff_on_429("[earn]", sym, fh.get("error") or "")
                    # retry once
                    limiter.before_call()
                    fh = fetch_earnings_date_finnhub(sym, FINNHUB_API_KEY, session=session, lookahead_days=FINNHUB_EARN_LOOKAHEAD_DAYS)
                    limiter.after_call()

                if fh.get("ok") and fh.get("earnings_date"):
                    earn_payload.update({"ok": True, "earnings_date": fh["earnings_date"], "source": "FH"})
                    earn_ok += 1
                else:
                    earn_payload.update({
                        "ok": False,
                        "source": "FH",
                        "status": fh.get("status"),
                        "error": fh.get("error") or "no earnings date (YF None; FH None)",
                    })
                    earn_err += 1
            else:
                earn_payload.update({"ok": False, "error": "YF None; FH disabled (missing API key)"})
                earn_err += 1

        save_earnings_cached(EARN_DIR, sym, earn_payload, d=run_day, also_update_latest=True)

        if i % 50 == 0:
            log(
                f"[progress] {i}/{len(symbols)} "
                f"recs ok={rec_ok} err={rec_err} 429={rec_429} | "
                f"earn ok={earn_ok} err={earn_err} 429={earn_429}"
            )

    log("Nightly cache done.")
    log(f"Recs: ok={rec_ok} err={rec_err} 429={rec_429}")
    log(f"Earnings: ok={earn_ok} err={earn_err} 429={earn_429}")
    log(f"Finnhub cached under: {FINNHUB_CACHE_DIR}/{run_day.isoformat()}/ and /latest/")
    log(f"Earnings cached under: {EARN_DIR}/{run_day.isoformat()}/ and /latest/")
    log(f"Macro cached under: {macro_csv}")
    log(f"Log saved: {log_path}")


if __name__ == "__main__":
    main()
