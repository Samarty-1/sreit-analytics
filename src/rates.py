"""Singapore interest-rate inputs, and an honest account of the proxies used.

Why this module has a long docstring
------------------------------------
Interest rates are the single most important input to S-REIT analysis, and
this repo cannot get the ideal series for free. Rather than quietly
substituting something and hoping nobody notices, the substitution is stated
here, its cost is described, and the code path for the real source is left in
place for when it works.

What we would want
------------------
The **Singapore Government Securities 10-year benchmark yield**, published
daily by MAS. It is the correct risk-free anchor for a Singapore-dollar
income asset, and the denominator of every yield-spread calculation below.

What MAS publishes and why it is not used here
-----------------------------------------------
MAS exposes SGS yields through ``eservices.mas.gov.sg``. In practice that
endpoint has been unreliable from this environment -- repeated 404s on the
documented bonds-and-bills path, and timeouts or truncated responses on the
datastore path. :func:`fetch_sgs_10y` implements it anyway and is tried first;
everything falls back automatically when it fails, and the fallback is
reported rather than silently swapped in.

The two proxies
---------------
**Yield LEVEL: the US 10-year Treasury (^TNX).** Not Singapore, and that
matters. It is used because SGS yields track US Treasuries closely -- the
Singapore dollar is managed against a trade-weighted basket rather than an
independent policy rate, so domestic long rates import most of their
variation from global, largely US, rates. The *level* differs though: SGS 10y
has generally traded modestly below UST 10y. So a yield SPREAD computed
against ^TNX is biased low by that gap in absolute terms.

This is why every valuation signal in this repo is built on a **z-score of
the spread against its own trailing history** rather than on the spread's
level. A roughly constant level offset largely drops out of a z-score, which
makes the analysis robust to exactly this substitution. Absolute spread
figures should be read as indicative; z-scores and rankings are the outputs
to trust.

**Rate CHANGES: the ABF Singapore Bond Index Fund (A35.SI).** Genuinely
Singaporean -- an SGX-listed fund holding SGS and quasi-government paper. Its
returns are a real, if levered-by-duration, read on SGD rate moves, which is
what the rate-sensitivity regressions in ``rate_sensitivity.py`` actually
need. Because it is a total-return price series rather than a yield, it gives
changes, not levels -- hence using it for one job and ^TNX for the other.
"""

from __future__ import annotations

import json
import urllib.request
import warnings

import pandas as pd
import yfinance as yf

MAS_DATASTORE = (
    "https://eservices.mas.gov.sg/api/action/datastore/search.json"
    "?resource_id=61e5e886-f7cc-4ded-9fe2-3ff8ab04c500&limit={limit}&sort=end_of_day%20desc"
)
US_10Y = "^TNX"
SG_BOND_FUND = "A35.SI"


def fetch_sgs_10y(limit: int = 5000, timeout: int = 30) -> pd.Series | None:
    """Real SGS 10-year yields from MAS. Returns None if unavailable.

    Tried first by :func:`risk_free_yield`. Returning None rather than raising
    is deliberate: an unreliable upstream should degrade the analysis to a
    documented proxy, not break the pipeline -- but the caller is told which
    it got.
    """
    try:
        request = urllib.request.Request(
            MAS_DATASTORE.format(limit=limit),
            headers={"User-Agent": "sreit-analytics (research)"},
        )
        payload = json.loads(urllib.request.urlopen(request, timeout=timeout).read())
        records = payload["result"]["records"]
    except Exception:  # network, schema drift, truncation -- all mean "use the proxy"
        return None

    frame = pd.DataFrame(records)
    date_col = next((c for c in frame.columns if "end_of_day" in c), None)
    yield_col = next((c for c in frame.columns if c.strip().lower() in {"yield_10y", "sgs10y"}), None)
    if not date_col or not yield_col:
        return None

    series = pd.Series(
        pd.to_numeric(frame[yield_col], errors="coerce").values,
        index=pd.to_datetime(frame[date_col]),
    ).dropna().sort_index()
    return series if len(series) > 100 else None


def fetch_us_10y(period: str = "6y") -> pd.Series:
    """US 10-year Treasury yield in percent."""
    data = yf.download(US_10Y, period=period, progress=False, auto_adjust=True)
    series = data["Close"]
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    series.index = pd.to_datetime(series.index).tz_localize(None)
    return series.dropna()


def fetch_sg_bond_fund(period: str = "6y") -> pd.Series:
    """ABF Singapore Bond Index Fund closing prices -- the SGD rate-move proxy."""
    data = yf.download(SG_BOND_FUND, period=period, progress=False, auto_adjust=True)
    series = data["Close"]
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    series.index = pd.to_datetime(series.index).tz_localize(None)
    return series.dropna()


def risk_free_yield(period: str = "6y", allow_proxy: bool = True) -> tuple[pd.Series, str]:
    """The risk-free yield series, and a label saying where it came from.

    Returns (series, source). `source` is either "MAS SGS 10Y" or
    "PROXY: US 10Y (^TNX)". Callers are expected to surface the label -- an
    analysis built on a proxy should say so in its own output, not bury it in
    a module docstring.
    """
    real = fetch_sgs_10y()
    if real is not None:
        return real, "MAS SGS 10Y"

    if not allow_proxy:
        raise RuntimeError("MAS SGS yields unavailable and allow_proxy=False")

    warnings.warn(
        "MAS SGS 10Y unavailable; falling back to the US 10Y proxy. "
        "Spread LEVELS are biased by the SGS-UST gap; z-scores are largely unaffected. "
        "See src/rates.py.",
        stacklevel=2,
    )
    return fetch_us_10y(period), "PROXY: US 10Y (^TNX)"
