"""Does the yield-spread z-score actually predict forward returns?

The valuation logic in ``metrics.py`` is the standard S-REIT framework and it
is entirely plausible: a REIT whose yield spread over bonds is unusually wide
should be cheap, and should subsequently outperform. Plausible is not
evidence, so this module tests it and reports whatever comes out.

Discipline applied here
-----------------------
* **Cross-sectional, not time-series.** Ranking REITs against each other on
  each date nets out whatever the whole sector did, which is dominated by
  rates and would otherwise be most of the measured "signal".
* **Trailing z-scores only.** ``rolling_zscore`` standardises against a
  trailing window, never the full sample. A full-sample z-score is the most
  common way a valuation signal ends up accidentally clairvoyant.
* **Selection and confirmation sets.** The data is split in two by date up
  front. Anything chosen -- horizon, quantile depth -- is chosen on the
  selection half. The confirmation half is looked at once, at the end, and not
  returned to. Tuning on all the data and reporting the best result is how a
  noise finding becomes a published one.
* **Information coefficient, not just quantile returns.** The rank
  correlation between signal and forward return uses the whole cross-section
  rather than only the tails, so it is far harder to flatter with a lucky
  quantile boundary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def information_coefficient(signal: pd.DataFrame, forward_returns: pd.DataFrame) -> pd.Series:
    """Per-date Spearman rank correlation between signal and forward return.

    The mean IC is the headline; its t-statistic is what says whether the mean
    is distinguishable from zero. Cross-sectional ICs on a couple of dozen
    names are noisy, and a mean IC of 0.03 over 40 dates is not a finding.
    """
    common_dates = signal.index.intersection(forward_returns.index)
    out = {}
    for date in common_dates:
        s = signal.loc[date].dropna()
        r = forward_returns.loc[date].dropna()
        shared = s.index.intersection(r.index)
        if len(shared) >= 8:
            out[date] = s[shared].corr(r[shared], method="spearman")
    return pd.Series(out).dropna().sort_index()


def ic_statistics(ic: pd.Series) -> dict:
    if len(ic) < 3:
        return {"n_dates": len(ic), "mean_ic": np.nan, "ic_tstat": np.nan, "hit_rate": np.nan}
    mean = float(ic.mean())
    tstat = float(mean / (ic.std(ddof=1) / np.sqrt(len(ic)))) if ic.std(ddof=1) > 0 else np.nan
    return {
        "n_dates": int(len(ic)),
        "mean_ic": round(mean, 4),
        "ic_std": round(float(ic.std(ddof=1)), 4),
        "ic_tstat": round(tstat, 3),
        "hit_rate": round(float((ic > 0).mean()), 4),
        "significant_at_5pct": bool(abs(tstat) > 1.96) if np.isfinite(tstat) else False,
    }


def quantile_spread(
    signal: pd.DataFrame, forward_returns: pd.DataFrame, quantile: float = 0.3
) -> pd.Series:
    """Return of (top quantile by signal) minus (bottom quantile), per date."""
    common_dates = signal.index.intersection(forward_returns.index)
    out = {}
    for date in common_dates:
        s = signal.loc[date].dropna()
        r = forward_returns.loc[date].dropna()
        shared = s.index.intersection(r.index)
        if len(shared) < 8:
            continue
        s, r = s[shared].sort_values(), r[shared]
        cut = max(int(len(shared) * quantile), 1)
        out[date] = float(r[s.index[-cut:]].mean() - r[s.index[:cut]].mean())
    return pd.Series(out).dropna().sort_index()


def forward_returns(prices: pd.DataFrame, dates: pd.DatetimeIndex, horizon_days: int) -> pd.DataFrame:
    """Forward return over `horizon_days` starting at each date in `dates`.

    Indexed by the date the signal is OBSERVED, and computed strictly forward
    from it, so a signal is never paired with a return that overlaps its own
    observation window.
    """
    rows = {}
    index = prices.index
    for date in dates:
        start_positions = index.searchsorted(date)
        end_positions = start_positions + horizon_days
        if start_positions >= len(index) or end_positions >= len(index):
            continue
        start_prices = prices.iloc[start_positions]
        end_prices = prices.iloc[end_positions]
        rows[date] = (end_prices / start_prices - 1)
    return pd.DataFrame(rows).T


def split_selection_confirmation(dates: pd.DatetimeIndex, fraction: float = 0.5):
    """Split dates in two by time. Everything is chosen on the first half."""
    dates = pd.DatetimeIndex(sorted(dates))
    cut = int(len(dates) * fraction)
    return dates[:cut], dates[cut:]


def evaluate(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    horizon_days: int,
    dates: pd.DatetimeIndex | None = None,
    quantile: float = 0.3,
) -> dict:
    """IC and quantile-spread statistics for one signal at one horizon."""
    dates = dates if dates is not None else signal.index
    fwd = forward_returns(prices, pd.DatetimeIndex(dates), horizon_days)
    if fwd.empty:
        return {"horizon_days": horizon_days, "n_dates": 0, "mean_ic": np.nan}

    aligned_signal = signal.reindex(fwd.index)
    ic = information_coefficient(aligned_signal, fwd)
    spread = quantile_spread(aligned_signal, fwd, quantile)

    periods_per_year = 252 / horizon_days
    result = {"horizon_days": horizon_days, **ic_statistics(ic)}
    if len(spread) >= 3 and spread.std(ddof=1) > 0:
        result["mean_quantile_spread"] = round(float(spread.mean()), 4)
        result["quantile_spread_sharpe"] = round(
            float(spread.mean() * periods_per_year / (spread.std(ddof=1) * np.sqrt(periods_per_year))), 3
        )
    else:
        result["mean_quantile_spread"] = np.nan
        result["quantile_spread_sharpe"] = np.nan
    return result
