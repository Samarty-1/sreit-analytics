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
* **Overlapping windows corrected, not assumed away.** Observation dates are
  21 trading days apart while horizons run to 126 days, so six consecutive
  observations share most of their return window. A t-statistic that treats
  them as independent is inflated by roughly the square root of the overlap,
  and a Sharpe annualised on the sampling frequency is inflated the same way.
  Both are corrected here, and the effective number of independent draws is
  reported next to the nominal count.

The correction earns its place
------------------------------
This module previously used the plain ``mean / (sd / sqrt(n))``. On the
selection set at a 126-day horizon that read **t = 2.37**, and it was the only
result in the entire study flagged significant. The IC series has lag-1
autocorrelation of **+0.48** -- the overlap signature -- and the Newey-West
statistic is **t = 1.83**, which is not significant. Twenty-four monthly
observations of a 126-day return are worth about **four** independent draws.

The published conclusion (the signal fails confirmation) was already correct.
What was wrong was the evidence offered for the selection-set result along the
way, and the docstring above claiming discipline the arithmetic did not have.
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


def newey_west_tstat(series: pd.Series, lags: int) -> float:
    """t-statistic of the mean, with a Bartlett-kernel HAC correction.

    The plain t-statistic ``mean / (sd / sqrt(n))`` assumes the observations
    are independent. Overlapping forward returns are not: sampling monthly and
    measuring a 126-day forward return means six consecutive observations share
    most of their return window, so consecutive ICs are mechanically
    correlated (+0.48 on this data). Treating them as independent inflates the
    t-statistic by roughly the square root of the overlap.

    Newey-West is the standard correction and is what the overlapping-returns
    literature (Hansen-Hodrick, Newey-West) exists for. ``lags`` should be the
    number of observations that overlap: ``ceil(horizon / sampling_step) - 1``.
    """
    x = series.dropna().astype(float)
    n = len(x)
    if n < 3:
        return float("nan")

    deviations = (x - x.mean()).to_numpy()
    variance = float((deviations @ deviations) / n)
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        covariance = float(deviations[lag:] @ deviations[:-lag] / n)
        variance += 2.0 * weight * covariance

    if not np.isfinite(variance) or variance <= 0:
        return float("nan")
    return float(x.mean() / np.sqrt(variance / n))


def ic_statistics(ic: pd.Series, overlap_lags: int = 0, step_days: int | None = None,
                  horizon_days: int | None = None) -> dict:
    """Summary statistics for a series of per-date ICs.

    ``ic_tstat`` is the HAC-corrected statistic and is the one to read.
    ``ic_tstat_naive`` is kept alongside it precisely because the two disagree:
    on this data the naive statistic reported the only "significant" result in
    the study, and it was an artifact of overlapping return windows.
    """
    if len(ic) < 3:
        return {"n_dates": len(ic), "mean_ic": np.nan, "ic_tstat": np.nan, "hit_rate": np.nan}

    mean = float(ic.mean())
    sd = float(ic.std(ddof=1))
    naive = float(mean / (sd / np.sqrt(len(ic)))) if sd > 0 else np.nan
    tstat = newey_west_tstat(ic, overlap_lags) if overlap_lags > 0 else naive

    stats = {
        "n_dates": int(len(ic)),
        "mean_ic": round(mean, 4),
        "ic_std": round(sd, 4),
        "ic_tstat": round(tstat, 3) if np.isfinite(tstat) else np.nan,
        "ic_tstat_naive": round(naive, 3) if np.isfinite(naive) else np.nan,
        "overlap_lags": int(overlap_lags),
        "ic_autocorr_lag1": round(float(ic.autocorr(1)), 3) if len(ic) > 3 else np.nan,
        "hit_rate": round(float((ic > 0).mean()), 4),
        "significant_at_5pct": bool(abs(tstat) > 1.96) if np.isfinite(tstat) else False,
    }
    if step_days and horizon_days:
        # How many genuinely independent draws the sample is worth. Reporting
        # it stops a t-statistic computed on 24 overlapping observations from
        # reading as if it had 24 degrees of freedom -- at a 126-day horizon
        # sampled monthly, it has about 4.
        stats["n_independent"] = round(len(ic) * step_days / horizon_days, 1)
    return stats


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


def overlap_lags(horizon_days: int, step_days: int) -> int:
    """How many consecutive observations share part of a return window.

    Sampling every ``step_days`` and measuring ``horizon_days`` forward means
    each observation overlaps the next ``ceil(horizon/step) - 1`` of them.
    Zero when the sampling step is at least the horizon -- non-overlapping
    windows need no correction.
    """
    if step_days <= 0:
        return 0
    return max(int(np.ceil(horizon_days / step_days)) - 1, 0)


def evaluate(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    horizon_days: int,
    dates: pd.DatetimeIndex | None = None,
    quantile: float = 0.3,
    step_days: int = 21,
) -> dict:
    """IC and quantile-spread statistics for one signal at one horizon.

    ``step_days`` is the spacing between observation dates. It is not
    cosmetic: together with ``horizon_days`` it determines how much
    consecutive observations overlap, and therefore how much the naive
    t-statistic and Sharpe ratio overstate the evidence.
    """
    dates = dates if dates is not None else signal.index
    fwd = forward_returns(prices, pd.DatetimeIndex(dates), horizon_days)
    if fwd.empty:
        return {"horizon_days": horizon_days, "n_dates": 0, "mean_ic": np.nan}

    aligned_signal = signal.reindex(fwd.index)
    ic = information_coefficient(aligned_signal, fwd)
    spread = quantile_spread(aligned_signal, fwd, quantile)
    lags = overlap_lags(horizon_days, step_days)

    result = {
        "horizon_days": horizon_days,
        **ic_statistics(ic, overlap_lags=lags, step_days=step_days, horizon_days=horizon_days),
    }

    if len(spread) >= 3 and spread.std(ddof=1) > 0:
        # Annualise on the number of INDEPENDENT periods per year, not on the
        # sampling frequency. Sampling monthly does not give twelve independent
        # 126-day bets a year; it gives two. Scaling by 252/step would inflate
        # the Sharpe by sqrt(horizon/step) for exactly the same reason the
        # naive t-statistic is inflated.
        periods_per_year = 252 / horizon_days
        result["mean_quantile_spread"] = round(float(spread.mean()), 4)
        result["quantile_spread_sharpe"] = round(
            float(spread.mean() / spread.std(ddof=1) * np.sqrt(periods_per_year)), 3
        )
        result["spread_tstat"] = round(newey_west_tstat(spread, lags), 3) if lags else round(
            float(spread.mean() / (spread.std(ddof=1) / np.sqrt(len(spread)))), 3
        )
    else:
        result["mean_quantile_spread"] = np.nan
        result["quantile_spread_sharpe"] = np.nan
        result["spread_tstat"] = np.nan
    return result
