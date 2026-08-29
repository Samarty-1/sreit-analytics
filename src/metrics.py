"""REIT-specific metrics. None of these are technical indicators.

A REIT is a pass-through vehicle: it must distribute most of its income, it
carries a lot of debt against long-lived assets, and it trades against its
book value in a way an operating company does not. So the metrics that matter
are the ones below -- distribution yield, the spread of that yield over the
risk-free rate, gearing, and price to book -- not moving averages.

The one that drives everything is the **yield spread**. A REIT yielding 5.5%
when the government bond yields 3.0% offers 250bp of compensation for taking
property and leverage risk. Whether that is cheap or expensive is a question
about the spread's own history, not about the yield level: a 5.5% yield is
generous against a 2% risk-free rate and thin against a 5% one. Every
valuation signal here is built on the spread, and z-scored against its own
trailing history so the answer adapts as the rate regime moves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def trailing_distributions(distributions: pd.Series, dates: pd.DatetimeIndex, window_days: int = 365) -> pd.Series:
    """Sum of distributions paid in the trailing `window_days` at each date.

    Uses only distributions with an ex-date on or before each observation
    date, so the series never contains a payment that had not happened yet.
    A rolling sum on a forward-filled series is the usual way this goes wrong.
    """
    if distributions.empty:
        return pd.Series(0.0, index=dates)

    dist = distributions.copy()
    dist.index = pd.to_datetime(dist.index).tz_localize(None)
    dist = dist.sort_index()

    values = []
    for date in dates:
        window_start = date - pd.Timedelta(window_days, unit="D")
        values.append(float(dist[(dist.index > window_start) & (dist.index <= date)].sum()))
    return pd.Series(values, index=dates)


def distribution_yield(prices: pd.Series, distributions: pd.Series, window_days: int = 365) -> pd.Series:
    """Trailing-twelve-month distribution yield, in percent.

    Note this is a TRAILING yield, not the forward yield a broker quotes. It
    is backward-looking by construction, which is the right choice here: a
    forward yield embeds an analyst's estimate of a future distribution, and
    using one in a historical study would import a forecast that was not
    available at the time.
    """
    trailing = trailing_distributions(distributions, prices.index, window_days)
    return (100.0 * trailing / prices).replace([np.inf, -np.inf], np.nan)


def yield_spread(dist_yield: pd.Series, risk_free_yield: pd.Series) -> pd.Series:
    """Distribution yield minus the risk-free yield, in percentage points."""
    aligned_rf = risk_free_yield.reindex(dist_yield.index).ffill()
    return dist_yield - aligned_rf


def rolling_zscore(series: pd.Series, window: int = TRADING_DAYS * 2, min_periods: int = TRADING_DAYS) -> pd.Series:
    """Z-score against the series' OWN trailing history.

    Expanding/rolling rather than full-sample: a full-sample z-score would use
    the mean and standard deviation of the entire history including the
    future, which is the most common way a "valuation signal" ends up
    accidentally clairvoyant. A positive z-score here means the spread is
    wider than it has recently been -- the REIT is cheap relative to bonds.
    """
    mean = series.rolling(window, min_periods=min_periods).mean()
    sd = series.rolling(window, min_periods=min_periods).std()
    return ((series - mean) / sd.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def gearing_ratio(total_debt: float | None, total_assets: float | None) -> float:
    """Debt / assets, in percent -- the metric MAS actually caps.

    Singapore REITs operate under a regulatory leverage ceiling (45%, or 50%
    subject to an interest-coverage test), so gearing is not just a risk
    measure here: a REIT near the cap has no debt headroom to fund
    acquisitions and is a candidate to raise equity, which dilutes existing
    holders. It is a structural constraint with a direct valuation
    consequence.
    """
    if not total_debt or not total_assets or total_assets <= 0:
        return float("nan")
    return 100.0 * total_debt / total_assets


def price_to_book(price: float | None, book_value_per_unit: float | None) -> float:
    """P/B. For a REIT this is price to NAV -- below 1.0 means the market
    values the portfolio at less than the appraised value of the buildings."""
    if not price or not book_value_per_unit or book_value_per_unit <= 0:
        return float("nan")
    return price / book_value_per_unit


def annualized_metrics(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict:
    r = returns.dropna()
    if len(r) < 2:
        return {"n": len(r), "total_return": 0.0, "annualized_return": 0.0, "annualized_vol": 0.0,
                "sharpe": 0.0, "max_drawdown": 0.0}

    cumulative = (1 + r).cumprod()
    total = float(cumulative.iloc[-1] - 1)
    vol = float(r.std(ddof=1) * np.sqrt(periods_per_year))
    drawdown = float((cumulative / cumulative.cummax() - 1).min())
    return {
        "n": int(len(r)),
        "total_return": round(total, 4),
        "annualized_return": round(float((1 + total) ** (periods_per_year / len(r)) - 1), 4),
        "annualized_vol": round(vol, 4),
        "sharpe": round(float(r.mean() * periods_per_year / vol), 4) if vol > 0 else 0.0,
        "max_drawdown": round(drawdown, 4),
    }


def distribution_growth(distributions: pd.Series, years: int = 3) -> float:
    """Compound annual growth of the trailing-year distribution over `years`.

    A REIT cutting its distribution is the sector's clearest bear signal --
    it means either the assets are underperforming or the debt is squeezing
    the payout. Returns NaN when there is not enough history rather than
    extrapolating from a short window.
    """
    if distributions.empty:
        return float("nan")

    dist = distributions.copy()
    dist.index = pd.to_datetime(dist.index).tz_localize(None)
    end = dist.index.max()
    start = end - pd.DateOffset(years=years)

    recent = dist[(dist.index > end - pd.DateOffset(years=1))].sum()
    old = dist[(dist.index > start - pd.DateOffset(years=1)) & (dist.index <= start)].sum()
    if old <= 0 or recent <= 0:
        return float("nan")
    return float(100.0 * ((recent / old) ** (1 / years) - 1))
