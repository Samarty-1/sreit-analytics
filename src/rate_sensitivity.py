"""How much S-REITs actually move with interest rates, measured per sub-sector.

"REITs are rate-sensitive" is the sector's most repeated claim and almost
never a number. This module produces the number, and the interesting result is
that it is not the same number for every REIT -- which is the practically
useful part, because it says which parts of the sector to hold into a rate
move and which to avoid.

Method
------
Regress each REIT's daily excess return on the daily return of the Singapore
bond fund (the SGD rate-move proxy, see ``rates.py``), controlling for the
local equity market:

    r_reit = alpha + beta_rate * r_bonds + beta_market * r_STI + e

`beta_rate` is an **empirical duration**: how much the REIT moves for a given
move in SGD rates, after stripping out whatever the Singapore equity market
did that day. Including the market term matters -- without it, `beta_rate`
picks up the general risk-on/risk-off correlation between bonds and equities
and overstates genuine rate sensitivity.

Why bond-fund returns rather than yield changes: the bond fund is a real SGD
instrument (see ``rates.py`` on why the yield *level* has to be proxied with
US Treasuries but the *changes* do not). A positive `beta_rate` means the REIT
rises when bonds rise, i.e. when rates FALL -- so a larger positive beta means
more rate-sensitive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _align(*series: pd.Series) -> pd.DataFrame:
    frame = pd.concat(series, axis=1).dropna()
    return frame


def daily_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()


def rate_beta(
    reit_prices: pd.Series,
    bond_prices: pd.Series,
    market_prices: pd.Series | None = None,
    min_observations: int = 200,
) -> dict:
    """Empirical duration of one REIT.

    Returns the rate beta, the market beta, alpha, R^2 and the t-statistic on
    the rate beta. The t-stat is reported because a beta estimated from noisy
    daily data is easy to over-read -- a beta of 0.8 with a t of 1.1 is not a
    finding.
    """
    reit = daily_returns(reit_prices).rename("reit")
    bonds = daily_returns(bond_prices).rename("bonds")

    columns = [reit, bonds]
    if market_prices is not None:
        columns.append(daily_returns(market_prices).rename("market"))

    frame = _align(*columns)
    if len(frame) < min_observations:
        return {"n": len(frame), "rate_beta": np.nan, "market_beta": np.nan,
                "alpha": np.nan, "r_squared": np.nan, "rate_beta_tstat": np.nan}

    y = frame["reit"].to_numpy()
    predictors = ["bonds"] + (["market"] if "market" in frame else [])
    X = np.column_stack([np.ones(len(frame))] + [frame[c].to_numpy() for c in predictors])

    coefficients, *_ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ coefficients
    dof = len(frame) - X.shape[1]

    sigma_squared = float(residuals @ residuals) / dof
    covariance = sigma_squared * np.linalg.pinv(X.T @ X)
    standard_errors = np.sqrt(np.diag(covariance))

    total_ss = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - float(residuals @ residuals) / total_ss if total_ss > 0 else np.nan

    return {
        "n": int(len(frame)),
        "alpha": float(coefficients[0]),
        "rate_beta": float(coefficients[1]),
        "rate_beta_tstat": float(coefficients[1] / standard_errors[1]) if standard_errors[1] > 0 else np.nan,
        "market_beta": float(coefficients[2]) if len(coefficients) > 2 else np.nan,
        "r_squared": float(r_squared),
    }


def rate_beta_table(
    reit_prices: pd.DataFrame,
    bond_prices: pd.Series,
    market_prices: pd.Series | None,
    sub_sectors: dict[str, str],
) -> pd.DataFrame:
    """Rate beta for every REIT, labelled by sub-sector."""
    rows = []
    for ticker in reit_prices.columns:
        stats = rate_beta(reit_prices[ticker].dropna(), bond_prices, market_prices)
        rows.append({"ticker": ticker, "sub_sector": sub_sectors.get(ticker, "Unknown"), **stats})
    return pd.DataFrame(rows).sort_values("rate_beta", ascending=False).reset_index(drop=True)


def sub_sector_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Mean rate beta by sub-sector, with the dispersion inside each.

    `n_reits` is worth reading before the mean: a sub-sector represented by one
    or two names is a single company's idiosyncrasy, not a sector property.
    """
    return (
        table.dropna(subset=["rate_beta"])
        .groupby("sub_sector")
        .agg(
            n_reits=("ticker", "count"),
            mean_rate_beta=("rate_beta", "mean"),
            std_rate_beta=("rate_beta", "std"),
            mean_market_beta=("market_beta", "mean"),
            mean_r_squared=("r_squared", "mean"),
        )
        .sort_values("mean_rate_beta", ascending=False)
        .round(3)
        .reset_index()
    )


def rolling_rate_beta(
    reit_prices: pd.Series,
    bond_prices: pd.Series,
    market_prices: pd.Series | None = None,
    window: int = 252,
    step: int = 21,
) -> pd.Series:
    """Rate beta through time.

    Rate sensitivity is not a constant -- it rises when rates are the dominant
    market narrative and falls when something else is. A single full-sample
    beta averages those regimes into a number that describes neither.
    """
    reit = daily_returns(reit_prices).rename("reit")
    bonds = daily_returns(bond_prices).rename("bonds")
    columns = [reit, bonds]
    if market_prices is not None:
        columns.append(daily_returns(market_prices).rename("market"))
    frame = _align(*columns)

    out = {}
    for end in range(window, len(frame) + 1, step):
        chunk = frame.iloc[end - window:end]
        y = chunk["reit"].to_numpy()
        predictors = ["bonds"] + (["market"] if "market" in chunk else [])
        X = np.column_stack([np.ones(len(chunk))] + [chunk[c].to_numpy() for c in predictors])
        coefficients, *_ = np.linalg.lstsq(X, y, rcond=None)
        out[chunk.index[-1]] = float(coefficients[1])
    return pd.Series(out).sort_index()
