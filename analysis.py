"""Run the full S-REIT analysis.

    python analysis.py --period 6y

Four sections:
  1. the universe and its current valuation metrics
  2. rate sensitivity (empirical duration) per REIT and per sub-sector
  3. distribution sustainability
  4. an honest test of whether the yield-spread signal predicts returns
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from src import metrics, rate_sensitivity, rates, signal_test
from src.universe import MARKET_PROXY, SREITS, sub_sector_of, tickers, universe_frame

warnings.filterwarnings("ignore")
REPORTS = Path(__file__).resolve().parent / "reports"
# Minimum names needed for a cross-sectional rank correlation to mean anything.
# Matches the floor in signal_test.information_coefficient.
MIN_NAMES = 8


def fetch(period: str):
    universe = tickers()
    raw = yf.download(universe + [MARKET_PROXY], period=period, group_by="ticker",
                      auto_adjust=True, progress=False, threads=True)
    prices, distributions = {}, {}
    for ticker in universe:
        if ticker not in raw.columns.get_level_values(0):
            continue
        series = raw[ticker]["Close"].dropna()
        if len(series) < 200:
            continue
        series.index = pd.to_datetime(series.index).tz_localize(None)
        prices[ticker] = series
        distributions[ticker] = yf.Ticker(ticker).dividends

    market = raw[MARKET_PROXY]["Close"].dropna()
    market.index = pd.to_datetime(market.index).tz_localize(None)
    return pd.DataFrame(prices), distributions, market


def valuation_table(prices: pd.DataFrame, distributions: dict, rf: pd.Series) -> pd.DataFrame:
    rows = []
    for ticker in prices.columns:
        px = prices[ticker].dropna()
        dy = metrics.distribution_yield(px, distributions.get(ticker, pd.Series(dtype=float)))
        spread = metrics.yield_spread(dy, rf)
        z = metrics.rolling_zscore(spread)
        rows.append({
            "ticker": ticker,
            "name": SREITS[ticker][0],
            "sub_sector": sub_sector_of(ticker),
            "price": round(float(px.iloc[-1]), 3),
            "dist_yield_pct": round(float(dy.iloc[-1]), 2) if np.isfinite(dy.iloc[-1]) else np.nan,
            "yield_spread_pp": round(float(spread.iloc[-1]), 2) if np.isfinite(spread.iloc[-1]) else np.nan,
            "spread_zscore": round(float(z.iloc[-1]), 2) if np.isfinite(z.iloc[-1]) else np.nan,
            "dist_growth_3y_pct": round(metrics.distribution_growth(
                distributions.get(ticker, pd.Series(dtype=float)), 3), 2),
            "total_return_pct": round(100 * float(px.iloc[-1] / px.iloc[0] - 1), 1),
        })
    return pd.DataFrame(rows).sort_values("spread_zscore", ascending=False).reset_index(drop=True)


def build_signal(prices: pd.DataFrame, distributions: dict, rf: pd.Series) -> pd.DataFrame:
    """Panel of yield-spread z-scores: dates x tickers."""
    columns = {}
    for ticker in prices.columns:
        px = prices[ticker].dropna()
        dy = metrics.distribution_yield(px, distributions.get(ticker, pd.Series(dtype=float)))
        columns[ticker] = metrics.rolling_zscore(metrics.yield_spread(dy, rf))
    return pd.DataFrame(columns)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--period", default="6y")
    ap.add_argument("--horizons", default="21,63,126")
    args = ap.parse_args()

    print("=" * 78)
    print("SINGAPORE REIT ANALYTICS")
    print("=" * 78)

    rf, rf_source = rates.risk_free_yield(args.period)
    rf.index = pd.to_datetime(rf.index).tz_localize(None)
    print(f"\nRisk-free source: {rf_source}")
    if rf_source.startswith("PROXY"):
        print("  Spread LEVELS carry the SGS-UST gap; z-scores and rankings are the")
        print("  outputs to trust. See src/rates.py.")

    print(f"\nFetching {args.period} of prices and distributions ...")
    prices, distributions, market = fetch(args.period)
    bonds = rates.fetch_sg_bond_fund(args.period)
    print(f"  {prices.shape[1]} REITs, {prices.shape[0]} trading days")

    print(f"\n[1] Universe ({len(SREITS)} S-REITs)")
    print(universe_frame().groupby("sub_sector").size().to_string())

    print("\n[2] Current valuation, ranked by yield-spread z-score (higher = cheaper vs bonds)")
    valuation = valuation_table(prices, distributions, rf)
    print(valuation.drop(columns=["name"]).to_string(index=False))

    print("\n[3] Rate sensitivity: empirical duration vs the SGD bond fund, market-adjusted")
    sub_sectors = {t: sub_sector_of(t) for t in prices.columns}
    betas = rate_sensitivity.rate_beta_table(prices, bonds, market, sub_sectors)
    print(betas[["ticker", "sub_sector", "rate_beta", "rate_beta_tstat",
                 "market_beta", "r_squared", "n"]].to_string(index=False))

    print("\n    By sub-sector (read n_reits before the mean):")
    print(rate_sensitivity.sub_sector_summary(betas).to_string(index=False))

    print("\n[4] Does the yield-spread z-score predict forward returns?")
    signal = build_signal(prices, distributions, rf)
    step = 21
    sampled = signal.index[::step]

    # Drop the rolling-z-score warmup before splitting, not after.
    #
    # rolling_zscore needs 252 observations, so the first ~12 sampled dates
    # carry no usable signal at all. Splitting the raw date list in half then
    # handed the selection set 12 empty dates and 24 real ones against the
    # confirmation set's 30+, and printed a selection window a year longer than
    # the one actually used. Splitting on usable dates makes the halves
    # comparable and the reported window true.
    usable = sampled[signal.reindex(sampled).notna().sum(axis=1) >= MIN_NAMES]
    dropped = len(sampled) - len(usable)
    selection, confirmation = signal_test.split_selection_confirmation(usable)
    horizons = [int(h) for h in args.horizons.split(",")]

    print(f"    {dropped} of {len(sampled)} sampled dates dropped to the z-score warmup")
    print(f"    selection:    {selection[0].date()} to {selection[-1].date()} ({len(selection)} dates)")
    print(f"    confirmation: {confirmation[0].date()} to {confirmation[-1].date()} ({len(confirmation)} dates)")
    print(f"    sampling every {step} trading days; overlap correction applied per horizon")

    print("\n    SELECTION SET (choose the horizon here, and only here)")
    selection_results = [
        signal_test.evaluate(signal, prices, h, selection, step_days=step) for h in horizons
    ]
    print(pd.DataFrame(selection_results).to_string(index=False))

    # Select on the overlap-corrected t-statistic, not on mean IC.
    #
    # Mean IC rises mechanically with horizon -- a longer window is smoother
    # and more autocorrelated -- so ranking horizons by mean IC picks the
    # longest one nearly every time, on a statistic that says nothing about
    # whether the mean is distinguishable from zero.
    best = max(
        (r for r in selection_results if np.isfinite(r.get("ic_tstat", np.nan))),
        key=lambda r: r["ic_tstat"], default=None,
    )
    if best is None:
        print("\n    No usable horizon on the selection set.")
        return

    print(f"\n    Horizon chosen on corrected t-stat: {best['horizon_days']}d "
          f"(t={best['ic_tstat']}, naive t={best.get('ic_tstat_naive')}). "
          f"{len(horizons)} horizons were tried, so even this is a 3-way search.")

    print(f"\n    CONFIRMATION SET (looked at once, horizon={best['horizon_days']}d fixed above)")
    confirmed = signal_test.evaluate(
        signal, prices, best["horizon_days"], confirmation, step_days=step
    )
    print(pd.DataFrame([confirmed]).to_string(index=False))

    verdict = (
        "signal confirmed out-of-sample"
        if confirmed.get("significant_at_5pct") and confirmed.get("mean_ic", 0) > 0
        else "NOT confirmed -- the selection-set result did not survive"
    )
    print(f"\n    VERDICT: {verdict}")

    REPORTS.mkdir(parents=True, exist_ok=True)
    valuation.to_csv(REPORTS / "valuation.csv", index=False)
    betas.to_csv(REPORTS / "rate_betas.csv", index=False)
    (REPORTS / "signal_test.json").write_text(json.dumps({
        "risk_free_source": rf_source,
        "sampling_step_days": step,
        "warmup_dates_dropped": int(dropped),
        "selection": selection_results,
        "confirmation": confirmed,
        "verdict": verdict,
    }, indent=2, default=str))
    print(f"\nSaved reports to {REPORTS}")


if __name__ == "__main__":
    main()
