"""Tests for the REIT metrics and the signal machinery.

The ones that matter are the look-ahead controls: trailing distributions must
never include a payment that had not happened, and z-scores must never use the
full sample. Both are easy to get wrong in a way that silently manufactures a
signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import metrics, rate_sensitivity, signal_test
from src.universe import SREITS, sub_sector_of, tickers


@pytest.fixture
def dates() -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=500)


def test_trailing_distributions_never_include_a_future_payment(dates):
    """The core look-ahead control: a distribution paid tomorrow must not
    appear in today's trailing yield."""
    payment_date = dates[100]
    distributions = pd.Series([0.05], index=[payment_date])

    trailing = metrics.trailing_distributions(distributions, dates)

    assert trailing.loc[dates[99]] == 0.0
    assert trailing.loc[payment_date] == 0.05
    assert trailing.loc[dates[101]] == 0.05


def test_trailing_distributions_roll_off_after_a_year(dates):
    payment_date = dates[10]
    distributions = pd.Series([0.05], index=[payment_date])
    trailing = metrics.trailing_distributions(distributions, dates)

    within_year = payment_date + pd.Timedelta(300, unit="D")
    beyond_year = payment_date + pd.Timedelta(400, unit="D")

    assert trailing[trailing.index <= within_year].iloc[-1] == 0.05
    assert trailing[trailing.index >= beyond_year].iloc[0] == 0.0


def test_distribution_yield_is_percent(dates):
    prices = pd.Series(2.0, index=dates)
    # Both payments must fall inside the trailing year ENDING at the last
    # date, otherwise they have correctly rolled off and the yield is zero.
    distributions = pd.Series([0.05, 0.05], index=[dates[-200], dates[-20]])
    dy = metrics.distribution_yield(prices, distributions)

    # 0.10 of distributions on a 2.00 price = 5%
    assert dy.iloc[-1] == pytest.approx(5.0)


def test_distribution_yield_is_zero_once_payments_roll_off(dates):
    prices = pd.Series(2.0, index=dates)
    distributions = pd.Series([0.05], index=[dates[10]])
    assert metrics.distribution_yield(prices, distributions).iloc[-1] == pytest.approx(0.0)


def test_yield_spread_subtracts_the_risk_free_rate(dates):
    dy = pd.Series(5.5, index=dates)
    rf = pd.Series(3.0, index=dates)
    assert metrics.yield_spread(dy, rf).iloc[-1] == pytest.approx(2.5)


def test_rolling_zscore_uses_only_trailing_data(dates):
    """A full-sample z-score would know the future. Given a series that is
    flat then jumps, the z-score at the jump must be large and positive -- if
    it were standardised on the whole sample the jump would already be in the
    mean and the score would be muted."""
    values = pd.Series(1.0, index=dates)
    values.iloc[400:] = 5.0

    z = metrics.rolling_zscore(values, window=252, min_periods=100)

    assert np.isnan(z.iloc[50])         # not enough history yet
    assert z.iloc[401] > 3              # the jump is a big surprise vs trailing history


def test_rolling_zscore_is_nan_before_min_periods(dates):
    values = pd.Series(np.random.default_rng(0).normal(size=len(dates)), index=dates)
    z = metrics.rolling_zscore(values, window=252, min_periods=252)
    assert z.iloc[:251].isna().all()


def test_gearing_ratio():
    assert metrics.gearing_ratio(450, 1000) == pytest.approx(45.0)
    assert np.isnan(metrics.gearing_ratio(450, 0))
    assert np.isnan(metrics.gearing_ratio(None, 1000))


def test_distribution_growth_detects_a_cut():
    """A REIT cutting its distribution is the sector's clearest bear signal."""
    dates = pd.date_range("2020-01-01", periods=16, freq="QE")
    growing = pd.Series(np.linspace(0.02, 0.04, 16), index=dates)
    cutting = pd.Series(np.linspace(0.04, 0.02, 16), index=dates)

    assert metrics.distribution_growth(growing, 3) > 0
    assert metrics.distribution_growth(cutting, 3) < 0


def test_rate_beta_recovers_a_known_relationship():
    """Construct a REIT that is exactly 1.5x the bond fund plus noise and check
    the regression finds it."""
    rng = np.random.default_rng(0)
    n = 800
    idx = pd.bdate_range("2020-01-01", periods=n)
    bond_returns = rng.normal(0, 0.003, n)
    reit_returns = 1.5 * bond_returns + rng.normal(0, 0.0005, n)

    bonds = pd.Series(100 * np.cumprod(1 + bond_returns), index=idx)
    reit = pd.Series(100 * np.cumprod(1 + reit_returns), index=idx)

    result = rate_sensitivity.rate_beta(reit, bonds)
    assert result["rate_beta"] == pytest.approx(1.5, abs=0.1)
    assert result["rate_beta_tstat"] > 10


def test_rate_beta_near_zero_for_unrelated_series():
    rng = np.random.default_rng(1)
    n = 800
    idx = pd.bdate_range("2020-01-01", periods=n)
    bonds = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.003, n)), index=idx)
    reit = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, n)), index=idx)

    result = rate_sensitivity.rate_beta(reit, bonds)
    assert abs(result["rate_beta"]) < 0.3
    assert abs(result["rate_beta_tstat"]) < 2


def test_rate_beta_returns_nan_when_too_little_data():
    idx = pd.bdate_range("2020-01-01", periods=50)
    series = pd.Series(np.linspace(1, 2, 50), index=idx)
    assert np.isnan(rate_sensitivity.rate_beta(series, series)["rate_beta"])


def test_forward_returns_are_strictly_forward():
    """A signal observed on date d must be paired with a return measured
    entirely AFTER d, never one overlapping its own observation window."""
    idx = pd.bdate_range("2020-01-01", periods=100)
    prices = pd.DataFrame({"A": np.arange(100, 200, dtype=float)}, index=idx)

    fwd = signal_test.forward_returns(prices, pd.DatetimeIndex([idx[10]]), horizon_days=5)
    expected = prices["A"].iloc[15] / prices["A"].iloc[10] - 1

    assert fwd.iloc[0]["A"] == pytest.approx(expected)


def test_information_coefficient_detects_a_perfect_signal():
    dates = pd.DatetimeIndex(["2020-01-01", "2020-02-01"])
    columns = [f"R{i}" for i in range(10)]
    signal = pd.DataFrame([np.arange(10), np.arange(10)], index=dates, columns=columns)
    forward = pd.DataFrame([np.arange(10), np.arange(10)], index=dates, columns=columns)

    ic = signal_test.information_coefficient(signal, forward)
    assert ic.tolist() == pytest.approx([1.0, 1.0])


def test_information_coefficient_detects_an_inverted_signal():
    dates = pd.DatetimeIndex(["2020-01-01"])
    columns = [f"R{i}" for i in range(10)]
    signal = pd.DataFrame([np.arange(10)], index=dates, columns=columns)
    forward = pd.DataFrame([np.arange(10)[::-1]], index=dates, columns=columns)

    assert signal_test.information_coefficient(signal, forward).iloc[0] == pytest.approx(-1.0)


def test_ic_statistics_flags_insignificance():
    noise = pd.Series(np.random.default_rng(3).normal(0, 0.3, 40))
    stats = signal_test.ic_statistics(noise)
    assert not stats["significant_at_5pct"]


def test_selection_confirmation_split_is_chronological():
    dates = pd.bdate_range("2020-01-01", periods=100)
    selection, confirmation = signal_test.split_selection_confirmation(dates)

    assert len(selection) == 50 and len(confirmation) == 50
    assert selection.max() < confirmation.min(), "confirmation must be strictly later"


def test_universe_is_internally_consistent():
    assert len(tickers()) == len(SREITS)
    assert all(sub_sector_of(t) for t in tickers())
    assert all(t.endswith(".SI") for t in tickers())


def test_keppel_dc_is_labelled_data_centre():
    """Yahoo Finance classifies Keppel DC REIT as 'REIT - Office'. It is a pure
    data-centre REIT, and sub-sector is the axis every comparison here cuts
    along, so inheriting the vendor's error would corrupt the results."""
    assert sub_sector_of("AJBU.SI") == "Data Centre"
