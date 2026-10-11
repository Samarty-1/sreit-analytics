"""Tests for the overlapping-window statistics.

Written as regressions: the plain t-statistic these replace produced the only
"significant" result in the published study, and it was an artifact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import signal_test


class TestOverlapLags:
    @pytest.mark.parametrize("horizon, step, expected", [
        (21, 21, 0),    # non-overlapping: one observation per window
        (63, 21, 2),    # three windows in flight at once
        (126, 21, 5),
        (5, 21, 0),     # horizon shorter than the step: still no overlap
        (252, 21, 11),
    ])
    def test_counts_windows_in_flight(self, horizon, step, expected):
        assert signal_test.overlap_lags(horizon, step) == expected

    def test_zero_step_is_handled(self):
        assert signal_test.overlap_lags(126, 0) == 0


class TestNeweyWestTstat:
    def test_matches_the_plain_tstat_at_zero_lags(self):
        rng = np.random.default_rng(0)
        x = pd.Series(rng.normal(0.5, 1.0, 200))
        plain = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
        assert signal_test.newey_west_tstat(x, 0) == pytest.approx(plain, rel=0.02)

    def test_deflates_a_positively_autocorrelated_series(self):
        """The whole point: overlapping windows inflate the naive statistic."""
        rng = np.random.default_rng(1)
        noise = rng.normal(0, 1, 400)
        # A rolling mean is exactly the structure an overlapping forward return
        # induces -- consecutive values share most of their inputs.
        overlapping = pd.Series(noise).rolling(6).mean().dropna() + 0.15

        naive = overlapping.mean() / (overlapping.std(ddof=1) / np.sqrt(len(overlapping)))
        corrected = signal_test.newey_west_tstat(overlapping, 5)

        assert overlapping.autocorr(1) > 0.5, "precondition: the series overlaps"
        assert abs(corrected) < abs(naive)

    def test_does_not_deflate_an_independent_series(self):
        rng = np.random.default_rng(2)
        x = pd.Series(rng.normal(0.2, 1.0, 400))
        naive = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
        corrected = signal_test.newey_west_tstat(x, 5)
        assert corrected == pytest.approx(naive, rel=0.25)

    def test_too_few_observations_is_nan(self):
        assert np.isnan(signal_test.newey_west_tstat(pd.Series([1.0, 2.0]), 1))


class TestIcStatistics:
    def test_reports_both_statistics_and_uses_the_corrected_one(self):
        rng = np.random.default_rng(3)
        ic = pd.Series(rng.normal(0, 1, 300)).rolling(6).mean().dropna() + 0.2
        stats = signal_test.ic_statistics(ic, overlap_lags=5, step_days=21, horizon_days=126)

        assert abs(stats["ic_tstat"]) < abs(stats["ic_tstat_naive"])
        assert stats["overlap_lags"] == 5
        assert stats["ic_autocorr_lag1"] > 0.5
        # Significance must be decided on the corrected statistic.
        assert stats["significant_at_5pct"] == (abs(stats["ic_tstat"]) > 1.96)

    def test_reports_effective_independent_draws(self):
        ic = pd.Series(np.linspace(-0.1, 0.3, 24))
        stats = signal_test.ic_statistics(ic, overlap_lags=5, step_days=21, horizon_days=126)
        assert stats["n_dates"] == 24
        assert stats["n_independent"] == pytest.approx(4.0)

    def test_no_correction_when_windows_do_not_overlap(self):
        rng = np.random.default_rng(4)
        ic = pd.Series(rng.normal(0.05, 0.2, 30))
        stats = signal_test.ic_statistics(ic, overlap_lags=0, step_days=21, horizon_days=21)
        assert stats["ic_tstat"] == stats["ic_tstat_naive"]
        assert stats["n_independent"] == pytest.approx(30.0)


class TestEvaluateAnnualisation:
    """The quantile-spread Sharpe must annualise on independent periods."""

    def _panel(self, n_days=800, n_names=12, seed=5):
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2019-01-01", periods=n_days)
        names = [f"R{i}" for i in range(n_names)]
        prices = pd.DataFrame(
            100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n_days, n_names)), axis=0)),
            index=dates, columns=names,
        )
        signal = pd.DataFrame(rng.normal(0, 1, (n_days, n_names)), index=dates, columns=names)
        return signal, prices

    def test_longer_horizons_do_not_get_a_frequency_bonus(self):
        signal, prices = self._panel()
        dates = signal.index[::21]
        result = signal_test.evaluate(signal, prices, 126, dates, step_days=21)

        assert result["overlap_lags"] == 5
        # Counted off the ICs actually produced, not the dates offered: the
        # last few dates have no room left for a 126-day forward window.
        assert result["n_dates"] < len(dates)
        assert result["n_independent"] == pytest.approx(result["n_dates"] * 21 / 126, rel=0.02)
        assert np.isfinite(result["quantile_spread_sharpe"])

    def test_step_days_flows_into_the_overlap_correction(self):
        signal, prices = self._panel()
        dates = signal.index[::21]
        overlapping = signal_test.evaluate(signal, prices, 126, dates, step_days=21)
        # Claiming a 126-day step would assert the windows do not overlap.
        assert overlapping["overlap_lags"] == 5
        independent = signal_test.evaluate(signal, prices, 126, dates, step_days=126)
        assert independent["overlap_lags"] == 0
        assert independent["ic_tstat"] == independent["ic_tstat_naive"]
