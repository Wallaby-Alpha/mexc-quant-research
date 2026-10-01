"""
tests/test_phase5.py
Unit and lookahead tests for Phase 5:
- Regime classifiers causality
- Control experiment execution model consistency
- Neighbor robustness metrics
- Walk-forward rolling window generator
- Holdout evaluation protection
"""

import pytest
import pandas as pd
import numpy as np

from strategy.btc_regime import BTCRegimeDetector
from strategy.controls import ControlExperimentRunner
from backtest.execution_model import ExecutionModel
from analysis.parameter_sweep import ParameterSweepAnalyzer, compute_deflated_sharpe_ratio
from analysis.walk_forward import WalkForwardValidator


def test_btc_regime_causality():
    """Verify that BTC regime classifier strictly uses closed trailing bars."""
    # Build synthetic 1H BTC dataframe
    dates = pd.date_range("2026-01-01", periods=300, freq="1h", tz="UTC")
    df_btc = pd.DataFrame({
        "open_time": dates,
        "open": np.linspace(50000, 60000, 300),
        "high": np.linspace(50100, 60100, 300),
        "low": np.linspace(49900, 59900, 300),
        "close": np.linspace(50050, 60050, 300),
        "volume": 100.0,
        "amount": 5000000.0
    })

    class MockCache:
        def load_klines(self, sym, tf):
            return df_btc

    det = BTCRegimeDetector(cache=MockCache())
    reg_df = det.compute_btc_1h_regime()

    assert not reg_df.empty
    assert "btc_trend_bull" in reg_df.columns
    assert "macro_regime" in reg_df.columns
    assert "btc_return_24h" in reg_df.columns

    # Check that index 24 uses close[24] / close[0] - 1
    expected_ret = (df_btc["close"].iloc[24] / df_btc["close"].iloc[0]) - 1.0
    actual_ret = reg_df["btc_return_24h"].iloc[24]
    assert np.isclose(expected_ret, actual_ret, atol=1e-5)


def test_walk_forward_window_generator():
    """Verify rolling walk-forward windows strictly respect holdout boundary."""
    wf = WalkForwardValidator(
        start_date="2025-10-01T00:00:00Z",
        holdout_start="2026-07-01T00:00:00Z",
        train_months=3,
        val_months=1,
        test_months=1
    )
    windows = wf.generate_windows()

    assert len(windows) >= 4
    for w in windows:
        # Train strictly before Val, Val strictly before Test
        assert w.train_end == w.val_start
        assert w.val_end == w.test_start
        # Test strictly before holdout
        assert w.test_end <= wf.holdout_start
        assert w.train_start < w.train_end


def test_neighbor_robustness_computation():
    """Verify neighbor robustness identifies isolated peaks vs plateaus."""
    df_trials = pd.DataFrame([
        {"param_val": 1.0, "net_expectancy_r": -0.80},
        {"param_val": 1.5, "net_expectancy_r": 0.50},  # Isolated spike
        {"param_val": 2.0, "net_expectancy_r": -0.70},
        {"param_val": 3.0, "net_expectancy_r": -0.90}
    ])

    rob = ParameterSweepAnalyzer.compute_neighbor_robustness(
        df_trials=df_trials,
        param_col="param_val",
        ordered_values=[1.0, 1.5, 2.0, 3.0]
    )

    assert len(rob) == 4
    row_spike = rob[rob["param_val"] == 1.5].iloc[0]
    # Neighbors of 1.5 are 1.0 and 2.0, average = (-0.80 + -0.70) / 2 = -0.75
    assert np.isclose(row_spike["neighbor_mean_net_r"], -0.75, atol=1e-4)
    # Neighbor dropoff is 0.50 - (-0.75) = 1.25
    assert np.isclose(row_spike["neighbor_dropoff"], 1.25, atol=1e-4)
    assert bool(row_spike["is_isolated_peak"]) is True


def test_deflated_sharpe_ratio():
    """Verify Deflated Sharpe Ratio calculation penalties."""
    # When multiple trials are run and Sharpe is low, DSR probability is low
    dsr = compute_deflated_sharpe_ratio(
        observed_sr=0.1,
        n_trials=50,
        var_trials_sr=0.25,
        sample_length=500
    )
    assert 0.0 <= dsr <= 1.0
    assert dsr < 0.5  # Definitely not significant under 50 trials
