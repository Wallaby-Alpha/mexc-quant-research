"""
research_market_neutral_kraken_breakout.py
------------------------------------------
Empirical Backtest: Market-Neutral Long/Short Relative Strength
on the Kraken Breakout Universe (65 Assets).

Models Evaluated:
1. Benchmark: Long-Only Rotational Momentum (Top 5 Coins, 20% exposure, no hedge).
2. Model A1: Cross-Sectional L/S (Long Top 3 Leaders / Short Bottom 3 Laggards).
3. Model A2: Alt-Leader Alpha vs. BTC Hedge (Long Top 3 Leaders / Short BTC).
4. Model A3: Beta-Hedged Alpha (Long Top 3 Leaders / Short 1.35x Beta BTC).
5. Model A4: Regime-Switching (Long-Only when BTC > 50 EMA; Market-Neutral L/S when BTC <= 50 EMA).

Evaluates Prop Firm Survival Metrics:
- Max Trailing Drawdown (Must stay < 6% - 8%)
- Worst Single-Day Drop (Must stay < 3% - 4%)
- Probability of Passing Prop Evaluation (+8% to +10% target)
- Sharpe Ratio & Profit Factor
- Net of realistic exchange fees (0.05% - 0.06% taker per leg)
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scanner.daily_rotation_scanner import DEFAULT_BREAKOUT_UNIVERSE

def load_universe_daily_data():
    """Loads all available Kraken Breakout coins from multi-year daily data."""
    data_dir = Path("data_cache/klines_multiyear_1d")
    dfs = {}
    
    # Load BTC first
    df_btc = pd.read_parquet(data_dir / "BTCUSDT.parquet")
    df_btc["datetime"] = pd.to_datetime(df_btc["open_time"], unit="ms")
    df_btc = df_btc.sort_values("datetime").reset_index(drop=True)
    df_btc["close"] = df_btc["close"].astype(float)
    dfs["BTC"] = df_btc.set_index("datetime")["close"]
    
    for coin in DEFAULT_BREAKOUT_UNIVERSE:
        p = data_dir / f"{coin}USDT.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
            df = df.sort_values("datetime").reset_index(drop=True)
            df["close"] = df["close"].astype(float)
            dfs[coin] = df.set_index("datetime")["close"]
            
    # Combine into a single price matrix
    price_df = pd.DataFrame(dfs).sort_index()
    return price_df

def run_backtests(price_df: pd.DataFrame, rebalance_cadence_days: int = 1, lookback_days: int = 7):
    # Fee: 0.06% taker per leg (0.12% roundtrip)
    fee_per_turn = 0.0006
    
    # Compute daily returns
    daily_rets = price_df.pct_change()
    
    # Precompute rolling 7-day Sharpe (mean return / std)
    roll_mean = daily_rets.rolling(lookback_days).mean()
    roll_std = daily_rets.rolling(lookback_days).std() + 1e-9
    scores = roll_mean / roll_std
    
    # BTC 50 EMA for regime detection
    btc_ema50 = price_df["BTC"].ewm(span=50, adjust=False).mean()
    btc_bull = price_df["BTC"] > btc_ema50
    
    dates = price_df.index[max(lookback_days + 50, 100):]
    
    # Sizing for prop firms: 20% Gross Exposure (10% Long, 10% Short for L/S; or 20% Long for Long-Only)
    gross_exposure = 0.20
    half_exposure = gross_exposure / 2.0 # 10% Long, 10% Short
    
    records = {
        "Benchmark: Long-Only Top 5": [],
        "Model A1: Top 3 Long / Bottom 3 Short": [],
        "Model A2: Top 3 Long / Short BTC": [],
        "Model A3: Top 3 Long / Beta-Hedged BTC (1.35x)": [],
        "Model A4: Regime-Switching (Long in Bull, L/S in Bear)": []
    }
    
    cur_longs_bmark = []
    cur_longs_a1 = []
    cur_shorts_a1 = []
    cur_longs_a2 = []
    cur_longs_a4 = []
    cur_shorts_a4 = []
    
    for i in range(1, len(dates)):
        decision_date = dates[i-1] # closed data at t-1
        exec_date = dates[i]       # trade outcome at t
        
        # Get universe available on decision_date with at least lookback days
        valid_coins = [c for c in price_df.columns if c != "BTC" and not np.isnan(scores.loc[decision_date, c])]
        if len(valid_coins) < 10:
            for k in records:
                records[k].append(0.0)
            continue
            
        coin_scores = scores.loc[decision_date, valid_coins].sort_values(ascending=False)
        top5 = coin_scores.head(5).index.tolist()
        top3 = coin_scores.head(3).index.tolist()
        bot3 = coin_scores.tail(3).index.tolist()
        
        is_bull = btc_bull.loc[decision_date]
        
        # Helper to compute robust mean
        def safe_mean(coin_list):
            vals = [daily_rets.loc[exec_date, c] for c in coin_list if c in daily_rets.columns and not np.isnan(daily_rets.loc[exec_date, c])]
            return np.mean(vals) if vals else 0.0

        # 1. Benchmark: Long-Only Top 5 (20% total exposure = 4% each)
        r_bmark = safe_mean(top5) * gross_exposure
        # Turnover fee
        turnover_bmark = len(set(top5) - set(cur_longs_bmark)) / 5.0
        fee_bmark = turnover_bmark * gross_exposure * fee_per_turn
        cur_longs_bmark = top5
        records["Benchmark: Long-Only Top 5"].append(r_bmark - fee_bmark)
        
        # 2. Model A1: Top 3 Long (10% total) / Bottom 3 Short (10% total)
        r_long_a1 = safe_mean(top3) * half_exposure
        r_short_a1 = -safe_mean(bot3) * half_exposure
        turnover_a1 = (len(set(top3) - set(cur_longs_a1)) + len(set(bot3) - set(cur_shorts_a1))) / 6.0
        fee_a1 = turnover_a1 * gross_exposure * fee_per_turn
        cur_longs_a1, cur_shorts_a1 = top3, bot3
        records["Model A1: Top 3 Long / Bottom 3 Short"].append((r_long_a1 + r_short_a1) - fee_a1)
        
        # 3. Model A2: Top 3 Long (10% total) / Short BTC (10% total)
        r_long_a2 = safe_mean(top3) * half_exposure
        btc_ret = daily_rets.loc[exec_date, "BTC"] if not np.isnan(daily_rets.loc[exec_date, "BTC"]) else 0.0
        r_short_btc = -btc_ret * half_exposure
        turnover_a2 = len(set(top3) - set(cur_longs_a2)) / 3.0
        fee_a2 = turnover_a2 * half_exposure * fee_per_turn
        cur_longs_a2 = top3
        records["Model A2: Top 3 Long / Short BTC"].append((r_long_a2 + r_short_btc) - fee_a2)
        
        # 4. Model A3: Top 3 Long (10% total) / Short Beta-Weighted BTC (13.5% short)
        r_short_btc_beta = -btc_ret * (half_exposure * 1.35)
        records["Model A3: Top 3 Long / Beta-Hedged BTC (1.35x)"].append((r_long_a2 + r_short_btc_beta) - fee_a2)
        
        # 5. Model A4: Regime-Switching (Bull -> Long-Only Top 3; Bear -> L/S Top 3 vs Bot 3)
        if is_bull:
            r_a4 = safe_mean(top3) * gross_exposure
            turnover_a4 = len(set(top3) - set(cur_longs_a4)) / 3.0
            fee_a4 = turnover_a4 * gross_exposure * fee_per_turn
            cur_longs_a4, cur_shorts_a4 = top3, []
        else:
            r_long_a4 = safe_mean(top3) * half_exposure
            r_short_a4 = -safe_mean(bot3) * half_exposure
            r_a4 = r_long_a4 + r_short_a4
            turnover_a4 = (len(set(top3) - set(cur_longs_a4)) + len(set(bot3) - set(cur_shorts_a4))) / 6.0
            fee_a4 = turnover_a4 * gross_exposure * fee_per_turn
            cur_longs_a4, cur_shorts_a4 = top3, bot3
        records["Model A4: Regime-Switching (Long in Bull, L/S in Bear)"].append(r_a4 - fee_a4)
        
    return pd.DataFrame(records, index=dates[1:])

def evaluate_prop_metrics(df_rets: pd.DataFrame):
    results = []
    
    for col in df_rets.columns:
        rets = df_rets[col].fillna(0.0)
        cum_equity = (1.0 + rets).cumprod()
        peak = cum_equity.cummax()
        dd = (cum_equity - peak) / peak
        max_dd = dd.min() * 100.0
        
        total_ret = (cum_equity.iloc[-1] - 1.0) * 100.0
        ann_ret = ((cum_equity.iloc[-1]) ** (365.0 / len(rets)) - 1.0) * 100.0 if cum_equity.iloc[-1] > 0 else -100.0
        
        daily_std = rets.std()
        sharpe = (rets.mean() / (daily_std + 1e-9)) * np.sqrt(365)
        
        # Daily drawdown distribution
        worst_day = rets.min() * 100.0
        p99_loss = rets.quantile(0.01) * 100.0
        p95_loss = rets.quantile(0.05) * 100.0
        
        # Prop firm pass probability: Probability of reaching +8% profit target before hitting -6% max drawdown
        passes = 0
        fails = 0
        rolling_windows = 60 # 60-day evaluation window
        for start_idx in range(0, len(rets) - rolling_windows, 5):
            window_rets = rets.iloc[start_idx : start_idx + rolling_windows]
            w_equity = (1.0 + window_rets).cumprod()
            w_peak = w_equity.cummax()
            w_dd = (w_equity - w_peak) / w_peak
            
            # Did it hit -6% max drawdown first, or +8% target first?
            hit_fail = (w_dd <= -0.06).any()
            hit_pass = (w_equity >= 1.08).any()
            
            if hit_pass and not hit_fail:
                passes += 1
            elif hit_fail:
                fails += 1
                
        total_evals = passes + fails
        pass_rate = (passes / total_evals * 100.0) if total_evals > 0 else 0.0
        
        wins = rets[rets > 0].sum()
        losses = abs(rets[rets < 0].sum())
        profit_factor = wins / losses if losses > 0 else np.nan
        
        results.append({
            "Strategy": col,
            "Total Net Return": f"{total_ret:+.1f}%",
            "Annualized Return": f"{ann_ret:+.1f}%",
            "Max Trailing DD": f"{max_dd:.2f}%",
            "Worst Single Day": f"{worst_day:.2f}%",
            "99th Percentile Day Loss": f"{p99_loss:.2f}%",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Profit Factor": f"{profit_factor:.2f}",
            "Prop Pass Rate": f"{pass_rate:.1f}%"
        })
        
    return pd.DataFrame(results)

def main():
    print("=" * 95)
    print("BACKTEST: MARKET-NEUTRAL LONG/SHORT MOMENTUM ON KRAKEN BREAKOUT UNIVERSE")
    print("=" * 95)
    
    price_df = load_universe_daily_data()
    print(f"Total Coins Loaded: {len(price_df.columns)}")
    print(f"Date Range: {price_df.index.min().date()} to {price_df.index.max().date()} ({len(price_df)} days)\n")
    
    rets_df = run_backtests(price_df)
    results_summary = evaluate_prop_metrics(rets_df)
    
    print(results_summary.to_string(index=False))
    
    # Analyze correlation to Bitcoin
    print("\n" + "=" * 95)
    print("CORRELATION TO BITCOIN DAILY MOVES")
    print("=" * 95)
    btc_rets = price_df["BTC"].pct_change().loc[rets_df.index]
    for col in rets_df.columns:
        corr = rets_df[col].corr(btc_rets)
        print(f"  • {col:<50}: Correlation to BTC = {corr:+.2f}")

if __name__ == "__main__":
    main()
