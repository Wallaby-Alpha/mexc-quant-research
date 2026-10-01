import numpy as np
import pandas as pd

def run_simulation():
    np.random.seed(42)
    starting_equity = 10000.0
    risk_pct = 0.03 # 3% per trade
    n_sims = 10000

    # Scenarios based on empirical 4H backtest data (annualized):
    # 1. Standard 4H Volume Absorption (1.5x Vol, 40% Wick)
    #    N_annual ~ 65 trades, WR = 47.5%, Avg Win = +1.38R, Avg Loss = -1.00R (Net Exp = +0.084R)
    # 2. Combined Model (RS Leaders + Volume Absorption)
    #    N_annual ~ 100 trades, WR = 50.0%, Avg Win = +1.40R, Avg Loss = -1.00R (Net Exp = +0.10R)
    # 3. High Climax Absorption (2.0x Vol, 50% Wick)
    #    N_annual ~ 25 trades, WR = 63.2%, Avg Win = +1.35R, Avg Loss = -1.00R (Net Exp = +0.335R)
    # 4. Conservative Lower Bound (95% CI Lower Bound)
    #    N_annual ~ 80 trades, WR = 43.0%, Avg Win = +1.35R, Avg Loss = -1.00R (Net Exp = +0.01R)

    scenarios = {
        "Standard 4H Absorption": {"n_trades": 65, "win_rate": 0.475, "avg_win_r": 1.38, "avg_loss_r": -1.00},
        "Combined RS + Absorption": {"n_trades": 100, "win_rate": 0.500, "avg_win_r": 1.40, "avg_loss_r": -1.00},
        "High Climax Selective": {"n_trades": 25, "win_rate": 0.632, "avg_win_r": 1.35, "avg_loss_r": -1.00},
        "Conservative / Stressed": {"n_trades": 80, "win_rate": 0.430, "avg_win_r": 1.35, "avg_loss_r": -1.00},
    }

    results = []

    for name, params in scenarios.items():
        n_trades = params["n_trades"]
        wr = params["win_rate"]
        win_r = params["avg_win_r"]
        loss_r = params["avg_loss_r"]
        expected_r_per_trade = wr * win_r + (1 - wr) * loss_r

        final_equities = np.zeros(n_sims)
        max_drawdowns = np.zeros(n_sims)

        for s in range(n_sims):
            outcomes = np.random.binomial(1, wr, n_trades)
            r_multiples = np.where(outcomes == 1, win_r, loss_r)
            
            # Compounding with 3% risk
            equity_curve = np.zeros(n_trades + 1)
            equity_curve[0] = starting_equity
            curr_eq = starting_equity
            
            peak = starting_equity
            max_dd = 0.0

            for t in range(n_trades):
                # 3% of current equity risked
                trade_r = r_multiples[t]
                # PnL = equity * 0.03 * trade_r
                curr_eq = curr_eq * (1.0 + risk_pct * trade_r)
                if curr_eq <= 0:
                    curr_eq = 0
                equity_curve[t + 1] = curr_eq
                
                if curr_eq > peak:
                    peak = curr_eq
                dd = (peak - curr_eq) / peak if peak > 0 else 1.0
                if dd > max_dd:
                    max_dd = dd

            final_equities[s] = curr_eq
            max_drawdowns[s] = max_dd

        profit_pcts = (final_equities - starting_equity) / starting_equity * 100.0
        
        results.append({
            "Scenario": name,
            "Trades/Yr": n_trades,
            "Net Exp (R)": expected_r_per_trade,
            "Win Rate": f"{wr*100:.1f}%",
            "Median Ending ($)": np.median(final_equities),
            "Median Profit ($)": np.median(final_equities) - starting_equity,
            "Median Return (%)": np.median(profit_pcts),
            "10th Pct ($)": np.percentile(final_equities, 10),
            "90th Pct ($)": np.percentile(final_equities, 90),
            "Median Max DD (%)": np.median(max_drawdowns) * 100.0,
            "Worst Max DD (%)": np.percentile(max_drawdowns, 95) * 100.0,
            "Prob of Profit (%)": np.mean(final_equities > starting_equity) * 100.0,
        })

    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    run_simulation()
