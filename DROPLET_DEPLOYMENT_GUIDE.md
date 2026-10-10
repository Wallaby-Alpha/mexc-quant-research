# DigitalOcean Droplet Deployment Guide: Quant Trading & Alert Suite

This guide walks through setting up a fresh DigitalOcean droplet to run all 3 quant systems simultaneously:

| System | Cadence / Engine | Telegram Alert Sample |
| :--- | :--- | :--- |
| **1. Kraken Dynamic Regime & RS Scanner** (`scanner/kraken_market_neutral_scanner.py`) | Daily at 00:01 UTC (via Cron) | `⚖️ KRAKEN DYNAMIC REGIME & RS SCANNER` |
| **2. 10-Coin Barbell Momentum Scanner** (`scanner/weekly_rotation_scanner.py`) | Weekly Mondays 00:02 UTC (via Cron) | `🎯 10-COIN BARBELL MOMENTUM SCANNER` |
| **3. WEEX SMC Order Block Liquidity Sweep Bot** (`bot/weex_ob_sweep_bot.py`) | 24/7 Continuous Daemon (every 15m candle close via Systemd) | `🚨/⚡ SMC ORDER BLOCK SWEEP` |

---

## Fast Track: 1-Click Automated Droplet Setup

### Step 1: Create a DigitalOcean Droplet
- **Image:** Ubuntu 24.04 (LTS) x64 (or 22.04 LTS).
- **Plan:** Basic ($6/month, 1GB RAM / 1 vCPU is plenty for all 3 bots).
- **Region:** London, Frankfurt, or Singapore.

### Step 2: SSH into Droplet
```bash
ssh root@<YOUR_DROPLET_IP>
```

### Step 3: Run the Turnkey Deployment Script
Run this single command on your droplet to pull the code, set up the virtual environment, install dependencies, register cron jobs, and configure the systemd service:

```bash
curl -sSL https://raw.githubusercontent.com/Wallaby-Alpha/mexc-quant-research/daily-rotation-scanner/deploy_droplet_suite.sh | bash
```
*(Or clone manually and run `./deploy_droplet_suite.sh`)*.

---

## Step 4: Configure Your Telegram Bot & WEEX Keys

After running the script, three configuration files are created in `/root/mexc-quant-research/`:

### 1. Kraken Regime Scanner Config:
```bash
nano /root/mexc-quant-research/kraken_market_neutral_config.json
```
```json
{
  "hedge_mode": "btc_hedge",
  "telegram_bot_token": "YOUR_TELEGRAM_BOT_TOKEN",
  "telegram_chat_id": "YOUR_TELEGRAM_CHAT_ID",
  "account_size_usd": 100000.0,
  "gross_exposure_pct": 25.0,
  "top_k": 3,
  "rank_exit_buffer": 6,
  "lookback_days": 7
}
```

### 2. 10-Coin Barbell Momentum Scanner Config:
```bash
nano /root/mexc-quant-research/scanner_config.json
```
```json
{
  "telegram_bot_token": "YOUR_TELEGRAM_BOT_TOKEN",
  "telegram_chat_id": "YOUR_TELEGRAM_CHAT_ID",
  "top_k": 7,
  "rank_exit_buffer": 11,
  "ranking_metric": "sharpe_momentum",
  "lookback_days": 30,
  "min_volume_24h_usdt": 500000.0
}
```

### 3. WEEX 15m Order Block Bot Config:
```bash
nano /root/mexc-quant-research/weex_bot_config.json
```
```json
{
  "api_key": "YOUR_WEEX_API_KEY",
  "secret_key": "YOUR_WEEX_SECRET_KEY",
  "passphrase": "YOUR_WEEX_PASSPHRASE",
  "dry_run": false,
  "top_universe_count": 100,
  "risk_per_trade_usd": 100.0,
  "account_size_usd": 10000.0,
  "max_concurrent_positions": 5,
  "risk_reward_ratio": 2.5,
  "atr_multiplier": 1.0,
  "telegram_bot_token": "YOUR_TELEGRAM_BOT_TOKEN",
  "telegram_chat_id": "YOUR_TELEGRAM_CHAT_ID"
}
```

---

## Step 5: Test Execution & Verify Telegram Delivery

Test both scheduled alert scanners immediately by executing them manually:

```bash
cd /root/mexc-quant-research

# 1. Test Kraken Dynamic Regime Alert:
venv/bin/python scanner/kraken_market_neutral_scanner.py

# 2. Test 10-Coin Barbell Momentum Alert:
venv/bin/python scanner/weekly_rotation_scanner.py
```
*You should receive both notifications in your Telegram channel immediately.*

---

## Step 6: Start the WEEX 24/7 Background Daemon

Start and enable the systemd service for the WEEX bot:

```bash
# Start the service
sudo systemctl start weex-bot

# Check status
sudo systemctl status weex-bot

# Watch live log stream
tail -f /root/mexc-quant-research/weex_ob_bot.log
```

---

## Ongoing Operations & Troubleshooting

- **Check Active Cron Jobs:**
  ```bash
  crontab -l
  ```
  Output shows:
  ```
  1 0 * * * cd /root/mexc-quant-research && /root/mexc-quant-research/venv/bin/python /root/mexc-quant-research/scanner/kraken_market_neutral_scanner.py >> /root/mexc-quant-research/market_neutral_scanner.log 2>&1
  2 0 * * 1 cd /root/mexc-quant-research && /root/mexc-quant-research/venv/bin/python /root/mexc-quant-research/scanner/weekly_rotation_scanner.py >> /root/mexc-quant-research/scanner.log 2>&1
  ```

- **View Logs for Each Bot:**
  - Kraken Scanner Log: `cat /root/mexc-quant-research/market_neutral_scanner.log`
  - Barbell Momentum Log: `cat /root/mexc-quant-research/scanner.log`
  - WEEX Daemon Log: `tail -f /root/mexc-quant-research/weex_ob_bot.log`

- **Updating Code from GitHub:**
  ```bash
  cd /root/mexc-quant-research
  git pull origin daily-rotation-scanner
  sudo systemctl restart weex-bot
  ```
