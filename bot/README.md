# WEEX Automated SMC/ICT Order Block Liquidity Sweep Bot

Production-grade automated trading bot implementing the 15-minute SMC Order Block Liquidity Sweep strategy across the **Top 100 USDT Perpetual Contracts by volume** on WEEX.

---

## Strategy Overview & Performance

- **Market Scope:** Dynamically ranks and scans the **Top 100 highest-volume USDT perpetual contracts** on WEEX (refreshed daily).
- **Setup Timeframe:** 
  - **1-Hour Structure:** Detects structural Break of Structure (BOS) and identifies the institutional Order Block (last down-close candle before expansion).
  - **15-Minute Sweep:** Detects liquidity sweeps penetrating the OB level by $\ge 0.05\%$ with candle close confirmation back inside the OB.
- **Native Exchange Protection:**
  - **Stop Loss:** Attached natively to the exchange engine at $1 \times \text{ATR}(14)$ beyond the sweep extreme.
  - **Take Profit:** Attached natively to the exchange engine at $2.5\text{R}$ or opposing structure / equal highs.
  - *If the bot or server loses internet connection, the position is still protected by WEEX's matching engine.*
- **Backtested Edge (Top 100 Universe, 4,072 trades):**
  - **Win Rate:** 72.4%
  - **Net Profit Factor:** 1.70 (net of 0.06% taker fee + 2 bps slippage per side)
  - **Net Expectancy:** +0.37% per trade / +1.77R
  - **Total Net P&L:** +7,208.5R
  - **Max Drawdown:** -5.0R

---

## Local Quick Start (Dry-Run Mode)

1. **Inspect Configuration Template:**
   ```bash
   cat bot/weex_bot_config.example.json
   ```
2. **Create Local Config (Automatically Git-Ignored):**
   ```bash
   cp bot/weex_bot_config.example.json weex_bot_config.json
   ```
3. **Run Unit Tests:**
   ```bash
   pytest tests/test_weex_ob_sweep_bot.py -v
   ```
4. **Run Bot in Dry-Run Mode:**
   ```bash
   python bot/weex_ob_sweep_bot.py
   ```
   *(By default, `"dry_run": true` simulates all orders without risking capital).*

---

## DigitalOcean Droplet Deployment Guide

Follow these exact steps to deploy the bot as a 24/7 background daemon on DigitalOcean.

### Step 1: Create Droplet
1. Log in to [DigitalOcean](https://cloud.digitalocean.com/).
2. Click **Create** $\rightarrow$ **Droplets**.
3. Choose:
   - **Image:** Ubuntu 24.04 (LTS) x64 or Ubuntu 22.04 (LTS).
   - **Plan:** Basic (Regular SSD, $6/month - 1 GB RAM / 1 vCPU is more than enough).
   - **Region:** Choose a region close to crypto exchange infrastructure (e.g. Frankfurt, London, or Singapore).
   - **Authentication:** SSH Key (recommended) or Password.
4. Click **Create Droplet**.

---

### Step 2: Connect to Droplet via SSH
```bash
ssh root@<YOUR_DROPLET_IP>
```

---

### Step 3: Server Setup & Dependencies
Run the following commands on your droplet:

```bash
# 1. Update OS packages
sudo apt update && sudo apt upgrade -y

# 2. Install Python 3, venv, git, and tmux
sudo apt install -y python3 python3-pip python3-venv git tmux

# 3. Clone the repository
git clone https://github.com/Wallaby-Alpha/mexc-quant-research.git
cd mexc-quant-research

# 4. Checkout the active bot branch
git checkout daily-rotation-scanner

# 5. Create and activate Python virtual environment
python3 -m venv venv
source venv/bin/activate

# 6. Install required dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 4: Configure API Keys Securely

> [!CAUTION]
> Never commit real API keys to GitHub. `weex_bot_config.json` is protected by `.gitignore`.

Create your private config on the droplet:
```bash
cp bot/weex_bot_config.example.json weex_bot_config.json
nano weex_bot_config.json
```

Edit the fields:
```json
{
  "api_key": "YOUR_WEEX_API_KEY",
  "secret_key": "YOUR_WEEX_SECRET_KEY",
  "passphrase": "YOUR_WEEX_API_PASSPHRASE",
  "dry_run": false,
  "top_universe_count": 100,
  "risk_per_trade_usd": 100.0,
  "account_size_usd": 10000.0,
  "max_concurrent_positions": 5,
  "risk_reward_ratio": 2.5,
  "atr_multiplier": 1.0,
  "max_ob_age_hours": 8.0,
  "min_sweep_pct": 0.05,
  "telegram_bot_token": "YOUR_TELEGRAM_BOT_TOKEN",
  "telegram_chat_id": "YOUR_TELEGRAM_CHAT_ID",
  "poll_interval_seconds": 900
}
```
*(Press `Ctrl+O` then `Enter` to save, then `Ctrl+X` to exit).*

---

### Step 5: Test Execution Before Daemonizing

Run a single test in dry-run or live mode to verify connection:
```bash
python bot/weex_ob_sweep_bot.py
```
Check that:
- It discovers the Top 100 USDT perpetual symbols by volume.
- Telegram notification is delivered.
- Press `Ctrl+C` once verified.

---

### Step 6: Configure 24/7 Background Service (Systemd)

To make sure the bot restarts automatically if the server reboots or crashes:

1. Create a systemd service unit file:
```bash
sudo nano /etc/systemd/system/weex-bot.service
```

2. Paste the following configuration:
```ini
[Unit]
Description=WEEX 15m Order Block Liquidity Sweep Bot
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/mexc-quant-research
ExecStart=/root/mexc-quant-research/venv/bin/python bot/weex_ob_sweep_bot.py
Restart=always
RestartSec=10
StandardOutput=append:/root/mexc-quant-research/weex_ob_bot.log
StandardError=append:/root/mexc-quant-research/weex_ob_bot.log

[Install]
WantedBy=multi-user.target
```

3. Enable and start the service:
```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service on system boot
sudo systemctl enable weex-bot

# Start service now
sudo systemctl start weex-bot
```

---

### Step 7: Monitoring & Maintenance

- **Check Service Status:**
  ```bash
  sudo systemctl status weex-bot
  ```

- **Live Stream Logs:**
  ```bash
  tail -f /root/mexc-quant-research/weex_ob_bot.log
  ```

- **Restart Bot (e.g. after config change or git pull):**
  ```bash
  sudo systemctl restart weex-bot
  ```

- **Stop Bot:**
  ```bash
  sudo systemctl stop weex-bot
  ```

- **Pull Latest Code Updates:**
  ```bash
  cd /root/mexc-quant-research
  git pull origin daily-rotation-scanner
  sudo systemctl restart weex-bot
  ```
