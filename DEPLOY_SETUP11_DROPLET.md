# DigitalOcean Droplet Deployment Guide: WEEX Setup 11 Bot

Production deployment guide for the **Setup 11 ("London Dip to NY Peak Momentum Ride")** automated trading bot on WEEX with strict **$1.00 risk sizing** and **Telegram alerting**.

---

## Strategy Overview & Operational Cadence

* **07:55 UTC (Pre-London Scan):** 
  * Scans all WEEX USDT perpetuals.
  * Discovers coins with yesterday's closed gain $\ge +10.0\%$.
  * Sizes positions to risk strictly $\le \$1.00$ with an $8.0\%$ exchange stop loss.
  * Automatically **rejects** coins if the minimum contract size risks $> \$1.00$.
  * Dispatches Morning Telegram Report.
* **08:00 UTC (London Open Entry):**
  * Submits Market Buy orders on all eligible momentum leaders.
  * Attaches native Hard Stop Loss ($-8.0\%$) directly to the WEEX matching engine.
  * Dispatches Execution Telegram Alert.
* **20:00 UTC (US Cash Close Time Exit):**
  * Market closes all open positions to capture peak intraday momentum.
  * Dispatches Daily Net P&L Summary to Telegram.

---

## Step 1: Create a DigitalOcean Droplet

1. Log in to [DigitalOcean Cloud Console](https://cloud.digitalocean.com/).
2. Click **Create** $\rightarrow$ **Droplets**.
3. Configure your droplet:
   * **Distribution:** Ubuntu 24.04 (LTS) x64 or 22.04 (LTS) x64.
   * **Plan:** Basic (Regular SSD, **$4 or $6/month** — 1 GB RAM / 1 vCPU is plenty).
   * **Datacenter Region:** Choose Frankfurt, London, or Singapore for minimal latency to crypto exchange endpoints.
   * **Authentication:** SSH Key (recommended) or root password.
4. Click **Create Droplet** and note your Droplet's IPv4 address.

---

## Step 2: Connect via SSH

Open your terminal or PowerShell and connect to your droplet:
```bash
ssh root@<YOUR_DROPLET_IP>
```

---

## Step 3: Install Server Dependencies

Run the following commands on the droplet to update packages and install Python 3, venv, and git:

```bash
# Update Ubuntu package index
sudo apt update && sudo apt upgrade -y

# Install Python 3, venv, git, and tmux
sudo apt install -y python3 python3-pip python3-venv git tmux
```

---

## Step 4: Clone Repository & Set Up Virtual Environment

```bash
# Clone the repository
git clone https://github.com/Wallaby-Alpha/mexc-quant-research.git
cd mexc-quant-research

# Checkout active branch
git checkout daily-rotation-scanner

# Create and activate Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install requests pandas numpy scipy
```

---

## Step 5: Configure API Keys & Telegram Alerts

> [!CAUTION]
> Never commit your live API keys to GitHub. `weex_london_dip_config.json` is protected by `.gitignore`.

1. Copy the example configuration template:
   ```bash
   cp bot/weex_london_dip_config.example.json weex_london_dip_config.json
   nano weex_london_dip_config.json
   ```

2. Fill in your credentials:
   ```json
   {
     "api_key": "YOUR_WEEX_API_KEY",
     "secret_key": "YOUR_WEEX_SECRET_KEY",
     "passphrase": "YOUR_WEEX_PASSPHRASE",
     "dry_run": true,
     "max_risk_usd": 1.0,
     "min_prior_day_gain_pct": 10.0,
     "max_candidates": 5,
     "stop_loss_pct": 8.0,
     "entry_time_utc": "08:00",
     "scan_time_utc": "07:55",
     "exit_time_utc": "20:00",
     "telegram_bot_token": "YOUR_TELEGRAM_BOT_TOKEN",
     "telegram_chat_id": "YOUR_TELEGRAM_CHAT_ID",
     "send_heartbeat_daily": true
   }
   ```
   *(Press `Ctrl+O` then `Enter` to save, then `Ctrl+X` to exit nano).*

> [!TIP]
> **How to get Telegram Bot credentials:**
> 1. Message `@BotFather` on Telegram and send `/newbot` to get your `telegram_bot_token`.
> 2. Message `@userinfobot` on Telegram to get your numeric `telegram_chat_id`.

---

## Step 6: Test Execution in Dry-Run Mode

Verify that API connection, exchange rules, and Telegram alerts are working properly:

```bash
# Test 1: Run the morning scanner immediately
python bot/weex_london_dip_bot.py --scan-now

# Test 2: Run an immediate simulated dry-run buy
python bot/weex_london_dip_bot.py --dry-run-now
```

Check your Telegram channel to confirm the alert arrived!

---

## Step 7: Configure 24/7 Background Service (Systemd)

To make sure the bot runs continuously and automatically restarts if the server reboots:

1. Copy the systemd service file into place:
   ```bash
   sudo cp deploy/weex-london-dip.service /etc/systemd/system/weex-london-dip.service
   ```

2. Reload systemd and enable the service on boot:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable weex-london-dip
   ```

3. Start the bot service:
   ```bash
   sudo systemctl start weex-london-dip
   ```

4. Verify service status:
   ```bash
   sudo systemctl status weex-london-dip
   ```

---

## Step 8: Monitoring & Maintenance

* **Stream Live Logs:**
  ```bash
  tail -f /root/mexc-quant-research/weex_london_dip.log
  ```
* **Restart Bot (e.g. after updating config or code):**
  ```bash
  sudo systemctl restart weex-london-dip
  ```
* **Stop Bot:**
  ```bash
  sudo systemctl stop weex-london-dip
  ```
* **Switch from Dry-Run to Live Trading:**
  ```bash
  nano weex_london_dip_config.json
  # Change "dry_run": true -> "dry_run": false
  sudo systemctl restart weex-london-dip
  ```
* **Pull Latest Updates from GitHub:**
  ```bash
  cd /root/mexc-quant-research
  git pull origin daily-rotation-scanner
  sudo systemctl restart weex-london-dip
  ```
