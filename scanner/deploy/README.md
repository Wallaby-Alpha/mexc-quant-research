# MEXC Weekly Rotational Momentum Scanner Deployment Guide

Deploy your automated quantitative crypto scanner to a **DigitalOcean Droplet ($4 - $6/month)** and receive push notifications on your phone via Telegram every **Monday at 00:00 UTC**.

---

## 1. Create Your Telegram Bot (Takes 2 Minutes)

1. Open Telegram and search for `@BotFather`.
2. Send the message: `/newbot`
3. Choose a name (e.g., `My Crypto Rotation Scanner`) and username (e.g., `my_rotation_bot`).
4. `@BotFather` will reply with your **HTTP API Token** (e.g., `7123456789:AAFl...`). Copy this token.
5. Get your **Chat ID**:
   - Search for `@userinfobot` on Telegram and send `/start`.
   - It will reply with your numeric `Id` (e.g., `987654321`). Copy this ID.
   - Send any test message (e.g., "hello") to your new bot so it has permission to message you.

---

## 2. Launch a DigitalOcean Droplet

1. Log into your [DigitalOcean Dashboard](https://cloud.digitalocean.com).
2. Click **Create** > **Droplets**.
3. Choose:
   - **OS:** Ubuntu 24.04 LTS (x64)
   - **Plan:** Basic ($4/month or $6/month droplet is plenty)
   - **Region:** Any region (e.g., NYC, Frankfurt, Singapore)
   - **Authentication:** SSH Key or Root Password
4. Click **Create Droplet**.

---

## 3. Deploy the Scanner onto the Droplet

SSH into your droplet from your local terminal:
```bash
ssh root@YOUR_DROPLET_IP
```

### Step A: Run the Setup Script
```bash
# Clone or copy your scanner folder into /opt/mexc-rotation-scanner
mkdir -p /opt/mexc-rotation-scanner
cd /opt/mexc-rotation-scanner

# Download setup script
curl -sSL https://raw.githubusercontent.com/.../setup_droplet.sh -o setup_droplet.sh 
# (Or simply copy setup_droplet.sh and weekly_rotation_scanner.py over using SCP or Git)
bash setup_droplet.sh
```

### Step B: Configure Telegram Credentials
Create `scanner_config.json`:
```bash
nano /opt/mexc-rotation-scanner/scanner_config.json
```
Paste:
```json
{
  "telegram_bot_token": "YOUR_ACTUAL_BOT_TOKEN_FROM_BOTFATHER",
  "telegram_chat_id": "YOUR_ACTUAL_NUMERIC_CHAT_ID",
  "top_k": 10,
  "rank_exit_buffer": 15,
  "lookback_days": 30,
  "min_volume_24h_usdt": 500000.0
}
```
Press `Ctrl + O`, `Enter`, then `Ctrl + X` to save.

---

## 4. Test the Scanner Immediately

Run a live test to verify Telegram messages are arriving:
```bash
cd /opt/mexc-rotation-scanner
./venv/bin/python weekly_rotation_scanner.py
```

You will see:
1. BTC 50-day EMA calculation.
2. Top 150 MEXC coins downloaded and ranked by 30-day return relative to Bitcoin.
3. A push notification on your Telegram with exact **BUY / SELL / HOLD** orders!

---

## 5. Automated Weekly Schedule (Cron)

The setup script automatically registers a cron job that runs **every Monday at 00:01 UTC**:
```bash
# Verify the cron job is active:
crontab -l
```
You will see:
```cron
1 0 * * 1 cd /opt/mexc-rotation-scanner && /opt/mexc-rotation-scanner/venv/bin/python /opt/mexc-rotation-scanner/weekly_rotation_scanner.py >> /var/log/rotation_scanner.log 2>&1
```

To view past logs at any time:
```bash
cat /var/log/rotation_scanner.log
```
