# Breakout Prop Trading (Kraken Futures) Weekly Momentum Scanner

Dedicated quantitative scanner tailored specifically for **Breakout Prop Trading** evaluations on **Kraken Futures**.

---

## 1. Create a Separate Telegram Bot & Channel (2 Minutes)

To keep this strategy's alerts completely separate from your MEXC bot:

1. Open Telegram and search for `@BotFather`.
2. Send: `/newbot`
3. Choose a distinct name and username:
   * Name: `Breakout Kraken Scanner`
   * Username: `my_breakout_kraken_bot` (or similar)
4. `@BotFather` will reply with your **HTTP API Token** (e.g., `7123456789:AAFl...`). Copy this.
5. Get your numeric **Chat ID**:
   * Message `@userinfobot` on Telegram and send `/start`.
   * It will reply with your numeric `Id` (e.g., `987654321`).
   * (Alternatively, create a private Telegram channel/group, invite your bot as Admin, and use that group's Chat ID).
6. Send any initial message (e.g., "start") to your new bot on Telegram so it has permission to message you.

---

## 2. Configuration Setup

Copy the example configuration to `kraken_breakout_config.json`:

```bash
cp scanner/deploy/kraken_breakout_config.example.json kraken_breakout_config.json
```

Edit `kraken_breakout_config.json`:
```json
{
  "telegram_bot_token": "YOUR_NEW_BREAKOUT_BOT_TOKEN",
  "telegram_chat_id": "YOUR_CHAT_ID",
  "account_size_usd": 100000.0,
  "total_exposure_pct": 15.0,
  "top_k": 10,
  "lookback_days": 20,
  "min_24h_volume_usd": 100000.0,
  "daily_loss_circuit_breaker_pct": 2.2
}
```

* **`account_size_usd`**: Your Breakout evaluation balance (e.g., `$10,000`, `$50,000`, `$100,000`).
* **`total_exposure_pct`**: Set to `15.0` (15% total account exposure = 1.5% position size per coin) to safely comply with Breakout's 3.0% max daily loss limit and 5%/6% max drawdown rules.
* **`daily_loss_circuit_breaker_pct`**: Alerts you to close all positions if the account experiences a -2.2% intraday drop.

---

## 3. Immediate Testing

### Step A: Test Telegram Connection
```bash
python scanner/kraken_breakout_scanner.py --test-telegram
```
You should instantly receive a confirmation message in your Telegram chat!

### Step B: Run a Live Test Scan (Dry-Run)
```bash
python scanner/kraken_breakout_scanner.py --dry-run
```
This fetches live Kraken Futures data, computes the BTC 50-day EMA, ranks the top 10 momentum coins on Kraken, and displays the exact rebalance orders without sending alerts or overwriting saved state.

### Step C: Execute Live Rebalance & Send Telegram Alert
```bash
python scanner/kraken_breakout_scanner.py --force-scan
```

---

## 4. Scheduling Automated Weekly Runs

The scanner is designed to rebalance **every Monday at 00:01 UTC**.

### Using Cron (Linux / DigitalOcean Droplet):
Run `crontab -e` and add:
```cron
1 0 * * 1 cd /path/to/repo && /usr/bin/python3 scanner/kraken_breakout_scanner.py >> /var/log/kraken_breakout.log 2>&1
```

### Running on Windows Task Scheduler:
Create a basic task triggering every Monday at 00:01 UTC running `python scanner/kraken_breakout_scanner.py`.
