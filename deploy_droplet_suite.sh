#!/bin/bash
# ==============================================================================
# deploy_droplet_suite.sh
# ------------------------------------------------------------------------------
# Turnkey DigitalOcean Droplet Deployment for Quant Trading Suite:
# 1. ⚖️ KRAKEN DYNAMIC REGIME & RS SCANNER (Daily at 00:01 UTC via Cron)
# 2. 🎯 10-COIN BARBELL MOMENTUM SCANNER (Weekly Monday 00:02 UTC via Cron)
# 3. ⚡ WEEX SMC ORDER BLOCK LIQUIDITY SWEEP BOT (24/7 15m Daemon via Systemd)
# ==============================================================================

set -e

APP_DIR="/root/mexc-quant-research"
BRANCH="daily-rotation-scanner"
REPO_URL="https://github.com/Wallaby-Alpha/mexc-quant-research.git"

echo "=================================================================="
echo "🚀 DEPLOYING QUANT TRADING SUITE ON DIGITALOCEAN DROPLET"
echo "=================================================================="

# 1. Update OS and install dependencies
echo ">>> [1/6] Updating OS packages and installing prerequisites..."
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip python3-venv git curl cron tmux

# Ensure cron service is running and enabled
sudo systemctl enable cron
sudo systemctl start cron

# 2. Clone repository if not present, or pull latest branch
echo ">>> [2/6] Syncing repository branch: ${BRANCH}..."
if [ ! -d "${APP_DIR}/.git" ]; then
    git clone -b ${BRANCH} ${REPO_URL} ${APP_DIR}
    cd ${APP_DIR}
else
    cd ${APP_DIR}
    git fetch origin ${BRANCH}
    git checkout ${BRANCH}
    git pull origin ${BRANCH}
fi

# 3. Setup Python Virtual Environment
echo ">>> [3/6] Setting up Python virtual environment..."
if [ ! -d "${APP_DIR}/venv" ]; then
    python3 -m venv ${APP_DIR}/venv
fi
source ${APP_DIR}/venv/bin/activate
pip install --upgrade pip
pip install -r ${APP_DIR}/requirements.txt

# 4. Prepare Configuration Files
echo ">>> [4/6] Initializing configuration files..."
if [ ! -f "${APP_DIR}/kraken_market_neutral_config.json" ]; then
    cp ${APP_DIR}/scanner/deploy/kraken_market_neutral_config.example.json ${APP_DIR}/kraken_market_neutral_config.json
    echo "  -> Created kraken_market_neutral_config.json"
fi

if [ ! -f "${APP_DIR}/scanner_config.json" ]; then
    cp ${APP_DIR}/scanner/deploy/scanner_config.example.json ${APP_DIR}/scanner_config.json
    echo "  -> Created scanner_config.json"
fi

if [ ! -f "${APP_DIR}/weex_bot_config.json" ]; then
    cp ${APP_DIR}/bot/weex_bot_config.example.json ${APP_DIR}/weex_bot_config.json
    echo "  -> Created weex_bot_config.json"
fi

# 5. Setup Cron Jobs for Scheduled Scanners
echo ">>> [5/6] Registering Cron Schedules..."
# Kraken Scanner: Daily at 00:01 UTC
CRON_KRAKEN="1 0 * * * cd ${APP_DIR} && ${APP_DIR}/venv/bin/python ${APP_DIR}/scanner/kraken_market_neutral_scanner.py >> ${APP_DIR}/market_neutral_scanner.log 2>&1"
# Barbell Scanner: Mondays at 00:02 UTC
CRON_BARBELL="2 0 * * 1 cd ${APP_DIR} && ${APP_DIR}/venv/bin/python ${APP_DIR}/scanner/weekly_rotation_scanner.py >> ${APP_DIR}/scanner.log 2>&1"

# Add cron jobs without duplicates
(crontab -l 2>/dev/null | grep -Fv "kraken_market_neutral_scanner.py" | grep -Fv "weekly_rotation_scanner.py" ; echo "$CRON_KRAKEN" ; echo "$CRON_BARBELL") | crontab -

# 6. Setup Systemd Service for WEEX 24/7 Daemon
echo ">>> [6/6] Configuring Systemd Service for WEEX Order Block Bot..."
SERVICE_FILE="/etc/systemd/system/weex-bot.service"

sudo bash -c "cat > ${SERVICE_FILE}" <<EOF
[Unit]
Description=WEEX 15m Order Block Liquidity Sweep Bot
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=${APP_DIR}
ExecStart=${APP_DIR}/venv/bin/python ${APP_DIR}/bot/weex_ob_sweep_bot.py
Restart=always
RestartSec=10
StandardOutput=append:${APP_DIR}/weex_ob_bot.log
StandardError=append:${APP_DIR}/weex_ob_bot.log

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable weex-bot

echo "=================================================================="
echo "✅ DEPLOYMENT SETUP COMPLETED SUCCESSFULLY!"
echo "=================================================================="
echo "Next Steps:"
echo "1. Set your Telegram Bot credentials and WEEX API keys in the configs:"
echo "   nano ${APP_DIR}/kraken_market_neutral_config.json"
echo "   nano ${APP_DIR}/scanner_config.json"
echo "   nano ${APP_DIR}/weex_bot_config.json"
echo ""
echo "2. Test each alert immediately to verify Telegram delivery:"
echo "   # Test Kraken Dynamic Regime Alert:"
echo "   ${APP_DIR}/venv/bin/python ${APP_DIR}/scanner/kraken_market_neutral_scanner.py"
echo ""
echo "   # Test 10-Coin Barbell Momentum Alert:"
echo "   ${APP_DIR}/venv/bin/python ${APP_DIR}/scanner/weekly_rotation_scanner.py"
echo ""
echo "   # Start WEEX 24/7 background service:"
echo "   sudo systemctl start weex-bot"
echo "   sudo systemctl status weex-bot"
echo "=================================================================="
