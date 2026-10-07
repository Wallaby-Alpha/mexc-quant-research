#!/bin/bash
# setup_daily_droplet.sh
# Deployment script for Daily (24h with 7d lookback) MEXC Rotation Scanner
set -e

echo "=== 1. Updating System & Installing Python ==="
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip python3-venv git curl

echo "=== 2. Setting up Project Directory ==="
mkdir -p /opt/mexc-daily-scanner
cd /opt/mexc-daily-scanner

echo "=== 3. Creating Python Virtual Environment ==="
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install requests pandas numpy

echo "=== 4. Setting up Permissions ==="
chmod +x /opt/mexc-daily-scanner/daily_rotation_scanner.py || true

echo "=== 5. Setting up Daily Cron Job (Every Day at 00:01 UTC) ==="
# Runs every day at 00:01 UTC
CRON_JOB="1 0 * * * cd /opt/mexc-daily-scanner && /opt/mexc-daily-scanner/venv/bin/python /opt/mexc-daily-scanner/daily_rotation_scanner.py >> /var/log/daily_rotation_scanner.log 2>&1"
(crontab -l 2>/dev/null | grep -Fv "daily_rotation_scanner.py" ; echo "$CRON_JOB") | crontab -

echo "=========================================================="
echo " Setup complete!"
echo " 1. Copy scanner files into /opt/mexc-daily-scanner/"
echo " 2. Copy daily_scanner_config.json with your Telegram Bot Token"
echo " 3. Test run: ./venv/bin/python daily_rotation_scanner.py"
echo "=========================================================="
