#!/bin/bash
# setup_droplet.sh
# One-click setup script for DigitalOcean Ubuntu Droplet
set -e

echo "=== 1. Updating System & Installing Python ==="
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip python3-venv git curl

echo "=== 2. Setting up Project Directory ==="
mkdir -p /opt/mexc-rotation-scanner
cd /opt/mexc-rotation-scanner

echo "=== 3. Creating Python Virtual Environment ==="
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install requests pandas numpy

echo "=== 4. Setting up Permissions ==="
chmod +x /opt/mexc-rotation-scanner/weekly_rotation_scanner.py || true

echo "=== 5. Setting up Weekly Cron Job (Every Monday at 00:01 UTC) ==="
# Runs every Monday at 00:01 UTC
CRON_JOB="1 0 * * 1 cd /opt/mexc-rotation-scanner && /opt/mexc-rotation-scanner/venv/bin/python /opt/mexc-rotation-scanner/weekly_rotation_scanner.py >> /var/log/rotation_scanner.log 2>&1"
(crontab -l 2>/dev/null | grep -Fv "weekly_rotation_scanner.py" ; echo "$CRON_JOB") | crontab -

echo "=========================================================="
echo " Setup complete!"
echo " 1. Copy your scanner files into /opt/mexc-rotation-scanner/"
echo " 2. Copy scanner_config.json with your Telegram Bot Token"
echo " 3. Test immediately with: ./venv/bin/python weekly_rotation_scanner.py"
echo "=========================================================="
