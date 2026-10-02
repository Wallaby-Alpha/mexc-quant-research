#!/bin/bash
# setup_kraken_droplet.sh
# One-click setup script for Breakout Kraken Weekly Rotation Scanner on DigitalOcean Droplet
set -e

INSTALL_DIR="/opt/kraken-breakout-scanner"

echo "=========================================================="
echo " 🚀 Setting up Breakout Kraken Prop Scanner on Droplet"
echo "=========================================================="

echo "=== 1. Updating System & Installing Prerequisites ==="
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip python3-venv git curl

echo "=== 2. Setting up Directory at ${INSTALL_DIR} ==="
sudo mkdir -p ${INSTALL_DIR}
sudo chown -R $USER:$USER ${INSTALL_DIR}
cd ${INSTALL_DIR}

echo "=== 3. Pulling kraken-breakout-rotation from GitHub ==="
if [ ! -d ".git" ]; then
    git clone -b kraken-breakout-rotation https://github.com/Wallaby-Alpha/mexc-quant-research.git .
else
    git fetch origin kraken-breakout-rotation
    git checkout kraken-breakout-rotation
    git pull origin kraken-breakout-rotation
fi

echo "=== 4. Creating Python Virtual Environment ==="
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install requests pandas numpy

echo "=== 5. Setting up Configuration Template ==="
if [ ! -f "kraken_breakout_config.json" ]; then
    cp scanner/deploy/kraken_breakout_config.example.json kraken_breakout_config.json
    echo "Created kraken_breakout_config.json. Please edit it with your Telegram credentials."
fi

echo "=== 6. Configuring Cron Schedule (Every Monday at 00:01 UTC) ==="
CRON_CMD="1 0 * * 1 cd ${INSTALL_DIR} && ${INSTALL_DIR}/venv/bin/python ${INSTALL_DIR}/scanner/kraken_breakout_scanner.py >> /var/log/kraken_breakout_scanner.log 2>&1"
# Add cron job while ensuring no duplicate entries
(crontab -l 2>/dev/null | grep -Fv "kraken_breakout_scanner.py" ; echo "$CRON_CMD") | crontab -

echo "=========================================================="
echo " ✅ Setup Complete!"
echo " Next Steps:"
echo " 1. Edit your Telegram bot credentials:"
echo "    nano ${INSTALL_DIR}/kraken_breakout_config.json"
echo " 2. Test your Telegram bot connection:"
echo "    cd ${INSTALL_DIR} && ./venv/bin/python scanner/kraken_breakout_scanner.py --test-telegram"
echo " 3. Run a test dry-run scan:"
echo "    cd ${INSTALL_DIR} && ./venv/bin/python scanner/kraken_breakout_scanner.py --dry-run"
echo "=========================================================="
