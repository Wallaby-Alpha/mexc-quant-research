"""
tests/test_weex_london_dip_bot.py
=================================
Unit tests for WEEX Setup 11 ("London Dip to NY Peak") Trading Bot.

Validates:
1. WEEX V3 HMAC SHA256 base64 signature generation format.
2. Strict $1.00 risk sizing algorithm:
   - Correctly qualifies low-risk symbols and sizes to <= $1.00 risk.
   - Strictly REJECTS any coin whose minimum lot size would risk > $1.00.
3. Price and quantity formatting matching exchange lot steps.
4. Order placement payload construction and fallback mechanism.
"""

import pytest
from bot.weex_london_dip_bot import WeexClient, LondonDipBot


def test_weex_signature_generation():
    """Validates WEEX HMAC SHA256 base64 signature format."""
    client = WeexClient(
        api_key="my_api_key",
        secret_key="my_secret_key",
        passphrase="my_passphrase",
        dry_run=True
    )
    sig = client._generate_signature(
        timestamp="1680000000000",
        method="POST",
        path="/capi/v3/order",
        body='{"symbol":"SOLUSDT"}'
    )
    assert isinstance(sig, str)
    assert len(sig) > 20
    # Signature must be valid base64
    import base64
    decoded = base64.b64decode(sig)
    assert len(decoded) == 32 # SHA256 produces 32-byte digest


def test_calculate_risk_position_size_eligible():
    """Test standard altcoin where minimum order size allows <= $1.00 risk."""
    client = WeexClient("key", "sec", "pass", dry_run=True)
    # Mock exchange info for an altcoin
    client.exchange_info = {
        "DOGEUSDT": {
            "pricePrecision": 5,
            "quantityPrecision": 0,
            "contractVal": 10.0,
            "minOrderSize": 10.0
        }
    }

    # Price = $0.15, SL = 8% (0.08), Max risk = $1.00
    # Loss per unit = $0.15 * 0.08 = $0.012
    # Desired qty = $1.00 / $0.012 = 83.33 units
    # Step size = 10 -> rounded down to 80 units
    # Actual risk = 80 * 0.012 = $0.96 <= $1.00
    qty, risk_usd, eligible, reason = client.calculate_risk_position_size(
        symbol="DOGEUSDT",
        entry_price=0.15,
        sl_pct=0.08,
        max_risk_usd=1.00
    )

    assert eligible is True
    assert qty == 80.0
    assert risk_usd <= 1.00
    assert risk_usd == pytest.approx(0.96, abs=0.01)
    assert reason == "Eligible"


def test_calculate_risk_position_size_rejected_exceeds_max_risk():
    """Test high-priced or large-lot coin where buying 1 minimum contract risks > $1.00."""
    client = WeexClient("key", "sec", "pass", dry_run=True)
    # Mock BTCUSDT with minOrderSize = 0.001 BTC
    client.exchange_info = {
        "BTCUSDT": {
            "pricePrecision": 2,
            "quantityPrecision": 3,
            "contractVal": 0.001,
            "minOrderSize": 0.001
        }
    }

    # Price = $65,000, SL = 8% (0.08), Max risk = $1.00
    # Min order notional = 0.001 * $65,000 = $65.00
    # Min order risk = $65.00 * 0.08 = $5.20
    # $5.20 > $1.00 -> MUST BE REJECTED!
    qty, risk_usd, eligible, reason = client.calculate_risk_position_size(
        symbol="BTCUSDT",
        entry_price=65000.0,
        sl_pct=0.08,
        max_risk_usd=1.00
    )

    assert eligible is False
    assert qty == 0.0
    assert risk_usd > 1.00
    assert "exceeding max allowed $1.00" in reason


def test_format_values_precision():
    """Verifies numbers are correctly formatted according to exchange precision."""
    client = WeexClient("key", "sec", "pass", dry_run=True)
    client.exchange_info = {
        "SOLUSDT": {
            "pricePrecision": 3,
            "quantityPrecision": 2,
            "contractVal": 0.1,
            "minOrderSize": 0.1
        }
    }

    qty_str, entry_str, sl_str = client.format_values("SOLUSDT", 1.4589, 142.34567, 130.95801)
    assert qty_str == "1.46"
    assert entry_str == "142.346"
    assert sl_str == "130.958"


def test_dry_run_market_buy_and_close():
    """Verifies dry-run simulated execution cycle."""
    client = WeexClient("key", "sec", "pass", dry_run=True)
    client.exchange_info = {
        "PEPEUSDT": {
            "pricePrecision": 8,
            "quantityPrecision": 0,
            "contractVal": 1000.0,
            "minOrderSize": 1000.0
        }
    }

    # Place order
    res_buy = client.place_market_buy_with_sl("PEPEUSDT", 100000.0, 0.00001, 0.0000092)
    assert res_buy["status"] == "simulated"
    assert res_buy["symbol"] == "PEPEUSDT"
    assert "sim_" in res_buy["orderId"]

    # Close order
    res_close = client.close_position_market("PEPEUSDT", 100000.0)
    assert res_close["status"] == "simulated"
    assert res_close["qty"] == 100000.0
