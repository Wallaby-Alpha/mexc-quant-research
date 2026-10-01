"""
reports/setup_viewer.py
Interactive Setup Viewer for MEXC Swing-High Retest Research (Phase 5).
Generates a standalone, beautiful HTML dashboard (reports/setup_viewer.html)
allowing inspection of individual simulated trades, candlestick paths,
entry/stop/target levels, and financial outcomes.
"""

from pathlib import Path
from typing import List, Dict, Any
import json
import pandas as pd


def generate_setup_viewer_html(
    sample_trades: List[Dict[str, Any]],
    output_path: str = "reports/setup_viewer.html"
):
    """
    Renders standalone interactive setup viewer.
    """
    trades_json = json.dumps(sample_trades, default=str)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MEXC Swing-High Retest — Setup & Trade Viewer</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        :root {{
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-green: #22c55e;
            --accent-red: #ef4444;
            --accent-blue: #3b82f6;
            --border-color: #334155;
        }}
        body {{
            margin: 0;
            padding: 20px;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
        }}
        .header {{
            margin-bottom: 24px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 16px;
        }}
        h1 {{
            margin: 0 0 8px 0;
            font-size: 24px;
            font-weight: 700;
        }}
        .subtitle {{
            color: var(--text-secondary);
            font-size: 14px;
        }}
        .layout {{
            display: grid;
            grid-template-columns: 320px 1fr;
            gap: 20px;
            height: calc(100vh - 120px);
        }}
        .sidebar {{
            background: var(--bg-secondary);
            border-radius: 8px;
            border: 1px solid var(--border-color);
            overflow-y: auto;
            padding: 12px;
        }}
        .trade-card {{
            padding: 12px;
            border-radius: 6px;
            border: 1px solid var(--border-color);
            margin-bottom: 10px;
            cursor: pointer;
            transition: all 0.15s ease;
            background: #182234;
        }}
        .trade-card:hover, .trade-card.active {{
            border-color: var(--accent-blue);
            background: #233148;
        }}
        .trade-header {{
            display: flex;
            justify-content: space-between;
            font-weight: 600;
            font-size: 14px;
            margin-bottom: 4px;
        }}
        .badge-win {{ color: var(--accent-green); }}
        .badge-loss {{ color: var(--accent-red); }}
        .trade-meta {{
            font-size: 12px;
            color: var(--text-secondary);
        }}
        .main-view {{
            display: flex;
            flex-direction: column;
            gap: 16px;
        }}
        .chart-box {{
            flex: 1;
            background: var(--bg-secondary);
            border-radius: 8px;
            border: 1px solid var(--border-color);
            padding: 16px;
            min-height: 480px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 12px;
            background: var(--bg-secondary);
            border-radius: 8px;
            border: 1px solid var(--border-color);
            padding: 14px;
        }}
        .metric-item {{
            display: flex;
            flex-direction: column;
        }}
        .metric-label {{
            font-size: 11px;
            color: var(--text-secondary);
            text-transform: uppercase;
        }}
        .metric-val {{
            font-size: 16px;
            font-weight: 600;
            margin-top: 2px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>MEXC Swing-High Retest — Setup & Trade Viewer</h1>
        <div class="subtitle">Interactive Bar-by-Bar Trade Inspection • Net-of-Cost Execution • Zero Lookahead</div>
    </div>
    <div class="layout">
        <div class="sidebar" id="tradeList"></div>
        <div class="main-view">
            <div class="metrics-grid" id="metricsBox"></div>
            <div class="chart-box" id="chart"></div>
        </div>
    </div>

    <script>
        const trades = {trades_json};
        let activeIdx = 0;

        function renderSidebar() {{
            const listEl = document.getElementById("tradeList");
            listEl.innerHTML = "";
            trades.forEach((t, i) => {{
                const isWin = t.net_pnl_r > 0;
                const card = document.createElement("div");
                card.className = "trade-card" + (i === activeIdx ? " active" : "");
                card.innerHTML = `
                    <div class="trade-header">
                        <span>${{t.symbol}}</span>
                        <span class="${{isWin ? 'badge-win' : 'badge-loss'}}">${{t.net_pnl_r >= 0 ? '+' : ''}}${{t.net_pnl_r.toFixed(2)}} R</span>
                    </div>
                    <div class="trade-meta">
                        <div>${{t.entry_time.split('T')[0]}} • ${{t.exit_reason}}</div>
                        <div>Mode: ${{t.entry_mode}} • Hold: ${{t.holding_time_minutes}}m</div>
                    </div>
                `;
                card.onclick = () => {{
                    activeIdx = i;
                    renderSidebar();
                    renderView();
                }};
                listEl.appendChild(card);
            }});
        }}

        function renderView() {{
            const t = trades[activeIdx];
            if (!t) return;

            // Render Metrics
            const mBox = document.getElementById("metricsBox");
            mBox.innerHTML = `
                <div class="metric-item">
                    <span class="metric-label">Symbol / ID</span>
                    <span class="metric-val">${{t.symbol}}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Net Return</span>
                    <span class="metric-val" style="color: ${{t.net_pnl_r >= 0 ? 'var(--accent-green)' : 'var(--accent-red)'}}">${{t.net_pnl_r.toFixed(2)}} R (${{(t.net_pnl_pct * 100).toFixed(2)}}%)</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Gross Return</span>
                    <span class="metric-val">${{t.gross_pnl_r.toFixed(2)}} R (${{(t.gross_pnl_pct * 100).toFixed(2)}}%)</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Exit Reason</span>
                    <span class="metric-val">${{t.exit_reason}}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Fees + Slippage</span>
                    <span class="metric-val">${{((t.total_fees + t.slippage_cost) * 10000).toFixed(1)}} bps</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">MFE / MAE</span>
                    <span class="metric-val">${{t.mfe_r.toFixed(2)}}R / ${{t.mae_r.toFixed(2)}}R</span>
                </div>
            `;

            // Synthesize simple illustrative price path for chart
            const bars = t.candles || [];
            if (bars.length > 0) {{
                const trace = {{
                    x: bars.map(b => b.time),
                    open: bars.map(b => b.open),
                    high: bars.map(b => b.high),
                    low: bars.map(b => b.low),
                    close: bars.map(b => b.close),
                    type: 'candlestick',
                    name: t.symbol,
                    increasing: {{line: {{color: '#22c55e'}}}},
                    decreasing: {{line: {{color: '#ef4444'}}}}
                }};

                const layout = {{
                    title: `${{t.symbol}} — ${{t.trade_id}} (${{t.entry_mode}})`,
                    dragmode: 'zoom',
                    showlegend: false,
                    xaxis: {{rangeslider: {{visible: false}}, gridcolor: '#334155'}},
                    yaxis: {{gridcolor: '#334155'}},
                    paper_bgcolor: '#1e293b',
                    plot_bgcolor: '#1e293b',
                    font: {{color: '#f8fafc'}},
                    shapes: [
                        // Target
                        {{type: 'line', xref: 'paper', x0: 0, x1: 1, y0: t.target_price, y1: t.target_price, line: {{color: '#22c55e', width: 2, dash: 'dash'}}}},
                        // Entry
                        {{type: 'line', xref: 'paper', x0: 0, x1: 1, y0: t.entry_price, y1: t.entry_price, line: {{color: '#3b82f6', width: 1.5, dash: 'dot'}}}},
                        // Stop
                        {{type: 'line', xref: 'paper', x0: 0, x1: 1, y0: t.stop_price, y1: t.stop_price, line: {{color: '#ef4444', width: 2, dash: 'dash'}}}}
                    ]
                }};
                Plotly.newPlot('chart', [trace], layout, {{responsive: true}});
            }} else {{
                document.getElementById("chart").innerHTML = `
                    <div style="display:flex;height:100%;align-items:center;justify-content:center;color:var(--text-secondary);flex-direction:column;">
                        <h3>Trade Detail: ${{t.trade_id}}</h3>
                        <p>Entry: ${{t.entry_price.toFixed(4)}} | Stop: ${{t.stop_price.toFixed(4)}} | Target: ${{t.target_price.toFixed(4)}}</p>
                        <p>Exit: ${{t.exit_price.toFixed(4)}} (${{t.exit_reason}}) | Net: ${{t.net_pnl_r.toFixed(2)}}R</p>
                    </div>
                `;
            }}
        }}

        renderSidebar();
        renderView();
    </script>
</body>
</html>
"""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    return out_file
