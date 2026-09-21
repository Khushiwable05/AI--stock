import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from io import StringIO
import os

st.set_page_config(page_title="NeuralTrade AI", layout="wide", page_icon="⚡")

CSV_DATA = """Open,High,Low,Close,Volume,Signal
120,125,115,124,15000,Buy
130,135,128,126,14000,Sell
140,145,138,142,16000,Buy
150,155,145,148,17000,Sell
110,115,105,112,13000,Hold
100,105,98,104,12000,Buy
105,110,102,101,11000,Sell
115,120,112,118,14000,Buy
125,130,120,122,15000,Sell
135,140,130,137,16000,Buy
140,145,135,138,15000,Hold
145,150,140,142,14000,Sell
150,155,145,152,17000,Buy
155,160,150,153,18000,Hold
160,165,155,158,16000,Sell
165,170,160,168,17500,Buy
170,175,165,169,16500,Hold
175,180,170,172,16000,Sell
180,185,175,183,19000,Buy
185,190,180,186,18000,Hold
120,126,118,125,15000,Buy
130,132,125,127,14000,Sell
140,148,138,147,16000,Buy
150,152,145,146,17000,Sell
110,113,105,109,13000,Hold
100,106,98,105,12000,Buy
105,108,102,100,11000,Sell
115,123,112,121,14000,Buy
125,128,120,121,15000,Sell
135,142,130,140,16000,Buy
140,143,135,139,15000,Hold
145,148,140,141,14000,Sell
150,158,145,156,17000,Buy
155,158,150,152,18000,Hold
160,162,155,157,16000,Sell
165,173,160,171,17500,Buy
170,172,165,168,16500,Hold
175,178,170,171,16000,Sell
180,188,175,186,19000,Buy
185,188,180,183,18000,Hold
120,128,115,127,15000,Buy
130,134,128,129,14000,Sell
140,147,138,146,16000,Buy
150,153,145,147,17000,Sell
110,114,105,111,13000,Hold
100,107,98,106,12000,Buy
105,109,102,101,11000,Sell
115,124,112,123,14000,Buy
125,129,120,122,15000,Sell
135,143,130,141,16000,Buy
140,144,135,140,15000,Hold
145,149,140,142,14000,Sell
150,159,145,157,17000,Buy
155,159,150,153,18000,Hold
160,163,155,158,16000,Sell
165,174,160,172,17500,Buy
170,173,165,169,16500,Hold
175,179,170,172,16000,Sell
180,189,175,187,19000,Buy
185,189,180,184,18000,Hold
120,127,115,126,15000,Buy
130,133,125,126,14000,Sell
140,146,138,145,16000,Buy
150,151,145,145,17000,Sell
110,112,105,108,13000,Hold
100,104,98,103,12000,Buy
105,107,102,100,11000,Sell
115,122,112,120,14000,Buy
125,127,120,121,15000,Sell
135,141,130,139,16000,Buy
140,142,135,138,15000,Hold
145,147,140,141,14000,Sell
150,157,145,155,17000,Buy
155,157,150,152,18000,Hold
160,161,155,157,16000,Sell
165,172,160,170,17500,Buy
170,171,165,168,16500,Hold
175,177,170,171,16000,Sell
180,187,175,185,19000,Buy
185,187,180,183,18000,Hold"""

@st.cache_data
def load_and_train():
    if os.path.exists("stock_data.csv"):
        data = pd.read_csv("stock_data.csv")
    else:
        data = pd.read_csv(StringIO(CSV_DATA))
    data["Trend"] = data["Close"] - data["Open"]
    X = data[['Open','High','Low','Close','Volume','Trend']]
    y = data['Signal']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return data, model, acc

data, model, acc = load_and_train()
buy_count  = int((data['Signal']=='Buy').sum())
sell_count = int((data['Signal']=='Sell').sum())
hold_count = int((data['Signal']=='Hold').sum())

st.markdown("""
<style>
html,body,[data-testid="stAppViewContainer"],[data-testid="stAppViewBlockContainer"],
section.main,.main,.block-container{
  background:#f8f9fb !important;padding:0 !important;margin:0 !important;}
[data-testid="stHeader"],[data-testid="stToolbar"],
#MainMenu,footer,.stDeployButton{display:none !important;visibility:hidden !important;}
[data-testid="stAppViewContainer"]{padding-top:0 !important;}
.block-container{max-width:100% !important;padding:0 !important;}
iframe{display:block;border:none;}
</style>
""", unsafe_allow_html=True)

APP_HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0;}}
html{{scroll-behavior:smooth;}}
body{{
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  background:#f8f9fb;color:#1a1d23;min-height:100vh;overflow-x:hidden;
}}

/* HEADER */
.header{{
  background:#fff;border-bottom:1px solid #e2e5ea;
  padding:20px 40px 18px;display:flex;align-items:center;justify-content:space-between;
}}
.header-left h1{{font-size:20px;font-weight:600;color:#1a1d23;letter-spacing:-.01em;}}
.header-left p{{font-size:13px;color:#6b7280;margin-top:2px;}}
.header-right{{display:flex;gap:8px;}}
.stat-badge{{
  background:#f1f5f9;border:1px solid #e2e5ea;border-radius:8px;
  padding:6px 14px;font-size:12px;color:#374151;font-weight:500;
}}
.stat-badge.green{{background:#f0fdf4;border-color:#bbf7d0;color:#166534;}}
.stat-badge.red{{background:#fef2f2;border-color:#fecaca;color:#991b1b;}}
.stat-badge.blue{{background:#eff6ff;border-color:#bfdbfe;color:#1e40af;}}

/* MAIN */
.app{{max-width:1100px;margin:0 auto;padding:32px 32px 60px;}}

/* METRICS */
.metric-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:32px;}}
.mc{{
  background:#fff;border:1px solid #e2e5ea;border-radius:12px;
  padding:18px 20px;
}}
.mc-label{{font-size:11px;letter-spacing:.05em;color:#9ca3af;text-transform:uppercase;margin-bottom:8px;font-weight:500;}}
.mc-val{{font-size:26px;font-weight:600;color:#1a1d23;}}
.mc-sub{{font-size:12px;margin-top:4px;color:#6b7280;}}

/* SECTION LABEL */
.slabel{{font-size:11px;letter-spacing:.08em;color:#9ca3af;text-transform:uppercase;font-weight:500;margin-bottom:14px;}}

/* TWO COL */
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:24px;}}
.panel{{background:#fff;border:1px solid #e2e5ea;border-radius:14px;padding:26px;}}

/* FORM */
.input-grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;}}
.field{{display:flex;flex-direction:column;gap:5px;}}
.field label{{font-size:11px;letter-spacing:.05em;color:#6b7280;text-transform:uppercase;font-weight:500;}}
.field input{{
  background:#f8f9fb;border:1px solid #d1d5db;
  color:#1a1d23;border-radius:8px;padding:9px 12px;
  font-size:14px;outline:none;transition:border-color .15s;
}}
.field input:focus{{border-color:#3b82f6;}}
.trend-row{{
  display:flex;align-items:center;justify-content:space-between;
  background:#f8f9fb;border:1px solid #e2e5ea;border-radius:8px;
  padding:10px 14px;margin:12px 0;font-size:12px;color:#6b7280;font-weight:500;
}}
.trend-val{{font-size:16px;font-weight:600;}}

.radio-group{{display:flex;gap:8px;margin:12px 0;}}
.radio-btn{{
  flex:1;padding:8px;border-radius:8px;border:1px solid #d1d5db;
  background:#fff;color:#6b7280;font-size:12px;cursor:pointer;
  transition:all .15s;text-align:center;font-weight:500;
}}
.radio-btn.active{{
  background:#1e40af;color:#fff;border-color:#1e40af;
}}
.radio-btn:hover:not(.active){{border-color:#3b82f6;color:#3b82f6;}}

.predict-btn{{
  width:100%;height:48px;border-radius:10px;border:none;cursor:pointer;
  background:#1e40af;color:#fff;font-size:15px;font-weight:600;
  margin-top:4px;transition:background .15s,transform .1s;
}}
.predict-btn:hover{{background:#1d4ed8;}}
.predict-btn:active{{transform:scale(.98);}}
.predict-btn.loading{{opacity:.6;pointer-events:none;}}
.predict-btn .btn-spinner{{
  display:none;width:16px;height:16px;border-radius:50%;
  border:2px solid rgba(255,255,255,.35);border-top-color:#fff;
  animation:spin .7s linear infinite;vertical-align:middle;
}}
.predict-btn.loading .btn-spinner{{display:inline-block;}}
.predict-btn.loading .btn-text{{display:none;}}
@keyframes spin{{to{{transform:rotate(360deg)}}}}

/* RESULT */
.rbanner{{
  padding:16px 20px;display:flex;align-items:center;gap:14px;
  font-size:15px;font-weight:600;border-radius:10px;border:1px solid;
  margin-bottom:14px;
}}
.rbuy {{background:#f0fdf4;border-color:#bbf7d0;color:#166534;}}
.rsell{{background:#fef2f2;border-color:#fecaca;color:#991b1b;}}
.rhold{{background:#f8fafc;border-color:#cbd5e1;color:#475569;}}
.ric{{font-size:24px;}}
.rlabel{{font-size:11px;opacity:.6;margin-bottom:2px;text-transform:uppercase;letter-spacing:.05em;font-weight:500;}}

.insight-row{{display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap;}}
.insight-chip{{
  padding:4px 10px;border-radius:6px;font-size:11px;letter-spacing:.04em;font-weight:500;
  background:#f1f5f9;border:1px solid #e2e5ea;color:#374151;
}}
.insight-chip span{{color:#1e40af;font-weight:600;}}

.next-open-chip{{
  display:inline-flex;align-items:center;gap:8px;
  padding:8px 14px;border-radius:8px;border:1px solid;
  font-size:12px;font-weight:500;margin-bottom:14px;
}}

/* CHART */
.chart-wrap{{background:#f8f9fb;border:1px solid #e2e5ea;border-radius:12px;padding:16px;}}
.chart-hdr{{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:12px;}}
.chart-title{{font-size:11px;color:#9ca3af;text-transform:uppercase;letter-spacing:.05em;font-weight:500;}}
#chart-price{{font-size:17px;font-weight:600;color:#1a1d23;}}
#chart-area{{position:relative;height:200px;cursor:crosshair;}}
#chart-cv{{display:block;}}
#chart-tip{{
  position:absolute;background:#fff;border:1px solid #e2e5ea;
  border-radius:8px;padding:10px 12px;font-size:11px;pointer-events:none;
  color:#1a1d23;opacity:0;transition:opacity .1s;min-width:110px;
  box-shadow:0 4px 12px rgba(0,0,0,.08);z-index:10;
}}
.tipdate{{color:#9ca3af;margin-bottom:5px;font-size:10px;}}
.tipr{{display:flex;justify-content:space-between;gap:12px;margin-top:2px;font-size:11px;}}
.tipl{{color:#6b7280;}}
.tipv{{font-weight:600;}}

/* PLACEHOLDER */
.placeholder{{
  height:280px;display:flex;flex-direction:column;
  align-items:center;justify-content:center;
  border:1px dashed #d1d5db;border-radius:12px;
  color:#9ca3af;gap:10px;
}}
.ph-icon{{font-size:28px;opacity:.35;}}
.ph-txt{{font-size:12px;color:#9ca3af;}}

.divider{{height:1px;background:#e2e5ea;margin:28px 0;}}
</style>
</head>
<body>

<!-- HEADER -->
<div class="header">
  <div class="header-left">
    <h1>NeuralTrade AI</h1>
    <p>Decision Tree · NSE/BSE · Stock Signal Predictor</p>
  </div>
  <div class="header-right">
    <span class="stat-badge green">▲ {buy_count} Buy</span>
    <span class="stat-badge red">▼ {sell_count} Sell</span>
    <span class="stat-badge">◆ {hold_count} Hold</span>
    <span class="stat-badge blue">⚡ {round(acc*100,1)}% Accuracy</span>
  </div>
</div>

<!-- MAIN -->
<div class="app">

  <!-- METRICS -->
  <div class="metric-row">
    <div class="mc">
      <div class="mc-label">Model Accuracy</div>
      <div class="mc-val">{round(acc*100,1)}%</div>
      <div class="mc-sub">Decision Tree</div>
    </div>
    <div class="mc">
      <div class="mc-label">Dataset Size</div>
      <div class="mc-val">{len(data)}</div>
      <div class="mc-sub">Training rows</div>
    </div>
    <div class="mc">
      <div class="mc-label">Buy Signals</div>
      <div class="mc-val" style="color:#166534">{buy_count}</div>
      <div class="mc-sub" style="color:#16a34a">{round(buy_count/len(data)*100)}% of data</div>
    </div>
    <div class="mc">
      <div class="mc-label">Sell Signals</div>
      <div class="mc-val" style="color:#991b1b">{sell_count}</div>
      <div class="mc-sub" style="color:#dc2626">{round(sell_count/len(data)*100)}% of data</div>
    </div>
  </div>

  <div class="divider"></div>

  <!-- PREDICT -->
  <div class="slabel">Predict Your Stock</div>
  <div class="two-col">

    <!-- INPUT -->
    <div class="panel">
      <div class="input-grid">
        <div class="field"><label>Open Price</label><input type="number" id="i-open" value="120" min="0" oninput="recalc()"></div>
        <div class="field"><label>High Price</label><input type="number" id="i-high" value="125" min="0" oninput="recalc()"></div>
        <div class="field"><label>Low Price</label> <input type="number" id="i-low"  value="115" min="0" oninput="recalc()"></div>
        <div class="field"><label>Close Price</label><input type="number" id="i-close" value="123" min="0" oninput="recalc()"></div>
        <div class="field" style="grid-column:1/-1"><label>Volume</label><input type="number" id="i-vol" value="15000" min="0" oninput="recalc()"></div>
      </div>
      <div class="trend-row">
        <span>Computed Trend</span>
        <span class="trend-val" id="trend-val" style="color:#166534">+3</span>
      </div>
      <div class="radio-group">
        <button class="radio-btn active" id="rb-candle" onclick="setChart('candlestick')">Candlestick</button>
        <button class="radio-btn"        id="rb-line"   onclick="setChart('line')">Line Trend</button>
      </div>
      <button class="predict-btn" id="predict-btn" onclick="runPredict()">
        <div class="btn-spinner"></div>
        <span class="btn-text">Run AI Prediction</span>
      </button>
    </div>

    <!-- OUTPUT -->
    <div id="right-panel">
      <div class="placeholder" id="ph">
        <div class="ph-icon">⚡</div>
        <div class="ph-txt">Enter values and run prediction</div>
      </div>
    </div>

  </div>
</div>

<script>
const TIMES = Array.from({{length:30}}, (_,i)=>{{
  const mins = 10*60 + i*16;
  const h=Math.floor(mins/60), m=mins%60;
  const ap=h>=12?'PM':'AM', h12=h>12?h-12:h;
  return (h12===0?12:h12)+':'+(m<10?'0':'')+m+' '+ap;
}});

let CHART_OPENS=[], CHART_CLOSES=[], CHART_HIGHS=[], CHART_LOWS=[], CHART_PRICES=[];

function generateChartData(open, high, low, close) {{
  const N = 30;
  const rng = (min,max) => min + Math.random()*(max-min);
  const mids = [open];
  for (let i=1;i<N;i++) {{
    const progress = i/(N-1);
    const drift = (close - open) * progress;
    const noise = (high - low) * 0.08 * (Math.random()-0.5);
    mids.push(open + drift + noise);
  }}
  mids[N-1] = close;
  for (let i=0;i<N;i++) mids[i] = Math.max(low, Math.min(high, mids[i]));
  const opens=[], closes=[], highs=[], lows=[];
  for (let i=0;i<N;i++) {{
    const mid = mids[i];
    const bodyHalf = (high-low) * rng(0.01, 0.06);
    const bull = i===N-1 ? close>=open : Math.random()>0.45;
    const o2 = Math.max(low, Math.min(high, mid + (bull?-bodyHalf:bodyHalf)));
    const c2 = Math.max(low, Math.min(high, mid + (bull?bodyHalf:-bodyHalf)));
    const wickUp   = (high-low)*rng(0.01,0.07);
    const wickDown = (high-low)*rng(0.01,0.07);
    opens.push(parseFloat(o2.toFixed(2)));
    closes.push(parseFloat(c2.toFixed(2)));
    highs.push(parseFloat(Math.min(high, Math.max(o2,c2)+wickUp).toFixed(2)));
    lows.push(parseFloat(Math.max(low,  Math.min(o2,c2)-wickDown).toFixed(2)));
  }}
  opens[0]  = open;
  closes[0] = parseFloat((open + (close-open)*rng(0.02,0.12)).toFixed(2));
  highs[0]  = Math.max(opens[0], closes[0]) + (high-low)*rng(0.01,0.05);
  lows[0]   = Math.min(opens[0], closes[0]) - (high-low)*rng(0.01,0.05);
  closes[N-1] = close;
  opens[N-1]  = parseFloat((close - (close-open)*rng(0.02,0.12)).toFixed(2));
  highs[N-1]  = Math.max(opens[N-1], closes[N-1]) + (high-low)*rng(0.01,0.05);
  lows[N-1]   = Math.min(opens[N-1], closes[N-1]) - (high-low)*rng(0.01,0.05);
  const hiIdx = Math.floor(N*0.2 + Math.random()*N*0.6);
  highs[hiIdx] = high;
  opens[hiIdx] = Math.min(opens[hiIdx], high - (high-low)*0.03);
  closes[hiIdx]= Math.min(closes[hiIdx],high - (high-low)*0.03);
  lows[hiIdx]  = Math.min(lows[hiIdx], opens[hiIdx]);
  let loIdx = Math.floor(N*0.2 + Math.random()*N*0.6);
  if(loIdx===hiIdx) loIdx=(loIdx+5)%N;
  lows[loIdx]  = low;
  opens[loIdx] = Math.max(opens[loIdx], low + (high-low)*0.03);
  closes[loIdx]= Math.max(closes[loIdx],low + (high-low)*0.03);
  highs[loIdx] = Math.max(highs[loIdx], opens[loIdx]);
  const prices = mids.map(p=>parseFloat(Math.max(low,Math.min(high,p)).toFixed(2)));
  return {{opens, closes, highs, lows, prices}};
}}

function decisionTreePredict(open, high, low, close, volume) {{
  const trend      = close - open;
  const trend_pct  = open > 0 ? trend / open : 0;
  const range      = high - low;
  const volatility = open > 0 ? range / open : 0;
  const body       = Math.abs(close - open);
  const strength   = range > 0 ? body / range : 0;
  if (Math.abs(trend_pct) < 0.03)              return 'Hold';
  if (volatility > 0.40)                        return 'Hold';
  if (strength < 0.3)                           return 'Hold';
  if (range > 0.35 * open)                      return 'Hold';
  if (trend < -2)                               return 'Sell';
  if (low < open * 0.7 && close < open * 1.03) return 'Sell';
  if ((open - low) / open > 0.3 && trend >= 0 && trend < 2) return 'Sell';
  if (trend > 2)                                return 'Buy';
  if (strength > 0.6 && trend > 0)             return 'Buy';
  if (volatility < 0.25 && trend > 0)          return 'Buy';
  return 'Hold';
}}

function getConfidence(open, high, low, close, volume) {{
  const range      = high - low || 1;
  const body       = Math.abs(close - open);
  const strength   = body / range;
  const volatility = open > 0 ? range / open : 1;
  const stability  = Math.max(0, 1 - volatility * 2);
  const vol_score  = volume >= 17000 ? 1.0 : volume >= 14000 ? 0.75 : volume >= 11000 ? 0.5 : 0.25;
  const raw = 50 + strength * 25 + stability * 15 + vol_score * 8;
  return Math.round(Math.min(Math.max(raw, 45), 97));
}}

function getInsights(open, high, low, close, volume) {{
  const trend      = close - open;
  const range      = high - low;
  const body       = Math.abs(trend);
  const strength   = range > 0 ? (body / range * 100).toFixed(0) : 0;
  const volatility = open > 0 ? (range / open * 100).toFixed(1) : '0.0';
  const trend_pct  = open > 0 ? (trend / open * 100).toFixed(2) : '0.00';
  return [
    `Trend <span>${{trend >= 0 ? '+' : ''}}${{trend}} (${{trend_pct}}%)</span>`,
    `Strength <span>${{strength}}% of range</span>`,
    `Volatility <span>${{volatility}}%</span>`,
    `Volume <span>${{volume >= 15000 ? 'High' : volume >= 12000 ? 'Medium' : 'Low'}}</span>`
  ];
}}

function predictNextOpen(open, high, low, close) {{
  const trend      = close - open;
  const volatility = (high - low) / open;
  return Math.round(close + (trend * 0.3) - (volatility * open * 0.2));
}}

let chartType='candlestick';
let chartRendered=false;

function setChart(t){{
  chartType=t;
  document.getElementById('rb-candle').classList.toggle('active',t==='candlestick');
  document.getElementById('rb-line').classList.toggle('active',t==='line');
  if(chartRendered) renderChart();
}}

function recalc(){{
  const o=+document.getElementById('i-open').value||0;
  const c=+document.getElementById('i-close').value||0;
  const trend=c-o;
  const tv=document.getElementById('trend-val');
  tv.textContent=(trend>=0?'+':'')+trend;
  tv.style.color=trend>=0?'#166534':'#991b1b';
}}

function runPredict(){{
  const btn=document.getElementById('predict-btn');
  const o=+document.getElementById('i-open').value||120;
  const h=+document.getElementById('i-high').value||125;
  const l=+document.getElementById('i-low').value||115;
  const c=+document.getElementById('i-close').value||123;
  const v=+document.getElementById('i-vol').value||15000;
  btn.classList.add('loading');
  setTimeout(()=>{{
    const cd = generateChartData(o, h, l, c);
    CHART_OPENS  = cd.opens;
    CHART_CLOSES = cd.closes;
    CHART_HIGHS  = cd.highs;
    CHART_LOWS   = cd.lows;
    CHART_PRICES = cd.prices;
    const pred     = decisionTreePredict(o, h, l, c, v);
    const conf     = getConfidence(o, h, l, c, v);
    const insights = getInsights(o, h, l, c, v);
    const nextOpen = predictNextOpen(o, h, l, c);
    btn.classList.remove('loading');
    showResult(pred, conf, insights, c, nextOpen);
    chartRendered=true;
    renderChart();
  }}, 380);
}}

function showResult(pred, conf, insights, closePrice, nextOpen){{
  const rp=document.getElementById('right-panel');
  const cls=pred==='Buy'?'rbuy':pred==='Sell'?'rsell':'rhold';
  const icon=pred==='Buy'?'▲':pred==='Sell'?'▼':'◆';
  const msgMap = {{
    'Buy':  'Bullish pattern — upward momentum detected',
    'Sell': 'Bearish signal — downward pressure identified',
    'Hold': 'Consolidation phase — no clear directional edge'
  }};
  const insightChips = insights.map(i=>`<div class="insight-chip">${{i}}</div>`).join('');
  const nextDir   = nextOpen > closePrice ? '▲' : nextOpen < closePrice ? '▼' : '◆';
  const nextColor = nextOpen > closePrice ? '#166534' : nextOpen < closePrice ? '#991b1b' : '#475569';
  const nextBg    = nextOpen > closePrice ? '#f0fdf4' : nextOpen < closePrice ? '#fef2f2' : '#f8fafc';
  const nextBorder= nextOpen > closePrice ? '#bbf7d0' : nextOpen < closePrice ? '#fecaca' : '#cbd5e1';

  rp.innerHTML=`
    <div class="rbanner ${{cls}}">
      <span class="ric">${{icon}}</span>
      <div>
        <div class="rlabel">Decision Tree · ${{conf}}% confidence</div>
        ${{pred}} — ${{msgMap[pred]}}
      </div>
    </div>
    <div class="insight-row">${{insightChips}}</div>
    <div style="margin-bottom:14px;">
      <div class="next-open-chip" style="background:${{nextBg}};border-color:${{nextBorder}};color:${{nextColor}};">
        ${{nextDir}} Predicted next open &nbsp;
        <strong style="font-size:15px;color:${{nextColor}};">₹${{nextOpen}}</strong>
      </div>
    </div>
    <div class="chart-wrap">
      <div class="chart-hdr">
        <span class="chart-title" id="ct-label">Simulated movement</span>
        <span id="chart-price">₹${{closePrice}}</span>
      </div>
      <div id="chart-area">
        <canvas id="chart-cv"></canvas>
        <div id="chart-tip">
          <div class="tipdate" id="ct-date">—</div>
          <div class="tipr"><span class="tipl">Time</span><span class="tipv" id="ct-time">—</span></div>
          <div class="tipr"><span class="tipl">Open</span><span class="tipv" id="ct-o">—</span></div>
          <div class="tipr"><span class="tipl">Close</span><span class="tipv" id="ct-c">—</span></div>
          <div class="tipr"><span class="tipl">High</span><span class="tipv" id="ct-h">—</span></div>
          <div class="tipr"><span class="tipl">Low</span><span class="tipv" id="ct-l">—</span></div>
        </div>
      </div>
    </div>`;
  renderChart();
}}

function renderChart(){{
  const area=document.getElementById('chart-area');
  const cv=document.getElementById('chart-cv');
  if(!area||!cv)return;
  const ctx=cv.getContext('2d');
  const tip=document.getElementById('chart-tip');
  const lbl=document.getElementById('ct-label');
  if(lbl)lbl.textContent='Simulated movement · '+chartType;
  let W,H,step,minP,maxP;

  function setup(){{
    W=cv.width=area.offsetWidth;H=cv.height=area.offsetHeight;
    step=W/CHART_OPENS.length;
    const allH=chartType==='candlestick'?CHART_HIGHS:CHART_PRICES;
    const allL=chartType==='candlestick'?CHART_LOWS:CHART_PRICES;
    minP=Math.min(...allL)-2;maxP=Math.max(...allH)+2;
    draw();
  }}
  function py(p){{return H-((p-minP)/(maxP-minP))*(H-28)-14;}}

  function drawTimeAxis(){{
    ctx.fillStyle='#9ca3af';ctx.font='9px -apple-system,sans-serif';ctx.textAlign='center';
    [0,5,10,15,20,25,29].forEach(i=>{{
      if(i<TIMES.length) ctx.fillText(TIMES[i],i*step+step/2,H-2);
    }});
  }}

  function draw(hov=-1){{
    ctx.clearRect(0,0,W,H);
    ctx.strokeStyle='#f1f5f9';ctx.lineWidth=.5;
    for(let i=1;i<5;i++){{const y=(H/5)*i;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}}
    ctx.fillStyle='#9ca3af';ctx.font='9px -apple-system,sans-serif';ctx.textAlign='left';
    for(let i=1;i<5;i++){{
      const y=(H/5)*i;
      const pval=maxP-((i/5)*(maxP-minP));
      ctx.fillText(Math.round(pval),4,y-3);
    }}

    if(chartType==='candlestick'){{
      for(let i=0;i<CHART_OPENS.length;i++){{
        const o2=CHART_OPENS[i],c2=CHART_CLOSES[i],h2=CHART_HIGHS[i],l2=CHART_LOWS[i];
        const x=i*step+step/2;
        const bull=c2>=o2;
        const alpha=hov===-1?1:(i===hov?1:.2);
        ctx.globalAlpha=alpha;
        const col=bull?'#16a34a':'#dc2626';
        ctx.strokeStyle=col;ctx.lineWidth=1;
        ctx.beginPath();ctx.moveTo(x,py(h2));ctx.lineTo(x,py(l2));ctx.stroke();
        const cw=step*.48,y1=py(Math.max(o2,c2)),y2=py(Math.min(o2,c2));
        ctx.fillStyle=bull?'rgba(22,163,74,.6)':'rgba(220,38,38,.6)';
        ctx.fillRect(x-cw/2,y1,cw,Math.max(y2-y1,2));
        ctx.strokeStyle=col;ctx.lineWidth=.6;ctx.strokeRect(x-cw/2,y1,cw,Math.max(y2-y1,2));
      }}
    }} else {{
      ctx.globalAlpha=1;
      ctx.beginPath();
      CHART_PRICES.forEach((p,i)=>{{const x=i*step+step/2;i===0?ctx.moveTo(x,py(p)):ctx.lineTo(x,py(p));}});
      ctx.strokeStyle='#1e40af';ctx.lineWidth=2;ctx.stroke();
      ctx.lineTo((CHART_PRICES.length-1)*step+step/2,H);ctx.lineTo(step/2,H);ctx.closePath();
      ctx.fillStyle='rgba(30,64,175,.06)';ctx.fill();
      CHART_PRICES.forEach((p,i)=>{{
        const x=i*step+step/2;
        ctx.globalAlpha=hov===-1?1:(i===hov?1:.2);
        ctx.beginPath();ctx.arc(x,py(p),i===hov?5:3,0,Math.PI*2);
        ctx.fillStyle=i===hov?'#1e40af':'rgba(30,64,175,.6)';ctx.fill();
      }});
    }}
    ctx.globalAlpha=1;
    if(hov>=0){{
      const x=hov*step+step/2;
      ctx.strokeStyle='rgba(30,64,175,.25)';ctx.lineWidth=.5;ctx.setLineDash([4,4]);
      ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H-12);ctx.stroke();
      ctx.setLineDash([]);
    }}
    drawTimeAxis();
  }}

  area.onmousemove=e=>{{
    const r=area.getBoundingClientRect();
    const mx=e.clientX-r.left;
    const idx=Math.min(CHART_OPENS.length-1,Math.max(0,Math.floor(mx/step)));
    draw(idx);
    if(document.getElementById('ct-date'))document.getElementById('ct-date').textContent='Today · '+TIMES[idx];
    if(document.getElementById('ct-time'))document.getElementById('ct-time').textContent=TIMES[idx];
    if(chartType==='candlestick'){{
      document.getElementById('ct-o').textContent='₹'+CHART_OPENS[idx];
      document.getElementById('ct-c').textContent='₹'+CHART_CLOSES[idx];
      document.getElementById('ct-h').textContent='₹'+CHART_HIGHS[idx];
      document.getElementById('ct-l').textContent='₹'+CHART_LOWS[idx];
      document.getElementById('chart-price').textContent='₹'+CHART_CLOSES[idx];
    }}else{{
      document.getElementById('ct-o').textContent='₹'+Math.round(CHART_PRICES[idx]);
      document.getElementById('ct-c').textContent='—';
      document.getElementById('ct-h').textContent='—';
      document.getElementById('ct-l').textContent='—';
      document.getElementById('chart-price').textContent='₹'+Math.round(CHART_PRICES[idx]);
    }}
    if(tip){{
      tip.style.opacity='1';
      let tx=mx+14;if(tx+125>W)tx=mx-140;
      tip.style.left=tx+'px';tip.style.top='8px';
    }}
  }};
  area.onmouseleave=()=>{{if(tip)tip.style.opacity='0';draw();}};
  setup();
  const ro=new ResizeObserver(setup);ro.observe(area);
}}

recalc();
</script>
</body></html>"""

components.html(APP_HTML, height=1100, scrolling=True)
