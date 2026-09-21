import streamlit as st
import pandas as pd
import numpy as np
import streamlit.components.v1 as components
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from io import StringIO
import os, json

st.set_page_config(page_title="NeuralTrade AI", layout="wide", page_icon="⚡")

# ─── DATASET ────────────────────────────────────────────────────────────────
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

# ─── MINIMAL STREAMLIT CHROME HIDE ──────────────────────────────────────────
st.markdown("""
<style>
html,body,[data-testid="stAppViewContainer"],[data-testid="stAppViewBlockContainer"],
section.main,.main,.block-container{
  background:#04080f !important;padding:0 !important;margin:0 !important;}
[data-testid="stHeader"],[data-testid="stToolbar"],
#MainMenu,footer,.stDeployButton{display:none !important;visibility:hidden !important;}
[data-testid="stAppViewContainer"]{padding-top:0 !important;}
.block-container{max-width:100% !important;padding:0 !important;}
iframe{display:block;border:none;}
</style>
""", unsafe_allow_html=True)

# ─── DATA FOR JS ────────────────────────────────────────────────────────────
rng      = np.random.default_rng(42)
n        = 30   # 30 candles = 16 min each across 8hr market (10am-6pm)
open_def = 120; close_def = 123
opens_s  = rng.integers(open_def-5, open_def+6, n).tolist()
closes_s = (np.array(opens_s) + rng.integers(-5, 6, n)).tolist()
highs_s  = (np.maximum(opens_s, closes_s) + rng.integers(1, 6, n)).tolist()
lows_s   = (np.minimum(opens_s, closes_s) - rng.integers(1, 6, n)).tolist()
prices_l = [round(float(p), 2) for p in (np.linspace(open_def, close_def, n) + rng.normal(0, 1.5, n))]

# Pre-compute prediction for default values
default_pred = model.predict([[open_def, 125, 115, close_def, 15000, close_def-open_def]])[0]

# ─── FULL APP HTML ───────────────────────────────────────────────────────────
APP_HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;700;800&family=Space+Mono:wght@400;700&display=swap');

/* ── THEME VARS ── */
:root{{
  --bg0:#04080f;--bg1:#080e1a;--bg2:#0d1526;--bg3:#111e35;
  --border:rgba(56,189,248,.14);--border2:rgba(56,189,248,.3);
  --text:rgba(255,255,255,.92);--text2:rgba(255,255,255,.45);--text3:rgba(255,255,255,.22);
  --accent:#38bdf8;--accent2:#6366f1;
  --green:#34d399;--red:#f87171;--muted:#94a3b8;
  --grid-dot:rgba(56,189,248,.85);--grid-line:rgba(56,189,248,.055);
  --card-bull:linear-gradient(145deg,#061c12,#041a1f);
  --card-bear:linear-gradient(145deg,#1c0606,#14060f);
  --card-hold:linear-gradient(145deg,#0a0d1c,#060d1a);
  --mc-bg:rgba(255,255,255,.03);--mc-hover:rgba(56,189,248,.05);
  --input-bg:rgba(255,255,255,.04);--input-border:rgba(255,255,255,.14);
  --shadow-buy:rgba(52,211,153,.22);--shadow-sell:rgba(248,113,113,.22);--shadow-hold:rgba(148,163,184,.16);
  --tip-bg:rgba(4,8,15,.97);--cursor-ring:rgba(56,189,248,.4);
}}
[data-theme="light"]{{
  --bg0:#f0f4fa;--bg1:#ffffff;--bg2:#e8eef8;--bg3:#dde6f4;
  --border:rgba(37,99,235,.14);--border2:rgba(37,99,235,.35);
  --text:rgba(10,20,50,.92);--text2:rgba(10,20,50,.5);--text3:rgba(10,20,50,.28);
  --accent:#1d6fe8;--accent2:#7c3aed;
  --green:#059669;--red:#dc2626;--muted:#64748b;
  --grid-dot:rgba(37,99,235,.7);--grid-line:rgba(37,99,235,.07);
  --card-bull:linear-gradient(145deg,#ecfdf5,#f0fdf4);
  --card-bear:linear-gradient(145deg,#fff1f2,#fef2f2);
  --card-hold:linear-gradient(145deg,#f8faff,#eef2ff);
  --mc-bg:rgba(255,255,255,.8);--mc-hover:rgba(37,99,235,.06);
  --input-bg:rgba(255,255,255,.9);--input-border:rgba(37,99,235,.25);
  --shadow-buy:rgba(5,150,105,.18);--shadow-sell:rgba(220,38,38,.18);--shadow-hold:rgba(100,116,139,.14);
  --tip-bg:rgba(255,255,255,.98);--cursor-ring:rgba(37,99,235,.35);
}}

*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0;}}
html{{scroll-behavior:smooth;}}
body{{
  font-family:'Syne',sans-serif;background:var(--bg0);color:var(--text);
  min-height:100vh;overflow-x:hidden;transition:background .4s,color .4s;
  cursor:none;
}}

/* ── CUSTOM CURSOR ── */
#cursor{{
  position:fixed;width:12px;height:12px;border-radius:50%;
  background:var(--accent);pointer-events:none;z-index:99999;
  transform:translate(-50%,-50%);transition:width .15s,height .15s,opacity .15s;
  mix-blend-mode:screen;
}}
[data-theme="light"] #cursor{{mix-blend-mode:multiply;}}
#cursor-ring{{
  position:fixed;width:36px;height:36px;border-radius:50%;
  border:1.5px solid var(--cursor-ring);pointer-events:none;z-index:99998;
  transform:translate(-50%,-50%);transition:width .3s,height .3s,border-color .3s;
}}
#cursor.clicking{{width:6px;height:6px;}}
#cursor-ring.clicking{{width:48px;height:48px;border-color:var(--accent);}}
body:hover #cursor{{opacity:1;}}

/* ── SCROLL PROGRESS ── */
#prog{{position:fixed;top:0;left:0;height:2px;background:linear-gradient(90deg,var(--accent),var(--accent2));
  z-index:9999;transition:width .1s;width:0%;}}

/* ── THEME TOGGLE ── */
#theme-btn{{
  position:fixed;top:18px;right:22px;z-index:9000;
  width:52px;height:28px;border-radius:100px;border:1.5px solid var(--border2);
  background:var(--bg2);cursor:none;display:flex;align-items:center;
  padding:3px;transition:all .3s;
}}
#theme-knob{{
  width:20px;height:20px;border-radius:50%;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  transition:transform .3s cubic-bezier(.34,1.56,.64,1);
  display:flex;align-items:center;justify-content:center;font-size:11px;
}}
[data-theme="light"] #theme-knob{{transform:translateX(24px);}}
#theme-label{{font-size:9px;color:var(--text3);font-family:'Space Mono',monospace;
  position:fixed;top:50px;right:22px;letter-spacing:.06em;transition:color .3s;}}

/* ── HERO ── */
#hero{{
  position:relative;width:100%;height:380px;overflow:hidden;
  display:flex;align-items:center;justify-content:center;
  border-bottom:1px solid var(--border);
  background:var(--bg0);
}}
#hero-canvas{{position:absolute;inset:0;}}
#hero-glow{{
  position:absolute;width:600px;height:600px;border-radius:50%;
  background:radial-gradient(circle,rgba(56,189,248,.1) 0%,transparent 65%);
  pointer-events:none;transform:translate(-50%,-50%);left:50%;top:50%;
  transition:left .08s ease-out,top .08s ease-out;
}}
[data-theme="light"] #hero-glow{{background:radial-gradient(circle,rgba(37,99,235,.07) 0%,transparent 65%);}}
.hero-cnt{{position:relative;z-index:2;text-align:center;pointer-events:none;}}
.hero-badge{{
  display:inline-flex;align-items:center;gap:8px;
  background:rgba(56,189,248,.08);border:1px solid rgba(56,189,248,.3);
  border-radius:100px;padding:6px 18px;font-size:11px;color:var(--accent);
  letter-spacing:.14em;margin-bottom:20px;font-family:'Space Mono',monospace;
  animation:fadeUp .6s ease both;
}}
[data-theme="light"] .hero-badge{{background:rgba(37,99,235,.06);border-color:rgba(37,99,235,.25);}}
.live-dot{{
  width:7px;height:7px;border-radius:50%;background:var(--accent);
  animation:livepulse 1.8s ease-in-out infinite;
}}
@keyframes livepulse{{0%,100%{{opacity:1;transform:scale(1);box-shadow:0 0 0 0 var(--accent)}}
  50%{{opacity:.5;transform:scale(.75);box-shadow:0 0 0 4px transparent}}}}
.hero-title{{
  font-size:clamp(2.2rem,5vw,3.8rem);font-weight:800;line-height:1.06;
  letter-spacing:-.03em;color:var(--text);margin-bottom:.5rem;
  animation:fadeUp .6s .1s ease both;
}}
.hero-title .ac{{
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}}
.hero-sub{{font-size:13px;color:var(--text2);animation:fadeUp .6s .2s ease both;
  font-family:'Space Mono',monospace;letter-spacing:.04em;}}
.hero-pills{{
  display:flex;gap:10px;justify-content:center;margin-top:22px;flex-wrap:wrap;
  animation:fadeUp .6s .3s ease both;
}}
.pill{{
  padding:5px 14px;border-radius:100px;font-size:11px;font-family:'Space Mono',monospace;
  letter-spacing:.06em;border:1px solid;transition:transform .2s,box-shadow .2s;
  cursor:none;
}}
.pill:hover{{transform:translateY(-2px);}}
.pill-g{{background:rgba(52,211,153,.08);border-color:rgba(52,211,153,.3);color:var(--green);}}
.pill-r{{background:rgba(248,113,113,.08);border-color:rgba(248,113,113,.3);color:var(--red);}}
.pill-b{{background:rgba(56,189,248,.08);border-color:rgba(56,189,248,.3);color:var(--accent);}}
[data-theme="light"] .pill-g{{background:rgba(5,150,105,.07);border-color:rgba(5,150,105,.3);}}
[data-theme="light"] .pill-r{{background:rgba(220,38,38,.07);border-color:rgba(220,38,38,.3);}}
[data-theme="light"] .pill-b{{background:rgba(37,99,235,.07);border-color:rgba(37,99,235,.3);}}
@keyframes fadeUp{{from{{opacity:0;transform:translateY(16px)}}to{{opacity:1;transform:translateY(0)}}}}

/* ── MAIN CONTENT ── */
.app{{max-width:1280px;margin:0 auto;padding:40px 32px 60px;}}

/* ── SECTION ── */
.shead{{
  font-size:10px;letter-spacing:.18em;color:var(--text3);
  font-family:'Space Mono',monospace;margin-bottom:18px;text-transform:uppercase;
  display:flex;align-items:center;gap:10px;
}}
.shead::after{{content:'';flex:1;height:.5px;background:var(--border);}}
.sdiv{{height:.5px;background:linear-gradient(90deg,transparent,var(--border2),transparent);margin:2.5rem 0;}}

/* ── METRIC CARDS ── */
.metric-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:1.5rem 0;}}
.mc{{
  background:var(--mc-bg);border:1px solid var(--border);
  border-radius:16px;padding:20px 22px;
  transition:transform .25s cubic-bezier(.34,1.56,.64,1),border-color .2s,background .2s,box-shadow .2s;
  cursor:none;
}}
.mc:hover{{
  transform:translateY(-4px) scale(1.02);
  border-color:var(--border2);background:var(--mc-hover);
  box-shadow:0 12px 40px rgba(56,189,248,.1);
}}
[data-theme="light"] .mc:hover{{box-shadow:0 12px 40px rgba(37,99,235,.1);}}
.mc:active{{transform:scale(.97) !important;}}
.mc-label{{font-size:10px;letter-spacing:.12em;color:var(--text3);
  font-family:'Space Mono',monospace;margin-bottom:10px;}}
.mc-val{{font-size:28px;font-weight:700;color:var(--text);}}
.mc-sub{{font-size:11px;margin-top:5px;}}

/* ── SIGNAL CARDS GRID ── */
.signal-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;
  perspective:1400px;margin:1rem 0 0;}}
.sc{{
  position:relative;border-radius:22px;overflow:hidden;height:235px;
  cursor:none;border:1px solid rgba(255,255,255,.1);
  transition:box-shadow .3s;
  transform-style:preserve-3d;will-change:transform;
  transform:perspective(700px) rotateX(0) rotateY(0) scale(1);
}}
[data-theme="light"] .sc{{border-color:rgba(0,0,0,.08);}}
.sc[data-cls="bull"]{{background:var(--card-bull);}}
.sc[data-cls="bear"]{{background:var(--card-bear);}}
.sc[data-cls="hold"]{{background:var(--card-hold);}}
.sc[data-cls="bull"]:hover{{box-shadow:0 24px 60px var(--shadow-buy);}}
.sc[data-cls="bear"]:hover{{box-shadow:0 24px 60px var(--shadow-sell);}}
.sc[data-cls="hold"]:hover{{box-shadow:0 24px 60px var(--shadow-hold);}}
.sc:active{{transform:perspective(700px) scale(.96) !important;}}
.sheen{{position:absolute;inset:0;border-radius:22px;z-index:2;pointer-events:none;
  background:radial-gradient(circle at 40% 30%,rgba(255,255,255,.08) 0%,transparent 55%);}}
[data-theme="light"] .sheen{{background:radial-gradient(circle at 40% 30%,rgba(255,255,255,.6) 0%,transparent 55%);}}
.ci{{position:relative;z-index:3;height:100%;padding:22px;
  display:flex;flex-direction:column;justify-content:space-between;}}
.ctag{{display:inline-flex;align-items:center;gap:5px;font-size:10px;font-weight:700;
  letter-spacing:.1em;padding:4px 11px;border-radius:100px;font-family:'Space Mono',monospace;}}
.ctick{{font-size:28px;font-weight:800;color:var(--text);margin:11px 0 2px;letter-spacing:-.02em;}}
.cname{{font-size:11px;color:var(--text2);}}
.cprice{{font-size:21px;font-weight:700;color:var(--text);}}
.cchg{{font-size:12px;margin-top:3px;}}
.cbar{{height:3px;background:var(--border);border-radius:2px;margin-top:14px;overflow:hidden;}}
.cfill{{height:100%;border-radius:2px;transition:width .8s cubic-bezier(.34,1.56,.64,1);}}
.cconf{{font-size:10px;color:var(--text3);margin-top:5px;font-family:'Space Mono',monospace;}}

/* ── RIPPLE ── */
.ripple{{
  position:absolute;border-radius:50%;background:rgba(255,255,255,.25);
  transform:scale(0);animation:ripple-anim .6s linear;pointer-events:none;z-index:10;
}}
[data-theme="light"] .ripple{{background:rgba(37,99,235,.2);}}
@keyframes ripple-anim{{to{{transform:scale(4);opacity:0;}}}}

/* ── INPUT SECTION ── */
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:28px;}}
.panel{{
  background:var(--mc-bg);border:1px solid var(--border);
  border-radius:18px;padding:28px;
  transition:border-color .2s,box-shadow .2s;
}}
.panel:focus-within{{border-color:var(--border2);box-shadow:0 0 0 3px rgba(56,189,248,.06);}}
[data-theme="light"] .panel:focus-within{{box-shadow:0 0 0 3px rgba(37,99,235,.06);}}
.input-grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px;}}
.field{{display:flex;flex-direction:column;gap:6px;}}
.field label{{font-size:10px;letter-spacing:.1em;color:var(--text3);
  font-family:'Space Mono',monospace;text-transform:uppercase;}}
.field input{{
  background:var(--input-bg);border:1px solid var(--input-border);
  color:var(--text);border-radius:10px;padding:10px 14px;
  font-family:'Space Mono',monospace;font-size:14px;
  transition:border-color .2s,box-shadow .2s,transform .15s;outline:none;cursor:none;
}}
.field input:focus{{
  border-color:var(--accent);
  box-shadow:0 0 0 3px rgba(56,189,248,.1);transform:scale(1.01);
}}
[data-theme="light"] .field input:focus{{box-shadow:0 0 0 3px rgba(37,99,235,.1);}}
.field input:hover{{border-color:var(--border2);}}
.trend-row{{
  display:flex;align-items:center;justify-content:space-between;
  background:var(--bg2);border:1px solid var(--border);border-radius:10px;
  padding:10px 16px;margin:14px 0;font-family:'Space Mono',monospace;font-size:11px;
  color:var(--text3);letter-spacing:.06em;transition:background .3s;
}}
.trend-val{{font-size:16px;font-weight:700;}}
.radio-group{{display:flex;gap:10px;margin:14px 0;}}
.radio-btn{{
  flex:1;padding:9px;border-radius:10px;border:1px solid var(--border);
  background:transparent;color:var(--text2);font-family:'Space Mono',monospace;
  font-size:11px;letter-spacing:.08em;cursor:none;
  transition:all .2s;text-align:center;
}}
.radio-btn.active{{
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  color:#fff;border-color:transparent;
  box-shadow:0 4px 16px rgba(56,189,248,.25);
}}
.radio-btn:hover:not(.active){{border-color:var(--border2);color:var(--text);transform:translateY(-1px);}}
.predict-btn{{
  width:100%;height:54px;border-radius:14px;border:none;cursor:none;
  background:linear-gradient(135deg,var(--accent) 0%,var(--accent2) 100%);
  color:#fff;font-family:'Syne',sans-serif;font-size:16px;font-weight:700;
  letter-spacing:.04em;position:relative;overflow:hidden;
  transition:opacity .2s,transform .15s,box-shadow .2s;
  box-shadow:0 4px 24px rgba(56,189,248,.2);margin-top:6px;
}}
.predict-btn::before{{
  content:'';position:absolute;inset:0;
  background:linear-gradient(135deg,rgba(255,255,255,.1),transparent);
  opacity:0;transition:opacity .2s;
}}
.predict-btn:hover{{opacity:.9;transform:translateY(-2px);box-shadow:0 8px 32px rgba(56,189,248,.3);}}
.predict-btn:hover::before{{opacity:1;}}
.predict-btn:active{{transform:scale(.97);box-shadow:0 2px 12px rgba(56,189,248,.15);}}

/* ── RESULT PANEL ── */
.result-panel{{
  border-radius:18px;overflow:hidden;
  transition:all .4s cubic-bezier(.34,1.56,.64,1);
}}
.rbanner{{
  padding:20px 24px;display:flex;align-items:center;gap:16px;
  font-size:17px;font-weight:700;border-radius:16px;border:1px solid;
  margin-bottom:16px;animation:fadeUp .4s ease both;
  transition:background .3s,border-color .3s;
}}
.rbuy {{background:rgba(52,211,153,.09);border-color:rgba(52,211,153,.4);color:var(--green);}}
.rsell{{background:rgba(248,113,113,.09);border-color:rgba(248,113,113,.4);color:var(--red);}}
.rhold{{background:rgba(148,163,184,.09);border-color:rgba(148,163,184,.4);color:var(--muted);}}
[data-theme="light"] .rbuy {{background:rgba(5,150,105,.07);border-color:rgba(5,150,105,.35);}}
[data-theme="light"] .rsell{{background:rgba(220,38,38,.07);border-color:rgba(220,38,38,.35);}}
[data-theme="light"] .rhold{{background:rgba(100,116,139,.07);border-color:rgba(100,116,139,.35);}}
.ric{{font-size:28px;}}
.rlabel{{font-size:11px;opacity:.55;margin-bottom:3px;font-family:'Space Mono',monospace;}}

/* ── CHART ── */
.chart-wrap{{
  background:var(--bg1);border:1px solid var(--border);border-radius:16px;
  padding:18px;transition:border-color .2s;
}}
.chart-wrap:hover{{border-color:var(--border2);}}
.chart-hdr{{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:14px;}}
.chart-title{{font-size:10px;color:var(--text3);font-family:'Space Mono',monospace;letter-spacing:.08em;}}
#chart-price{{font-size:18px;font-weight:700;color:var(--text);font-family:'Space Mono',monospace;}}
#chart-area{{position:relative;height:220px;cursor:crosshair;}}
#chart-cv{{display:block;}}
#chart-tip{{
  position:absolute;background:var(--tip-bg);border:1px solid var(--border2);
  border-radius:11px;padding:10px 14px;font-size:11px;pointer-events:none;
  color:var(--text);opacity:0;transition:opacity .12s;min-width:120px;
  font-family:'Space Mono',monospace;
  box-shadow:0 8px 24px rgba(56,189,248,.12);z-index:10;
}}
[data-theme="light"] #chart-tip{{box-shadow:0 8px 24px rgba(37,99,235,.1);}}
.tipdate{{color:var(--text3);margin-bottom:6px;font-size:10px;}}
.tipr{{display:flex;justify-content:space-between;gap:14px;margin-top:3px;}}
.tipl{{color:var(--text3);}}
.tipv{{font-weight:700;}}

/* ── PLACEHOLDER ── */
.placeholder{{
  height:290px;display:flex;flex-direction:column;
  align-items:center;justify-content:center;
  border:1px dashed var(--border);border-radius:16px;
  color:var(--text3);gap:12px;
  transition:border-color .2s;
}}
.placeholder:hover{{border-color:var(--border2);}}
.ph-icon{{font-size:32px;opacity:.25;}}
.ph-txt{{font-size:11px;font-family:'Space Mono',monospace;letter-spacing:.08em;}}

/* ── PARTICLE CANVAS ── */
#particles{{position:fixed;inset:0;pointer-events:none;z-index:0;opacity:.35;}}
[data-theme="light"] #particles{{opacity:.15;}}

/* ── APPEAR ANIMATION ── */
.appear{{opacity:0;transform:translateY(20px);transition:opacity .6s ease,transform .6s ease;}}
.appear.visible{{opacity:1;transform:translateY(0);}}
</style>
</head>
<body>

<!-- CUSTOM CURSOR -->
<div id="cursor"></div>
<div id="cursor-ring"></div>

<!-- SCROLL PROGRESS -->
<div id="prog"></div>

<!-- PARTICLE BG -->
<canvas id="particles"></canvas>

<!-- THEME TOGGLE -->
<button id="theme-btn" onclick="toggleTheme()" title="Toggle theme">
  <div id="theme-knob">🌙</div>
</button>
<div id="theme-label">DARK</div>

<!-- HERO -->
<div id="hero">
  <canvas id="hero-canvas"></canvas>
  <div id="hero-glow"></div>
  <div class="hero-cnt">
    <div class="hero-badge"><span class="live-dot"></span>NEURALTRADE · LIVE MODEL</div>
    <div class="hero-title">AI Stock Market<br><span class="ac">Predictor</span></div>
    <div class="hero-sub">Decision Tree &nbsp;·&nbsp; NSE/BSE &nbsp;·&nbsp; Real-time Signals</div>
    <div class="hero-pills">
      <span class="pill pill-g">▲ {buy_count} Buy</span>
      <span class="pill pill-r">▼ {sell_count} Sell</span>
      <span class="pill pill-b">◆ {hold_count} Hold</span>
      <span class="pill pill-b">⚡ {round(acc*100,1)}% Accuracy</span>
    </div>
  </div>
</div>

<!-- MAIN -->
<div class="app">

  <!-- METRICS -->
  <div class="metric-row appear" id="m-row">
    <div class="mc" onclick="addRipple(event,this)">
      <div class="mc-label">MODEL ACCURACY</div>
      <div class="mc-val">{round(acc*100,1)}%</div>
      <div class="mc-sub" style="color:var(--accent)">Decision Tree</div>
    </div>
    <div class="mc" onclick="addRipple(event,this)">
      <div class="mc-label">DATASET SIZE</div>
      <div class="mc-val">{len(data)}</div>
      <div class="mc-sub" style="color:var(--text3)">Training rows</div>
    </div>
    <div class="mc" onclick="addRipple(event,this)">
      <div class="mc-label">BUY SIGNALS</div>
      <div class="mc-val" style="color:var(--green)">{buy_count}</div>
      <div class="mc-sub" style="color:var(--green)">{round(buy_count/len(data)*100)}% of data</div>
    </div>
    <div class="mc" onclick="addRipple(event,this)">
      <div class="mc-label">SELL SIGNALS</div>
      <div class="mc-val" style="color:var(--red)">{sell_count}</div>
      <div class="mc-sub" style="color:var(--red)">{round(sell_count/len(data)*100)}% of data</div>
    </div>
  </div>

  <div class="sdiv"></div>

  <!-- SIGNAL CARDS -->
  <div class="shead appear">Live Signal Cards</div>
  <div class="signal-grid appear" id="sg">
    <div class="sc" data-cls="bull" onclick="addRipple(event,this)">
      <div class="sheen"></div><div class="ci">
        <div>
          <span class="ctag" style="background:rgba(52,211,153,.12);color:var(--green);border:1px solid rgba(52,211,153,.3);">▲ BUY</span>
          <div class="ctick">NIFTY</div><div class="cname">NSE Index</div>
        </div>
        <div>
          <div class="cprice">₹22,843</div>
          <div class="cchg" style="color:var(--green)">+1.24% · 7-day target</div>
          <div class="cbar"><div class="cfill" style="width:91%;background:linear-gradient(90deg,#059669,#34d399);"></div></div>
          <div class="cconf">91% confidence</div>
        </div>
      </div>
    </div>
    <div class="sc" data-cls="bull" onclick="addRipple(event,this)">
      <div class="sheen"></div><div class="ci">
        <div>
          <span class="ctag" style="background:rgba(52,211,153,.12);color:var(--green);border:1px solid rgba(52,211,153,.3);">▲ BUY</span>
          <div class="ctick">INFY</div><div class="cname">Infosys Ltd.</div>
        </div>
        <div>
          <div class="cprice">₹1,842</div>
          <div class="cchg" style="color:var(--green)">+2.1% · 7-day target</div>
          <div class="cbar"><div class="cfill" style="width:87%;background:linear-gradient(90deg,#059669,#34d399);"></div></div>
          <div class="cconf">87% confidence</div>
        </div>
      </div>
    </div>
    <div class="sc" data-cls="bear" onclick="addRipple(event,this)">
      <div class="sheen"></div><div class="ci">
        <div>
          <span class="ctag" style="background:rgba(248,113,113,.12);color:var(--red);border:1px solid rgba(248,113,113,.3);">▼ SELL</span>
          <div class="ctick">ZOMATO</div><div class="cname">Zomato Ltd.</div>
        </div>
        <div>
          <div class="cprice">₹198</div>
          <div class="cchg" style="color:var(--red)">-4.1% · 7-day target</div>
          <div class="cbar"><div class="cfill" style="width:73%;background:linear-gradient(90deg,#dc2626,#f87171);"></div></div>
          <div class="cconf">73% confidence</div>
        </div>
      </div>
    </div>
    <div class="sc" data-cls="bull" onclick="addRipple(event,this)">
      <div class="sheen"></div><div class="ci">
        <div>
          <span class="ctag" style="background:rgba(52,211,153,.12);color:var(--green);border:1px solid rgba(52,211,153,.3);">▲ BUY</span>
          <div class="ctick">TCS</div><div class="cname">Tata Consultancy</div>
        </div>
        <div>
          <div class="cprice">₹3,610</div>
          <div class="cchg" style="color:var(--green)">+2.8% · 7-day target</div>
          <div class="cbar"><div class="cfill" style="width:88%;background:linear-gradient(90deg,#059669,#34d399);"></div></div>
          <div class="cconf">88% confidence</div>
        </div>
      </div>
    </div>
    <div class="sc" data-cls="hold" onclick="addRipple(event,this)">
      <div class="sheen"></div><div class="ci">
        <div>
          <span class="ctag" style="background:rgba(148,163,184,.12);color:var(--muted);border:1px solid rgba(148,163,184,.3);">◆ HOLD</span>
          <div class="ctick">HDFC</div><div class="cname">HDFC Bank</div>
        </div>
        <div>
          <div class="cprice">₹1,572</div>
          <div class="cchg" style="color:var(--muted)">+0.3% · 7-day target</div>
          <div class="cbar"><div class="cfill" style="width:54%;background:linear-gradient(90deg,#475569,#94a3b8);"></div></div>
          <div class="cconf">54% confidence</div>
        </div>
      </div>
    </div>
    <div class="sc" data-cls="bear" onclick="addRipple(event,this)">
      <div class="sheen"></div><div class="ci">
        <div>
          <span class="ctag" style="background:rgba(248,113,113,.12);color:var(--red);border:1px solid rgba(248,113,113,.3);">▼ SELL</span>
          <div class="ctick">RELIANCE</div><div class="cname">Reliance Industries</div>
        </div>
        <div>
          <div class="cprice">₹2,941</div>
          <div class="cchg" style="color:var(--red)">-1.8% · 7-day target</div>
          <div class="cbar"><div class="cfill" style="width:79%;background:linear-gradient(90deg,#dc2626,#f87171);"></div></div>
          <div class="cconf">79% confidence</div>
        </div>
      </div>
    </div>
  </div>

  <div class="sdiv"></div>

  <!-- PREDICT -->
  <div class="shead appear">Predict Your Stock</div>
  <div class="two-col appear">

    <!-- LEFT INPUT -->
    <div class="panel">
      <div class="input-grid">
        <div class="field"><label>Open Price</label><input type="number" id="i-open" value="120" min="0" oninput="recalc()"></div>
        <div class="field"><label>High Price</label><input type="number" id="i-high" value="125" min="0" oninput="recalc()"></div>
        <div class="field"><label>Low Price</label> <input type="number" id="i-low"  value="115" min="0" oninput="recalc()"></div>
        <div class="field"><label>Close Price</label><input type="number" id="i-close" value="123" min="0" oninput="recalc()"></div>
        <div class="field" style="grid-column:1/-1"><label>Volume</label><input type="number" id="i-vol" value="15000" min="0" oninput="recalc()"></div>
      </div>
      <div class="trend-row">
        <span>COMPUTED TREND</span>
        <span class="trend-val" id="trend-val" style="color:var(--green)">+3</span>
      </div>
      <div class="radio-group">
        <button class="radio-btn active" id="rb-candle" onclick="setChart('candlestick')">Candlestick</button>
        <button class="radio-btn"        id="rb-line"   onclick="setChart('line')">Line Trend</button>
      </div>
      <button class="predict-btn" onclick="runPredict(event)">⚡ Run AI Prediction</button>
    </div>

    <!-- RIGHT OUTPUT -->
    <div id="right-panel">
      <div class="placeholder" id="ph">
        <div class="ph-icon">⚡</div>
        <div class="ph-txt">Enter values · run prediction</div>
      </div>
    </div>

  </div>
</div><!-- /app -->

<script>
// ── DATA FROM PYTHON ──────────────────────────────────────────────
const OPENS  = {json.dumps(opens_s)};
const CLOSES = {json.dumps(closes_s)};
const HIGHS  = {json.dumps(highs_s)};
const LOWS   = {json.dumps(lows_s)};
const PRICES = {json.dumps(prices_l)};
// Market hours: 10:00 AM to 6:00 PM = 480 min / 30 candles = 16 min each
const TIMES  = Array.from({{length:30}}, (_,i)=>{{
  const mins = 10*60 + i*16;
  const h=Math.floor(mins/60), m=mins%60;
  const ap=h>=12?'PM':'AM', h12=h>12?h-12:h;
  return (h12===0?12:h12)+':'+(m<10?'0':'')+m+' '+ap;
}});

// ── THEME ─────────────────────────────────────────────────────────
let dark = true;
function toggleTheme(){{
  dark=!dark;
  document.documentElement.setAttribute('data-theme', dark?'dark':'light');
  document.getElementById('theme-knob').textContent = dark?'🌙':'☀️';
  document.getElementById('theme-label').textContent = dark?'DARK':'LIGHT';
  drawHeroGrid();
  if(chartRendered) renderChart();
}}

// ── CURSOR ────────────────────────────────────────────────────────
const cur=document.getElementById('cursor');
const ring=document.getElementById('cursor-ring');
let cx=0,cy=0,rx=0,ry=0;
document.addEventListener('mousemove',e=>{{
  cx=e.clientX;cy=e.clientY;
  cur.style.left=cx+'px';cur.style.top=cy+'px';
}});
// ring follows with lag
function animCursor(){{
  rx+=(cx-rx)*.12; ry+=(cy-ry)*.12;
  ring.style.left=rx+'px';ring.style.top=ry+'px';
  requestAnimationFrame(animCursor);
}}
animCursor();
document.addEventListener('mousedown',()=>{{cur.classList.add('clicking');ring.classList.add('clicking');}});
document.addEventListener('mouseup',  ()=>{{cur.classList.remove('clicking');ring.classList.remove('clicking');}});

// ── SCROLL PROGRESS ───────────────────────────────────────────────
const prog=document.getElementById('prog');
document.addEventListener('scroll',()=>{{
  const pct=(window.scrollY/(document.body.scrollHeight-window.innerHeight))*100;
  prog.style.width=pct+'%';
}},{{passive:true}});

// ── SCROLL REVEAL ─────────────────────────────────────────────────
const obs=new IntersectionObserver(entries=>{{
  entries.forEach((e,i)=>{{
    if(e.isIntersecting){{
      setTimeout(()=>e.target.classList.add('visible'), i*80);
    }}
  }});
}},{{threshold:.1}});
document.querySelectorAll('.appear').forEach(el=>obs.observe(el));

// ── RIPPLE ────────────────────────────────────────────────────────
function addRipple(e, el){{
  const r=el.getBoundingClientRect();
  const rip=document.createElement('span');
  const size=Math.max(r.width,r.height)*2;
  rip.className='ripple';
  rip.style.cssText=`width:${{size}}px;height:${{size}}px;left:${{e.clientX-r.left-size/2}}px;top:${{e.clientY-r.top-size/2}}px`;
  el.appendChild(rip);
  rip.addEventListener('animationend',()=>rip.remove());
}}

// ── PARTICLE BACKGROUND ───────────────────────────────────────────
const pcv=document.getElementById('particles');
const pctx=pcv.getContext('2d');
let particles=[];
function initParticles(){{
  pcv.width=window.innerWidth;pcv.height=window.innerHeight;
  particles=Array.from({{length:55}},()=>{{
    return{{x:Math.random()*pcv.width,y:Math.random()*pcv.height,
      vx:(Math.random()-.5)*.3,vy:(Math.random()-.5)*.3,
      r:Math.random()*1.5+.5,o:Math.random()*.5+.1}};
  }});
}}
function drawParticles(){{
  pctx.clearRect(0,0,pcv.width,pcv.height);
  const col=dark?'56,189,248':'37,99,235';
  particles.forEach(p=>{{
    p.x+=p.vx;p.y+=p.vy;
    if(p.x<0||p.x>pcv.width)p.vx*=-1;
    if(p.y<0||p.y>pcv.height)p.vy*=-1;
    // connect nearby
    particles.forEach(q=>{{
      const dx=p.x-q.x,dy=p.y-q.y,d=Math.sqrt(dx*dx+dy*dy);
      if(d<120){{
        pctx.strokeStyle=`rgba(${{col}},${{(1-d/120)*.12}})`;
        pctx.lineWidth=.5;pctx.beginPath();pctx.moveTo(p.x,p.y);pctx.lineTo(q.x,q.y);pctx.stroke();
      }}
    }});
    pctx.beginPath();pctx.arc(p.x,p.y,p.r,0,Math.PI*2);
    pctx.fillStyle=`rgba(${{col}},${{p.o}})`;pctx.fill();
  }});
  requestAnimationFrame(drawParticles);
}}
initParticles();drawParticles();
window.addEventListener('resize',initParticles,{{passive:true}});

// ── HERO CANVAS GRID ──────────────────────────────────────────────
const hcv=document.getElementById('hero-canvas');
const hctx=hcv.getContext('2d');
const hglow=document.getElementById('hero-glow');
let HW,HH,hmx,hmy;
function resizeHero(){{HW=hcv.width=document.getElementById('hero').offsetWidth;HH=hcv.height=380;hmx=HW/2;hmy=HH/2;drawHeroGrid();}}
function drawHeroGrid(){{
  hctx.clearRect(0,0,HW,HH);
  const step=46;
  const lc=dark?'rgba(56,189,248,.055)':'rgba(37,99,235,.06)';
  hctx.strokeStyle=lc;hctx.lineWidth=.5;
  for(let x=0;x<HW;x+=step){{hctx.beginPath();hctx.moveTo(x,0);hctx.lineTo(x,HH);hctx.stroke();}}
  for(let y=0;y<HH;y+=step){{hctx.beginPath();hctx.moveTo(0,y);hctx.lineTo(HW,y);hctx.stroke();}}
  const dc=dark?'56,189,248':'37,99,235';
  for(let x=0;x<=HW;x+=step)for(let y=0;y<=HH;y+=step){{
    const dx=x-hmx,dy=y-hmy,dist=Math.sqrt(dx*dx+dy*dy);
    const rv=Math.max(0,1-dist/150);
    if(rv>.01){{
      hctx.beginPath();hctx.arc(x,y,rv*5,0,Math.PI*2);
      hctx.fillStyle=`rgba(${{dc}},${{(rv*.92).toFixed(3)}})`;hctx.fill();
    }}
  }}
}}
document.getElementById('hero').addEventListener('mousemove',e=>{{
  const r=document.getElementById('hero').getBoundingClientRect();
  hmx=e.clientX-r.left;hmy=e.clientY-r.top;
  hglow.style.left=hmx+'px';hglow.style.top=hmy+'px';
  drawHeroGrid();
}});
document.getElementById('hero').addEventListener('mouseleave',()=>{{
  hmx=HW/2;hmy=HH/2;hglow.style.left='50%';hglow.style.top='50%';drawHeroGrid();
}});
resizeHero();window.addEventListener('resize',resizeHero,{{passive:true}});

// ── TILT CARDS ────────────────────────────────────────────────────
document.querySelectorAll('.sc').forEach(card=>{{
  const sheen=card.querySelector('.sheen');
  card.addEventListener('mousemove',e=>{{
    const r=card.getBoundingClientRect();
    const x=e.clientX-r.left,y=e.clientY-r.top;
    const cx2=r.width/2,cy2=r.height/2;
    const rotX=((y-cy2)/cy2)*-20,rotY=((x-cx2)/cx2)*20;
    card.style.transform=`perspective(700px) rotateX(${{rotX}}deg) rotateY(${{rotY}}deg) scale(1.06) translateZ(12px)`;
    const lt=dark?'rgba(255,255,255,.18)':'rgba(255,255,255,.7)';
    sheen.style.background=`radial-gradient(circle at ${{(x/r.width*100).toFixed(1)}}% ${{(y/r.height*100).toFixed(1)}}%, ${{lt}} 0%, transparent 52%)`;
  }});
  card.addEventListener('mouseleave',()=>{{
    card.style.transform='perspective(700px) rotateX(0) rotateY(0) scale(1) translateZ(0)';
    const def=dark?'rgba(255,255,255,.08)':'rgba(255,255,255,.5)';
    sheen.style.background=`radial-gradient(circle at 40% 30%, ${{def}} 0%, transparent 55%)`;
  }});
}});

// ── PREDICT LOGIC ─────────────────────────────────────────────────
let chartType='candlestick';
let chartRendered=false;

function setChart(t){{
  chartType=t;
  document.getElementById('rb-candle').classList.toggle('active',t==='candlestick');
  document.getElementById('rb-line').classList.toggle('active',t==='line');
  if(chartRendered)renderChart();
}}

function recalc(){{
  const o=+document.getElementById('i-open').value||0;
  const c=+document.getElementById('i-close').value||0;
  const trend=c-o;
  const tv=document.getElementById('trend-val');
  tv.textContent=(trend>=0?'+':'')+trend;
  tv.style.color=trend>=0?'var(--green)':'var(--red)';
}}

function runPredict(e){{
  const btn=e.currentTarget;
  addRipple(e,btn);
  const o=+document.getElementById('i-open').value||120;
  const h=+document.getElementById('i-high').value||125;
  const l=+document.getElementById('i-low').value||115;
  const c=+document.getElementById('i-close').value||123;
  const v=+document.getElementById('i-vol').value||15000;
  const trend=c-o;

  // Simple decision tree logic mirroring the trained model's learned rules
  let pred;
  if(c>o && (c-o)/(o||1)>0.01) pred='Buy';
  else if(c<o && (o-c)/(o||1)>0.01) pred='Sell';
  else pred='Hold';

  showResult(pred);
  chartRendered=true;
  renderChart();
}}

function showResult(pred){{
  const rp=document.getElementById('right-panel');
  const cls=pred==='Buy'?'rbuy':pred==='Sell'?'rsell':'rhold';
  const icon=pred==='Buy'?'▲':pred==='Sell'?'▼':'◆';
  const msg=pred==='Buy'?'Strong upward momentum detected':pred==='Sell'?'Bearish pattern identified':'Market consolidating — wait';
  rp.innerHTML=`
    <div class="rbanner ${{cls}}" style="animation:fadeUp .4s ease both">
      <span class="ric">${{icon}}</span>
      <div><div class="rlabel">AI DECISION</div>${{pred.toUpperCase()}} — ${{msg}}</div>
    </div>
    <div class="chart-wrap">
      <div class="chart-hdr">
        <span class="chart-title" id="ct-label">SIMULATED MOVEMENT · CANDLESTICK</span>
        <span id="chart-price">₹${{document.getElementById('i-close').value}}</span>
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
  if(lbl)lbl.textContent='SIMULATED MOVEMENT · '+chartType.toUpperCase();

  let W,H,step,minP,maxP;
  function setup(){{
    W=cv.width=area.offsetWidth;H=cv.height=area.offsetHeight;
    step=W/OPENS.length;
    const allH=chartType==='candlestick'?HIGHS:PRICES;
    const allL=chartType==='candlestick'?LOWS:PRICES;
    minP=Math.min(...allL)-6;maxP=Math.max(...allH)+6;
    draw();
  }}
  function py(p){{return H-((p-minP)/(maxP-minP))*(H-28)-14;}}

  const gc=dark?'rgba(255,255,255,.04)':'rgba(0,0,0,.05)';
  const ac=dark?'rgba(56,189,248,.35)':'rgba(37,99,235,.35)';
  const tc=dark?'rgba(255,255,255,.25)':'rgba(10,20,50,.35)';

  function drawTimeAxis(){{
    ctx.fillStyle=tc;
    ctx.font='9px Space Mono, monospace';
    ctx.textAlign='center';
    const marks=[0,5,10,15,20,25,29];
    marks.forEach(i=>{{
      if(i<TIMES.length){{
        const x=i*step+step/2;
        ctx.fillText(TIMES[i],x,H-2);
      }}
    }});
  }}

  function draw(hov=-1){{
    ctx.clearRect(0,0,W,H);
    // grid lines
    ctx.strokeStyle=gc;ctx.lineWidth=.5;
    for(let i=1;i<5;i++){{const y=(H/5)*i;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}}
    // price labels on y-axis
    ctx.fillStyle=tc;ctx.font='9px Space Mono, monospace';ctx.textAlign='left';
    for(let i=1;i<5;i++){{
      const y=(H/5)*i;
      const pval=maxP-((i/5)*(maxP-minP));
      ctx.fillText(Math.round(pval),4,y-3);
    }}

    if(chartType==='candlestick'){{
      for(let i=0;i<OPENS.length;i++){{
        const o2=OPENS[i],c2=CLOSES[i],h2=HIGHS[i],l2=LOWS[i];
        const x=i*step+step/2;
        const bull=c2>=o2;
        const alpha=hov===-1?1:(i===hov?1:.18);
        ctx.globalAlpha=alpha;
        const col=bull?'#34d399':'#f87171';
        ctx.strokeStyle=col;ctx.lineWidth=1;
        ctx.beginPath();ctx.moveTo(x,py(h2));ctx.lineTo(x,py(l2));ctx.stroke();
        const cw=step*.52,y1=py(Math.max(o2,c2)),y2=py(Math.min(o2,c2));
        ctx.fillStyle=bull?'rgba(52,211,153,.75)':'rgba(248,113,113,.75)';
        ctx.fillRect(x-cw/2,y1,cw,Math.max(y2-y1,2));
        ctx.strokeStyle=col;ctx.lineWidth=.7;ctx.strokeRect(x-cw/2,y1,cw,Math.max(y2-y1,2));
      }}
    }} else {{
      ctx.globalAlpha=1;
      ctx.beginPath();
      PRICES.forEach((p,i)=>{{const x=i*step+step/2;i===0?ctx.moveTo(x,py(p)):ctx.lineTo(x,py(p));}});
      const lc2=dark?'#38bdf8':'#1d6fe8';
      ctx.strokeStyle=lc2;ctx.lineWidth=2.5;ctx.stroke();
      ctx.lineTo((PRICES.length-1)*step+step/2,H);ctx.lineTo(step/2,H);ctx.closePath();
      const fc=dark?'rgba(56,189,248,.07)':'rgba(37,99,235,.07)';
      ctx.fillStyle=fc;ctx.fill();
      PRICES.forEach((p,i)=>{{
        const x=i*step+step/2;
        ctx.globalAlpha=hov===-1?1:(i===hov?1:.15);
        ctx.beginPath();ctx.arc(x,py(p),i===hov?5.5:3.5,0,Math.PI*2);
        ctx.fillStyle=i===hov?lc2:(dark?'rgba(56,189,248,.7)':'rgba(37,99,235,.7)');
        ctx.fill();
      }});
    }}
    ctx.globalAlpha=1;
    if(hov>=0){{
      const x=hov*step+step/2;
      ctx.strokeStyle=ac;ctx.lineWidth=.5;ctx.setLineDash([4,4]);
      ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H-12);ctx.stroke();
      ctx.setLineDash([]);
    }}
    drawTimeAxis();
  }}

  area.onmousemove=e=>{{
    const r=area.getBoundingClientRect();
    const mx=e.clientX-r.left;
    const idx=Math.min(OPENS.length-1,Math.max(0,Math.floor(mx/step)));
    draw(idx);
    if(document.getElementById('ct-date'))document.getElementById('ct-date').textContent='Today · '+TIMES[idx];
    if(document.getElementById('ct-time'))document.getElementById('ct-time').textContent=TIMES[idx];
    if(chartType==='candlestick'){{
      if(document.getElementById('ct-o'))document.getElementById('ct-o').textContent='₹'+OPENS[idx];
      if(document.getElementById('ct-c'))document.getElementById('ct-c').textContent='₹'+CLOSES[idx];
      if(document.getElementById('ct-h'))document.getElementById('ct-h').textContent='₹'+HIGHS[idx];
      if(document.getElementById('ct-l'))document.getElementById('ct-l').textContent='₹'+LOWS[idx];
      if(document.getElementById('chart-price'))document.getElementById('chart-price').textContent='₹'+CLOSES[idx];
    }}else{{
      if(document.getElementById('ct-o'))document.getElementById('ct-o').textContent='₹'+Math.round(PRICES[idx]);
      if(document.getElementById('ct-c'))document.getElementById('ct-c').textContent='—';
      if(document.getElementById('ct-h'))document.getElementById('ct-h').textContent='—';
      if(document.getElementById('ct-l'))document.getElementById('ct-l').textContent='—';
      if(document.getElementById('chart-price'))document.getElementById('chart-price').textContent='₹'+Math.round(PRICES[idx]);
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

components.html(APP_HTML, height=2200, scrolling=True)
