import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# =========================
# 🎨 PAGE CONFIG
# =========================
st.set_page_config(page_title="Stock AI Pro", layout="wide")

# =========================
# 🎨 CUSTOM CSS
# =========================
st.markdown("""
<style>
body {
    background-color: #0b0f19;
}
.main {
    background: linear-gradient(180deg, #0b0f19 0%, #05070d 100%);
    color: white;
}
h1, h2, h3 {
    color: #00f0ff;
}
.stButton>button {
    background: linear-gradient(90deg, #ff004c, #00f0ff);
    color: white;
    border-radius: 10px;
    height: 50px;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

st.title("⚡ AI Stock Market Expert System")

# =========================
# 📊 LOAD DATASET
# =========================
data = pd.read_csv("stock_data.csv")
data["Trend"] = data["Close"] - data["Open"]

X = data[['Open','High','Low','Close','Volume','Trend']]
y = data['Signal']

# =========================
# 🤖 MODEL
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))

# =========================
# 📊 METRICS
# =========================
col1, col2 = st.columns(2)

with col1:
    st.metric("📊 Model Accuracy", f"{round(acc*100,2)}%")

with col2:
    st.metric("📁 Dataset Size", len(data))

# =========================
# 🧾 INPUT
# =========================
st.subheader("Enter Stock Data")

c1, c2 = st.columns(2)

with c1:
    open_p = st.number_input("Open Price", value=120)
    high = st.number_input("High Price", value=125)
    low = st.number_input("Low Price", value=115)

with c2:
    close = st.number_input("Close Price", value=123)
    volume = st.number_input("Volume", value=15000)

trend = close - open_p

# =========================
# 📊 GRAPH TYPE SELECTOR
# =========================
graph_type = st.radio(
    "Select Graph Type:",
    ["Candlestick Chart", "Line Trend Graph"]
)

# =========================
# 🔮 PREDICTION
# =========================
if st.button("🚀 Predict Decision"):

    input_data = [[open_p, high, low, close, volume, trend]]
    prediction = model.predict(input_data)[0]

    if prediction == "Buy":
        st.success("📈 BUY SIGNAL")
    elif prediction == "Sell":
        st.error("📉 SELL SIGNAL")
    else:
        st.warning("⚖️ HOLD SIGNAL")

    # =========================
    # 📊 GRAPH SWITCH LOGIC
    # =========================

    if graph_type == "Candlestick Chart":

        st.subheader("📊 Candlestick Chart")

        num_points = 30
        opens = np.random.randint(open_p - 5, open_p + 5, num_points)
        closes = opens + np.random.randint(-5, 5, num_points)
        highs = np.maximum(opens, closes) + np.random.randint(1, 5, num_points)
        lows = np.minimum(opens, closes) - np.random.randint(1, 5, num_points)

        fig = go.Figure(data=[go.Candlestick(
            open=opens,
            high=highs,
            low=lows,
            close=closes
        )])

        fig.update_layout(
            template="plotly_dark",
            title="Simulated Market Movement",
            xaxis_title="Time",
            yaxis_title="Price",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    else:

        st.subheader("📈 Line Trend Graph")

        prices = np.linspace(open_p, close, 20) + np.random.normal(0, 1, 20)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            y=prices,
            mode='lines+markers',
            line=dict(color='#00f0ff', width=3),
            marker=dict(size=6),
            name="Price Trend"
        ))

        fig.update_layout(
            template="plotly_dark",
            title="Stock Movement Simulation",
            xaxis_title="Time",
            yaxis_title="Price",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)