
import streamlit as st
import requests

st.set_page_config(page_title="Stock Trend Dashboard", layout="centered")

st.title("📈 Stock Trend Prediction Dashboard")
st.write("This dashboard sends features to the FastAPI model and shows the predicted trend.")

# Sidebar for API config
st.sidebar.header("API Configuration")
api_url = st.sidebar.text_input("FastAPI URL", "http://127.0.0.1:8000/predict")

st.subheader("Input Features")

close = st.number_input("Close price", value=210.5)
daily_return = st.number_input("Daily return", value=0.003, format="%.5f")
sma_20 = st.number_input("SMA 20", value=208.1)
sma_50 = st.number_input("SMA 50", value=200.4)
ema_12 = st.number_input("EMA 12", value=207.9)
ema_26 = st.number_input("EMA 26", value=203.2)
rsi_14 = st.number_input("RSI 14", value=55.3)
macd = st.number_input("MACD", value=0.42)
bb_width = st.number_input("Bollinger Band Width", value=0.05, format="%.4f")
atr_14 = st.number_input("ATR 14", value=2.1)
volume_ratio = st.number_input("Volume ratio", value=1.2)
momentum_10 = st.number_input("Momentum 10", value=1.5)

if st.button("Predict Trend"):
    payload = {
        "close": close,
        "daily_return": daily_return,
        "sma_20": sma_20,
        "sma_50": sma_50,
        "ema_12": ema_12,
        "ema_26": ema_26,
        "rsi_14": rsi_14,
        "macd": macd,
        "bb_width": bb_width,
        "atr_14": atr_14,
        "volume_ratio": volume_ratio,
        "momentum_10": momentum_10,
    }

    try:
        res = requests.post(api_url, json=payload, timeout=10)
        if res.status_code == 200:
            data = res.json()
            st.success(f"Predicted trend: **{data['trend_label']}** (class {data['prediction_class']})")
            st.write("Probabilities:")
            st.json(data["probabilities"])
        else:
            st.error(f"API returned status code {res.status_code}")
            st.text(res.text)
    except Exception as e:
        st.error(f"Error calling API: {e}")
