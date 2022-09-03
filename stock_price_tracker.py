import yfinance as yf
import streamlit as st

st.title('Simple stock price tracker app')

google = yf.Ticker("GOOG")

goog_df = google.history(period="1mo")

st.line_chart(goog_df.Close)

st.line_chart(goog_df.Volume)
