import yfinance as yf
import streamlit as st

st.title('Simple stock price tracker app')

google = yf.Ticker("GOOG")

goog_df = google.history(period="1y")

st.write("""
## Closing Price
""")
st.line_chart(goog_df.Close)

st.write("""
## Volume Price
""")
st.line_chart(goog_df.Volume)
