import yfinance as yf
import streamlit as st

st.title('Simple stock price tracker app')

option = st.sidebar.selectbox(
    "Pick a stock",
    ("Google", "Apple", "Microsoft")
)

goog = yf.Ticker("GOOG")
aapl = yf.Ticker("AAPL")
msft = yf.Ticker("MSFT")

goog_df = goog.history(period="1y")
aapl_df = aapl.history(period="1y")
msft_df = msft.history(period="1y")

def display(stock):
  st.write("""
  ## Closing Price
  """)
  st.line_chart(stock.Close)
  st.write("""
  ## Volume
  """)
  st.line_chart(goog_df.Volume)
  
if option == "Google":
  display(goog_df)
elif option == "Apple":
  display(aapl_df)
else:
  display(msft_df)
  

