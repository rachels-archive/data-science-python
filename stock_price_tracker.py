import yfinance as yf
import streamlit as st

st.title('Simple stock price tracker app')

interval = st.selectbox(
     'Select interval',
     ('1d','1mo','6mo','1y','5y','10y')
)


option = st.sidebar.selectbox(
    "Pick a stock",
    ("Google", "Apple", "Microsoft")
)

goog = yf.Ticker("GOOG")
aapl = yf.Ticker("AAPL")
msft = yf.Ticker("MSFT")

goog_df = goog.history(period=interval)
aapl_df = aapl.history(period=interval)
msft_df = msft.history(period=interval)

def display(stock):
  st.write("""
  ## Closing Price
  """)
  st.line_chart(stock.Close)
  st.write("""
  ## Trading Volume
  """)
  st.line_chart(goog_df.Volume)
  
if option == "Google":
  display(goog_df)
elif option == "Apple":
  display(aapl_df)
else:
  display(msft_df)
  

