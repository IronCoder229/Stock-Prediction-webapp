import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
from sklearn import preprocessing

import streamlit as st
import plotly.express as px
import yfinance as yf



st.title('Stock Trend Prediction')


ticker = st.sidebar.text_input('Ticker','AAPL')
start_date = st.sidebar.text_input('Start Date','2020-01-01')
end_date = st.sidebar.text_input('End Date','2024-01-01')

try:
    df = yf.download(ticker, start=start_date, end=end_date)
    if not df.empty:
        fig = px.line(df, x=df.index, y=df['Close'], title=ticker)
        st.plotly_chart(fig)
    else:
        st.error("No data available for the specified ticker and date range.")
except Exception as e:
    st.error(f"An error occurred: {e}")


#Describing Data
st.subheader('Data for selected time ')
st.write(df.describe())


# Calculate the 100-day and 200-day moving averages
ma100 = df['Close'].rolling(window=100).mean()
ma200 = df['Close'].rolling(window=200).mean()

# Create the figure using Plotly Express
fig = px.line()
fig.add_scatter(x=df.index, y=ma100, mode='lines', name='100MA', line=dict(color='red'))
fig.add_scatter(x=df.index, y=ma200, mode='lines', name='200MA', line=dict(color='blue'))
fig.add_scatter(x=df.index, y=df['Close'], mode='lines', name='Closing Price', line=dict(color='green'))

# Update layout
fig.update_layout(title='Closing Price Vs Time Chart with 100MA & 200MA',
                  title_font=dict(size=25),
                  xaxis_title='Time',
                  yaxis_title='Price',
                  width=1000,  # Adjust width to make it slightly smaller
                  height=500)  # Adjust height to make it slightly smaller

# Display the figure
st.plotly_chart(fig)


#Splitting data into Training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70) : int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)



#load my model
model = load_model('keras_model.h5')


#Testing Part

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0])
x_test, y_test = np.array(x_test), np.array(y_test)


y_predicted = model.predict(x_test)
scaler = scaler.scale_


scale_factor = 1/scaler[0]
y_predicted = y_predicted*scale_factor
y_test = y_test * scale_factor

# Final Graphs

# st.subheader('Predictions Vs Original')
# fig2=plt.figure(figsize = (12,6))
# plt.plot(y_test, 'b', label = 'Orignal Price')
# plt.plot(y_predicted, 'r', label = 'Predicted Price')
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.legend()
# st.pyplot(fig2)

# Assuming y_test and y_predicted are numpy arrays
data = pd.DataFrame({'Original Price': y_test.ravel(), 'Predicted Price': y_predicted.ravel()})

# Create the plot using Plotly Express
fig = px.line(data, labels={'value': 'Price', 'index': 'Time'},
              title='Predictions Vs Original')

# Add legend
fig.update_traces(mode='lines', name='Original Price')
fig.add_scatter(y=y_predicted.ravel(), mode='lines', name='Predicted Price')

# Show the plot using Streamlit
st.plotly_chart(fig)


# Stock Dashboard

pricing_data, fundamental_data, news =st.tabs(["Pricing Data", "Fundamental Data", "Top 10 News"])

with pricing_data:
    st.header('Price Movements')
    df2 = df
    df2['% Change'] = df['Close']/ df['Close'].shift(1) -1
    df2.dropna(inplace =True)
    st.write(df)
    annual_return = df2['% Change'].mean()*252*100
    st.write('Annual Return is ',annual_return,'%')
    stdev = np.std(df2['% Change'])*np.sqrt(252)
    st.write('Standard Deviation is ',stdev*100,'%')
    st.write('Risk Adj. Return is ',annual_return/(stdev*100))

from alpha_vantage.fundamentaldata import FundamentalData
with fundamental_data:
    key ='OW1639L63B5UCYYL'
    fd =FundamentalData(key,output_format = 'pandas')
    st.subheader('Balance Sheet')
    balance_sheet = fd.get_balance_sheet_annual(ticker)[0]
    bs = balance_sheet.T[2:]
    bs.columns= list(balance_sheet.T.iloc[0])
    st.write(bs)
    st.subheader('Income Statement')
    income_statement = fd.get_income_statement_annual(ticker)[0]
    is1 = income_statement.T[2:]
    is1.columns= list(income_statement.T.iloc[0])
    st.write(is1)
    st.subheader('Cash Flow Statement')
    cash_flow = fd.get_cash_flow_annual (ticker) [0]
    cf = cash_flow.T[2:]
    cf.columns = list(cash_flow.T.iloc[0])
    st.write(cf)


from stocknews import StockNews
with news:
    st.header(F'News of {ticker}')
    sn = StockNews(ticker, save_news=False)
    df_news = sn.read_rss()
    for i in range(10):
        st.subheader(f'News {i+1}')
        st.write(df_news['published'][i])
        st.write(df_news['title'][i])
        st.write(df_news['summary'][i])
        title_sentiment = df_news['sentiment_title'][i]
        st.write(f'Title Sentiment {title_sentiment}')
        news_sentiment = df_news['sentiment_summary'][i]
        st.write(f'News Sentiment {news_sentiment}')
