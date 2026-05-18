import yfinance as yf
import py_vollibimport yfinance as yf
import py_vollib as pv
import pandas as pd

def fetch_options_data(ticker):
    stock = yf.Ticker(ticker)
    expiration_dates = stock.options
    options_data = []

    for exp_date in expiration_dates[:1]:  # Limiting to the first expiration date for simplicity
        opt_chain = stock.option_chain(exp_date)
        calls = opt_chain.calls
        puts = opt_chain.puts
        calls['type'] = 'call'
        puts['type'] = 'put'
        options_data.append(calls)
        options_data.append(puts)

    return pd.concat(options_data)

def calculate_implied_volatility(options_data, underlying_price):
    options_data['impliedVolatility'] = pv.implied_volatility(
        options_data['lastPrice'],
        underlying_price,
        options_data['strike'],
        0.01,  # Assuming risk-free rate of 1%
        (pd.to_datetime(options_data['expiration']) - pd.Timestamp.now()).days / 365,
        options_data['type'].str[0].str.lower()  # 'c' for call, 'p' for put
    )
    return options_data

def recommend_trades(options_data, threshold=0.5):
    high_iv_options = options_data[options_data['impliedVolatility'] > threshold]
    return high_iv_options[['contractSymbol', 'type', 'strike', 'lastPrice', 'impliedVolatility']]

def main():
    ticker = 'AAPL'
    underlying_stock = yf.Ticker(ticker)
    underlying_price = underlying_stock.history(period='1d')['Close'].iloc[0]
    
    options_data = fetch_options_data(ticker)
    options_data = calculate_implied_volatility(options_data, underlying_price)
    
    recommendations = recommend_trades(options_data, threshold=0.3)  # Adjust the threshold as needed
    print("Recommended Trades:")
    print(recommendations)

if __name__ == "__main__":
    main() as pv
import pandas as pd

def fetch_options_data(ticker):
    stock = yf.Ticker(ticker)
    expiration_dates = stock.options
    options_data = []

    for exp_date in expiration_dates[:1]:  # Limiting to the first expiration date for simplicity
        opt_chain = stock.option_chain(exp_date)
        calls = opt_chain.calls
        puts = opt_chain.puts
        calls['type'] = 'call'
        puts['type'] = 'put'
        options_data.append(calls)
        options_data.append(puts)

    return pd.concat(options_data)

def calculate_implied_volatility(options_data, underlying_price):
    options_data['impliedVolatility'] = pv.implied_volatility(
        options_data['lastPrice'],
        underlying_price,
        options_data['strike'],
        0.01,  # Assuming risk-free rate of 1%
        (pd.to_datetime(options_data['expiration']) - pd.Timestamp.now()).days / 365,
        options_data['type'].str[0].str.lower()  # 'c' for call, 'p' for put
    )
    return options_data

def recommend_trades(options_data, threshold=0.5):
    high_iv_options = options_data[options_data['impliedVolatility'] > threshold]
    return high_iv_options[['contractSymbol', 'type', 'strike', 'lastPrice', 'impliedVolatility']]

def main():
    ticker = 'AAPL'
    underlying_stock = yf.Ticker(ticker)
    underlying_price = underlying_stock.history(period='1d')['Close'].iloc[0]
    
    options_data = fetch_options_data(ticker)
    options_data = calculate_implied_volatility(options_data, underlying_price)
    
    recommendations = recommend_trades(options_data, threshold=0.3)  # Adjust the threshold as needed
    print("Recommended Trades:")
    print(recommendations)

if __name__ == "__main__":
    main()