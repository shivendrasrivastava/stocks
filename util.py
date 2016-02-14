"""MLT: Utility code."""

import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import requests
def symbol_to_path(symbol, base_dir="data"):#os.path.join("..", "data")):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

def get_data(symbols, dates, addSPY=True):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if addSPY and 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols = ['SPY'] + symbols
    for symbol in symbols:
        download_data(symbol)
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])
    return df
def download_data(ticker,
                      start_date=(2000,1,1),
                      end_date=datetime.date.today().timetuple()[0:3]):
    """
    Obtains data from Yahoo Finance 

    ticker: Yahoo Finance ticker symbol, e.g. "GOOG" for Google, Inc.
    start_date: Start date in (YYYY, M, D) format
    end_date: End date in (YYYY, M, D) format
    """

    # Construct the Yahoo URL with the correct integer query parameters
    # for start and end dates. Note that some parameters are zero-based!
    #yahoo_url = "http://ichart.finance.yahoo.com/table.csv?s=%s&a=%s&b=%s&c=%s&d=%s&e=%s&f=%s" % \
    #    (ticker, start_date[1] - 1, start_date[2], start_date[0], end_date[1] - 1, end_date[2], end_date[0])
    yahoo_url = "http://chart.finance.yahoo.com/table.csv?s=%s&a=0&b=1&c=2000&d=%s&e=%s&f=%s&g=d&ignore=.csv" % \
        (ticker,end_date[1],end_date[2],end_date[0])
    filename = symbol_to_path(ticker)
    # Try connecting to Yahoo Finance and obtaining the data
    # On failure, print an error message
    try:
        if not os.path.isfile(filename):
            r = requests.get(yahoo_url)
            fo = open(filename,"wb")
            fo.write(r.text.encode(('utf-8')))
            fo.close()
    except Exception, e:
        print "Could not download data: %s" % e

    return True


def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12, figsize = [10,8], grid=True)
    lines, labels = ax.get_legend_handles_labels()
    labels[1] = 'SPY'    
    ax.legend(lines, labels, loc='best')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()
