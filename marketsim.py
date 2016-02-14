"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import os

from util import get_data, plot_data
from analysis import get_portfolio_value, get_portfolio_stats, plot_normalized_data

pd.options.mode.chained_assignment = None

def compute_portvals(start_date, end_date, orders, startval):
    # get the trading days using SPY as reference
    dates = pd.date_range(start_date, end_date)
    df = get_data(['SPY'], dates)
    # Make the sell orders a negative value
    orders['Shares'][orders['Order'].str.upper()=='SELL'] = -orders['Shares'][orders['Order'].str.upper()=='SELL']

    # Create a data frame to hold a matrix of all the stocks
    symbols = np.unique(orders['Symbol'].values.ravel())  
    for stock in symbols: 
        df[stock]=0   
    
    # Get the prices for each day in the index
    # Front fill the prices where we have an NA, then backfill
    prices = get_data(symbols, df.index, False)
    prices = prices.fillna(method='ffill', axis=0)
    prices = prices.fillna(method='bfill', axis=0)

    # Add the starting value and a cash value
    df['Cash'] = startval + 0.0
    prices['Cash'] = 1
    orders['Prices'] = 0
    for ind, row in orders.iterrows():
        # calculate leverage        
        # leverage = (sum(longs) + sum(abs(shorts)) / ((sum(longs) - sum(abs(shorts)) + cash)
        # get temporary table after the transaction is made, and before the transaction is made
        df_chk, df_chk_b4 = df.ix[ind,1:], df.ix[ind,1:]
        df_chk [row['Symbol']] = df[row['Symbol']][ind] + row['Shares']
        df_chk ['Cash'] = df['Cash'][ind] - prices[row['Symbol']][ind] * row['Shares']
        df_chk        = prices.ix[ind] * df_chk
        df_chk_b4  = prices.ix[ind] * df_chk_b4
        # calculate the leverage after and before 
        lev_after = sum(abs(df_chk[:-1])) / sum(df_chk )
        lev_before = sum(abs(df_chk_b4[:-1])) / sum(df_chk_b4 )
        # print lev_after, lev_before, ind
        #if lev_after < 1000.0 or lev_after < lev_before :      
        df[row['Symbol']][ind:end_date] = df[row['Symbol']][ind:end_date] + row['Shares']
        df['Cash'][ind:end_date] = df['Cash'][ind:end_date] - prices[row['Symbol']][ind] * row['Shares']
        #else:
        #    print "Cancel the order", ind, row['Symbol'], row['Shares'], "Lev before", lev_before , "Lev after",  lev_after 
    
    df = df.iloc[:,1:] * prices
    portvals = df.sum(axis=1)
    # print portvals
    return portvals   


#def test_run():
    """Driver function."""
    # Define input parameters
    start_date = '2011-01-05'
    end_date = '2011-01-20'
    orders_file = os.path.join("orders", "orders-short.csv")
    start_val = 1000000

def test_run(start_date, end_date, orders, start_val,plot=True):
    # Process orders
    portvals = compute_portvals(start_date, end_date, orders, start_val)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # if a DataFrame is returned select the first column to get a Series
    
    # Get portfolio stats
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(portvals)

    # Simulate a SPY-only reference portfolio to get stats
    prices_SPX = get_data(['SPY'], pd.date_range(start_date, end_date))
    prices_SPX = prices_SPX[['SPY']]  # remove SPY
    portvals_SPX = get_portfolio_value(prices_SPX, [1.0])
    cum_ret_SPX, avg_daily_ret_SPX, std_daily_ret_SPX, sharpe_ratio_SPX = get_portfolio_stats(portvals_SPX)

    # Compare portfolio against $SPX
    print "Data Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of $SPX: {}".format(sharpe_ratio_SPX)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of $SPX: {}".format(cum_ret_SPX)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of $SPX: {}".format(std_daily_ret_SPX)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of $SPX: {}".format(avg_daily_ret_SPX)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])
    print "Final SPY Value: {}".format(portvals_SPX[-1]*start_val)
    
    # Plot computed daily portfolio value
    if plot == True:
        df_temp = pd.concat([portvals, prices_SPX['SPY']], keys=['Portfolio', 'SPY'], axis=1)
        plot_normalized_data(df_temp, title="Daily portfolio value", ylabel="Normalized Price")

if __name__ == "__main__":

    test_run( '2006-12-31', '2013-12-31', os.path.join('orders', "orders_pt3_bb.csv"), 10000)
    #test_run( '2006-12-31', '2013-12-31', os.path.join('orders', "orders_mst.csv"), 10000)
#    test_run( '2011-01-05', '2011-01-20', os.path.join("orders", "orders-short.csv"), 1000000)
#    test_run( '2011-01-10', '2011-12-20', os.path.join("orders", "orders.csv"), 1000000)
#    test_run( '2011-01-14', '2011-12-14', os.path.join("orders", "orders2.csv"), 1000000)
#    
#    
#    test_run( '2011-05-16', '2011-10-05', os.path.join("orders", "test_case_1.csv"), 1000000)
#    test_run( '2011-05-16', '2011-10-05', os.path.join("orders", "test_case_2.csv"), 1000000)
#    test_run( '2012-04-23', '2012-09-12', os.path.join("orders", "test_case_3.csv"), 1000000)

