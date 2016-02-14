"""
Predict stock movement 
"""

import os
import numpy as np
import pandas as pd
import math
import LinRegLearner as lrl
import KNNLearner as knn
import BagLearner as bl
import matplotlib.pyplot as plt
import datetime
import marketsim as msim
import util

pd.options.mode.chained_assignment = None # http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

def rmse(act, pred):
    rmse = math.sqrt(((act - pred ) ** 2).sum()/act.shape[0])
    return rmse

def get_vals(start_date, end_date, symbol):
    # Read in adjusted closing prices for given symbols, date range
    # to allow getting SMA from day 1 on start_date we initially read in an earlier date to calc SMA
    dates = pd.date_range(start_date, end_date)
    prices_all = util.get_data(symbol, dates)  # automatically adds SPY
    prices = prices_all[symbol]  # only portfolio symbols       
    return prices

def indicators(prices, window=20):
    
    # Calculate out the Moving Average and the SD of the prices
    sma = pd.stats.moments.rolling_mean(prices, window) 
    sd = pd.stats.moments.rolling_std(prices, window)
    # Ratio to the bolligner bands (above upper band is > 1; below lower band <-1)    
    bb_diff =  (prices - sma)/(2*sd)
        
    # Ratio to a recent high in a specific window   
    winmax = pd.stats.moments.rolling_max(prices, window)
    high_diff = (prices-winmax)/(2*sd)
    
    # Ratio to a recent low in a specific window         
    winlow = pd.stats.moments.rolling_min(prices, window)
    low_diff = (prices-winlow)/(2*sd)
    
    #Moving Average Convergence Divergence (MACD) using exponentially weighted Moving average
    macd=pd.ewma(prices,span=window) - pd.ewma(prices,window/2) 
    
    # combine all to a dataframe 
    df = pd.DataFrame(pd.concat([bb_diff, high_diff, low_diff, macd], axis=1))
    df.columns = ['bb_diff', 'high_diff', 'low_diff',  'macd']
    df.columns =  [prices.columns.values[0] + '_' + x for x in df.columns]

    # Normalaise the data to between -1 and 1    
    df = df.apply(lambda x: ((x - np.min(x)))*2 / (np.max(x) - np.min(x))-1)    
    return df[[0,1,2,3]]

def predict_outsamp(X_trn, y_trn, X_tst, y_tst, symbol, start_date, end_date, k):
    # create a linear regression learner and train it
    lrlearner = lrl.LinRegLearner() # create a LinRegLearner
    lrlearner.addEvidence(X_trn, y_trn) # train it
    ytst_lr = lrlearner.query(X_tst)

    # create a KNN learner (k=10) and train it    
    knnlearn = knn.KNNLearner(k) # constructor
    knnlearn.addEvidence(X_trn, y_trn) # training step
    ytst_knn = knnlearn.query(X_tst)    

    # create a Bag learner and train it    
    baglearn = bl.BagLearner(learner = knn.KNNLearner, kwargs = {"k":k}, bags = 100, boost = False) # constructor
    baglearn.addEvidence(X_trn, y_trn) # training step
    ytst_bag = baglearn.query(X_tst)

    # Combine all models
    combined = (ytst_lr+ytst_knn+ytst_bag)/3
    
    print ""
    print "Out of sample predictions for %s data from %s to %s"  %(symbol[0], start_date, end_date)
    print "KNN RMSE %0.4f; LinReg RMSE %0.4f; BagReg RMSE %0.4f; Combined RMSE %0.4f" %(rmse(y_tst, ytst_knn), rmse(y_tst, ytst_lr), rmse(y_tst, ytst_bag), rmse(y_tst, combined))
    print "KNN corr %0.4f; LinReg corr %0.4f; BagReg corr %0.4f" %(np.corrcoef(y_tst, ytst_knn)[0,1], np.corrcoef(y_tst, ytst_lr)[0,1], np.corrcoef(y_tst, ytst_bag)[0,1])
    print "KNN mean %0.4f; LinReg mean %0.4f; BagReg mean %0.4f" %(abs(y_tst - ytst_knn).mean(), abs(y_tst - ytst_lr).mean(), abs(y_tst - ytst_bag).mean())
    print "Actual mean 5 day change %0.4f" %abs(y_tst).mean()
    print ""
    return ytst_lr, ytst_knn, ytst_bag

def gen_orders(predsdf, threshold, symbol, model=["PredLr", "PredKnn"][1]): 
    # Read the list of predicted 5 day rise or fall in price
    orders = predsdf
    # Initialise and order if the prediction is above a threshold
    orders['Order'] = "NONE"
    orders['Order'][orders[model]>threshold] = 'BUY'
    orders['Order'][orders[model]<-(threshold/2)] = 'SELL'
    #orders['Order'][orders[model]<-0] = 'SELL'
    # Remove and rows where there is no buy or sell, or duplicate order
    orders = orders[orders['Order'] != 'NONE']
    orders = orders[orders.Order <> orders.Order.shift(1)]
    orders['Symbol'] = symbol
    # Remove all rows except the order and symbol
    orders = orders[["Symbol", "Order"]]
    # Buy 100 for the first order, and 200 for all the following
    orders['Shares'] = 100
    orders['Shares'][1:] = 200
    return orders   


def run_code(symbol, start_date_in, end_date_in, start_date_out, end_date_out, end_date_plot, window, threshold, k=5, insamp=True, mod = ['PredLr']):
    """Driver function."""
    prices = get_vals('2000-01-01', '2016-02-10', symbol) 

    # Get the stock indicators and the target (5 day change in price)
    # convert to a numpy array to match the required learner input
    X_trn = indicators(prices, window)[start_date_in : end_date_in].as_matrix()
    y_trn = prices.pct_change(5).shift(-5)[start_date_in : end_date_in].as_matrix()[:,0]
    X_tst = indicators(prices, window)[start_date_out : end_date_out].as_matrix()
    y_tst = prices.pct_change(5).shift(-5)[start_date_out : end_date_out].as_matrix()[:,0]
   
    if insamp==True:
        ypred_lr, ypred_knn, ypred_bag = predict_insamp(X_trn, y_trn, symbol, start_date_in, end_date_in, k)
    else:
        ypred_lr, ypred_knn, ypred_bag = predict_outsamp(X_trn, y_trn, X_tst, y_tst, symbol, start_date_out, end_date_out, k)

    # Create a data frame with the actual prices and predictions
    preds = pd.DataFrame(index=prices[start_date_out : end_date_out].index)
    preds['Actual Price'] = prices[start_date_out : end_date_out]
    preds['Actual Day+5 Price'] = prices.shift(-5)[start_date_out : end_date_out]
    preds['Pred Day+5 Price LinReg'] = preds['Actual Price'] * (1 + ypred_lr)
    #preds['Pred Day+5 Price Knn (k=5)'] = preds['Actual Price'] * (1 + ypred_knn)
    #preds['Pred Day+5 Price Bagged Knn'] = preds['Actual Price'] * (1 + ypred_bag)
    #preds['Pred Day+5 Price combined'] = (preds['Pred Day+5 Price LinReg']+preds['Pred Day+5 Price Bagged Knn'])/2

    # Store the pift values to study them
    predslift = pd.DataFrame(index=prices[start_date_out : end_date_out].index)
    predslift['Actual'] = y_trn if insamp == True else y_tst
    predslift['PredLr'] = ypred_lr
    predslift['PredKnn'] = ypred_knn  
    predslift['PredBag'] = ypred_bag   
    
    # Generate the orders data using the function above and run through a market simulator
    orders = gen_orders(predslift, threshold, symbol[0], model=mod[0])
    msim.test_run(start_date_in if insamp == True else start_date_out, end_date_in if insamp == True else end_date_out, orders, 10000,True)
    
    # Plot the charts 
    # Predictied 5 day price change
    ax = preds.ix[start_date_out:end_date_plot].plot(figsize=(12,6), lw=1.5,
                    title = "5 Day Price Change Predictions " + symbol[0])
    #plt.show()
    #print preds['Pred Day+5 Price LinReg'].pct_change()


if __name__=="__main__":
	today = datetime.date.today().timetuple()[0:3]
	today = str(today[0])+"-"+str(today[1])+"-"+str(today[2])
	#symbol, start_date_in, end_date_in, start_date_out, end_date_out, end_date_plot, window, threshold, k=5, insamp=True, mod = ['PredLr'])
	#a, b, c, lift,e = mc3_p2_run(['$VIX'], '2008-01-01', '2009-12-31', '2010-01-01', '2010-12-31', '2010-06-01', 20, .01, k=5, insamp=False)  
	run_code(['SPY2'], '2008-01-01', '2013-06-30', '2013-07-01', '2016-02-01', '2016-02-01', 20, .005, k=5, insamp=False)  
	#run_code(['IBM'], '2008-01-01', '2013-06-30', '2013-06-01', '2016-01-30', '2016-02-01', 20, .03, k=5, insamp=False)  




