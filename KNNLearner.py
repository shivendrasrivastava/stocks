"""
A simple wrapper for knn.  (c) 2015 Darragh Hanley  
"""

import numpy as np
from scipy.spatial import distance

class KNNLearner(object):

    def __init__(self, k = 3):
        """
        @summary: Store parameters
        """
        if k < 1:
            raise ValueError('Choose a value of k greater than 0.')
        self.k = k

    def addEvidence(self, dataX, dataY):
        """
        @summary: Add training data to learner
        """
        # Store the training data for use once we get test data
        self.Xtrain, self.Ytrain = dataX, dataY   
        
    def query(self, points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        # Store the test data 
        self.Xtest = points
        
        # Scale the train and test data together so train and test is set to same scale
        # Loop through the train columns and scale them, then split back to train and test
        Xfull = np.concatenate((self.Xtrain, self.Xtest), axis=0)   
        rowtrn, coltrn = self.Xtrain.shape
        rowtst, coltst = self.Xtest.shape
        
        for col in range(coltrn):
            Xfull[:,col] = (Xfull[:,col] - abs(Xfull[:,col]).min()) / (abs(Xfull[:,col]).max())
        Xtrn = Xfull[:rowtrn]
        Xtst = Xfull[rowtrn:]
        
        # Get a matrix of euclidean distance between train and test 
        # Get the indices of the k closest train values for each test value    
        
#        dist_matrix = np.zeros((rowtrn, rowtst), dtype=np.double)
#        for i in xrange(0, rowtrn):
#            for j in xrange(0, rowtst):
#                dist_matrix[i, j] = distance.euclidean(Xtrn[i, :], Xtst[j, :])
        dist_matrix = distance.cdist(Xtrn, Xtst, 'euclidean')
        neighbour_index = np.argsort(dist_matrix, axis=0)[:self.k,:]
        neighbour_index = neighbour_index.transpose()     
        
        # Loop through each point and get the mean response for k nearest neighbours using the indexes.
        y_pred = np.empty([rowtst])
        for i in range(rowtst):
            y_pred[i] = self.Ytrain[neighbour_index[i]].mean()
        return y_pred    

if __name__=="__main__":
    print "the secret clue is not to be found here"