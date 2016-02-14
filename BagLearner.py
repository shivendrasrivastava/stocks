"""
A simple wrapper for Bagging.  (c) 2015 Darragh Hanley  
"""
import numpy as np
import LinRegLearner as lrl
import KNNLearner as knn

class BagLearner(object):

    def __init__(self, learner, kwargs, bags, boost):
        """
        @summary: Store parameters
        """
        self.learner = learner(**kwargs)   
        self.boost = boost
        self.bags = bags
        self.learners = []
        for i in range(0,self.bags):
            self.learners.append(learner(**kwargs))
             
    def addEvidence(self, dataX, dataY):
        """
        @summary: Add training data to learner
        """
        # Store the training data for use once we get test data  
        # self.learner.addEvidence(self.Xtrain, self.Ytrain)
        
        if self.boost:
            sample_index = np.random.choice(len(dataY), len(dataY), replace=True)
            for i in range(0,self.bags):
                dataXsamp = dataX[sample_index,:]            
                dataYsamp = dataY[sample_index]

                adaboost = self.learner
                adaboost.addEvidence(dataXsamp, dataYsamp)
                Yada = adaboost.query(dataX)
                error = abs(dataY - Yada)/sum(abs(dataY - Yada))
                sample_index = np.random.choice(len(dataY), len(dataY), replace=True, p = error) 
                self.learners[i].addEvidence(dataXsamp, dataYsamp)            
        else:   
            for i in range(0,self.bags):
                sample_index = np.random.choice(len(dataY), len(dataY), replace=True)
                dataXsamp = dataX[sample_index,:]            
                dataYsamp = dataY[sample_index]
                self.learners[i].addEvidence(dataXsamp, dataYsamp)
        
        
    def query(self, points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
         
        y_pred = np.empty([self.bags, points.shape[0]])
        for i in range(0,self.bags):
            y_pred[i] = self.learners[i].query(points)
        return np.mean(y_pred, axis=0)
            

if __name__=="__main__":
    print "the secret clue is not to be found here"