from numpy import *
from random import *

class HW4():

    def __init__(self):
        self.data = genfromtxt("spam-train", delimiter=',')
        self.dataTest = genfromtxt("spam-test", delimiter=',')


    def normalizeData(self):
        (m, n) = self.data.shape
        self.data[:, 0:(n - 1)] = (self.data[:, 0:n - 1] - mean(self.data[:, 0:n - 1])) / std(self.data[:, 0:n - 1])
        self.trainingAns = copy(self.data[:, n - 1])
        self.data[:, n - 1] = ones(m)

        (m, n) = self.dataTest.shape
        self.dataTest[:, 0:(n - 1)] = (self.dataTest[:, 0:n - 1] - mean(self.dataTest[:, 0:n - 1])) / std(self.dataTest[:, 0:n - 1])
        self.testingAns = copy(self.dataTest[:, n - 1])
        self.dataTest[:, n - 1] = ones(m)

    def binarizeData(self):
        (m, n) = self.data.shape
        self.trainingAns = copy(self.data[:, n - 1])
        self.data = where(self.data > 0, 1, 0)
        self.data[:, n - 1] = ones(m)

        (m, n) = self.dataTest.shape
        self.testingAns = copy(self.dataTest[:, n - 1])
        self.dataTest = where(self.dataTest > 0, 1, 0)
        self.dataTest[:, n - 1] = ones(m)

    def transformLog(self):
        (m, n) = self.data.shape
        self.trainingAns = copy(self.data[:, n - 1])
        self.data = log(self.data + 0.1)
        self.data[:, n - 1] = ones(m)

        (m, n) = self.dataTest.shape
        self.testingAns = copy(self.dataTest[:, n - 1])
        self.dataTest = log(self.dataTest + 0.1)
        self.dataTest[:, n - 1] = ones(m)

    def sigmoid(self, z):
        g = 1 / (1 + exp(-z))
        return g

    def decision_boundary(self, prob):
        return 1 if prob > .5 else 0

    def classify(self, preds):
        """
        This function returns a vector of 1's and 0's
        :param preds: The output value from the sigmoid function
        :return: A vector of 1's and 0's for classification
        """
        decision_boundary = vectorize(self.decision_boundary)
        return decision_boundary(preds)

    def predict(self, features, weights):
        z = dot(features, weights)
        return self.sigmoid(z)

    def cost_function(self, features, labels, weights):
        '''
        Using Mean Absolute Error

        Features: Matrix of feature values
        Labels: Vector of true values
        Weights:Vector of Weights
        return: Mean Absolute cost of prediction vs true values.
        '''
        N = len(labels)

        predictions = self.predict(features, weights)

        costOne = -labels * log(predictions)

        costZero = (1 - labels) * log(1 - predictions)

        cost = costOne - costZero

        cost = cost.sum() / N

        return cost

    def logisticRegressionGradientDescent(self):
        Y = self.trainingAns
        X = self.data[:, :] #vector of features

        grad = None
        theta = zeros(58) #random.normal(0,1,96)
        alpha = 0.001
        eps = float('1e-6')
        N = len(X)

        previousLoss = self.cost_function(X, Y, theta)
        loss = None

        i = 0
        #print(previousLoss)

        while True:

            # Use training data to calculate the gradient
            h_x = self.predict(X, theta)
            grad = theta + ( alpha * (1/N) * dot(X.T, (Y - h_x)))

            # Use training data to calculate the square error loss
            loss = self.cost_function(X, Y, grad)

            #print(loss)


            if abs(loss - previousLoss) < eps: # If |L(w+1) - L(w)| < 0.0001
                break # Break out of the cycle.

            theta = grad # Update our weight vector
            previousLoss = loss # Set previousLoss to currentLoss so we don't have to recalculate this.
            i += 1


        classification = self.classify(self.predict(X, grad))
        classificationTest = self.classify((self.predict(self.dataTest, grad)))
        print( "Mean Training Error Rate: {}".format((classification  == Y).sum() / N) )
        print("Mean Testing Error Rate: {}".format((classificationTest == self.testingAns).sum() / N))

if __name__ == "__main__":
    test3 = HW4()
    test3.normalizeData()
    test3.logisticRegressionGradientDescent()

    test2 = HW4()
    test2.transformLog()
    test2.logisticRegressionGradientDescent()

    test = HW4()
    test.binarizeData()
    test.logisticRegressionGradientDescent()
