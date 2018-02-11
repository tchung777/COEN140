from numpy import *

class HW3():

    def __init__(self):
        self.data = genfromtxt("crime-train.txt", names=True, dtype=float)

        # Do all of this just to get our data into matrix form...
        test = []
        for x in self.data.dtype.names:
            test.append(self.data[x])

        test.append(ones(1595)) #include column of all ones
        self.trainData = array(test)
        self.trainData = transpose(self.trainData)

        self.t_data = genfromtxt("crime-test.txt", names=True, dtype=float)

        # Do all of this just to get our data into matrix form...
        test = []
        for x in self.t_data.dtype.names:
            test.append(self.t_data[x])

        test.append(ones(399))  # include column of all ones
        self.testData = array(test)
        self.testData = transpose(self.testData)

    def linearRegression(self):
        Y = self.trainData[:,0]
        X = self.trainData[:,1:]
        Y_i = self.testData[:,0]
        X_i = self.testData[:,1:]

        W = dot(dot(linalg.inv(dot(transpose(X),X)),transpose(X)),Y)
        predict = dot(X_i,transpose(W))
        RSME = sqrt( sum( (Y_i - predict)**2 ) / 399 )

        print(RSME)

    def rigidRegression(self):

    def crossValidation(self):
        for k in range(5):
            

if __name__ == "__main__":
    t = HW3()

    t.linearRegression()

