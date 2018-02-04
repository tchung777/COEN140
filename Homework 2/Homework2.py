from numpy import *
from random import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

class HW2():

    def __init__(self):
        self.data = genfromtxt("iris.data", delimiter=',', dtype= None, names='sepalLength, sepalWidth, petalLength, petalWidth, class')
        self.irisSetosa = self.data[0:50]
        self.irisVersicolor = self.data[50:100]
        self.irisVirginica = self.data[100:150]

        # Have ndarrays to partition training sets and testing sets.
        self.trainDataSetosa = None
        self.testDataSetosa = None

        self.trainDataVersicolor = None
        self.testDataVeriscolor = None

        self.trainDataVirginica = None
        self.testDataVirginica = None

        # Boolean to determine what feature we're including to train and test.
        self.includeSepLength = True
        self.includeSepWidth = True
        self.includePetLength = True
        self.includePetWidth = True

        # Variable to keep track number of features.
        self.numOfFeatures = 4

        # Boolean to assume that features are independent.
        self.indepFeatures = True

    def extractTrainingSet(self):
        """
        This function extracts the data we get from 'iris.data' and partition the data for the three clases into
        sets of training data and testing data.

        Depending on the boolean settings that determines what features we want to include, arrays will be created
        containing columns for those features. Again, we will have one array that contains the overall training data
        and testing data.
        :return: None.
        """
        #shuffle(self.irisSetosa)
        #shuffle(self.irisVirginica)
        #shuffle(self.irisVersicolor)

        self.trainDataSetosa = self.irisSetosa[:40]
        self.testDataSetosa = self.irisSetosa[40:]

        self.trainDataVersicolor = self.irisVersicolor[:40]
        self.testDataVeriscolor = self.irisVersicolor[40:]

        self.trainDataVirginica = self.irisVirginica[:40]
        self.testDataVirginica = self.irisVirginica[40:]

        test = []

        if self.includeSepLength:
            test.append( (self.data['sepalLength']) )

        if self.includeSepWidth:
            test.append( (self.data['sepalWidth']) )

        if self.includePetLength:
            test.append( (self.data['petalLength']) )

        if self.includePetWidth:
            test.append( (self.data['petalWidth']) )
        self.testData = array(test)

        self.testData = transpose(self.testData)

        self.trainOne = []
        self.trainTwo = []
        self.trainThree = []

        if self.includeSepLength:
            self.trainOne.append(self.trainDataSetosa['sepalLength'])
            self.trainTwo.append(self.trainDataVersicolor['sepalLength'])
            self.trainThree.append(self.trainDataVirginica['sepalLength'])

        if self.includeSepWidth:
            self.trainOne.append(self.trainDataSetosa['sepalWidth'])
            self.trainTwo.append(self.trainDataVersicolor['sepalWidth'])
            self.trainThree.append(self.trainDataVirginica['sepalWidth'])

        if self.includePetLength:
            self.trainOne.append(self.trainDataSetosa['petalLength'])
            self.trainTwo.append(self.trainDataVersicolor['petalLength'])
            self.trainThree.append(self.trainDataVirginica['petalLength'])

        if self.includePetWidth:
            self.trainOne.append(self.trainDataSetosa['petalWidth'])
            self.trainTwo.append(self.trainDataVersicolor['petalWidth'])
            self.trainThree.append(self.trainDataVirginica['petalWidth'])

        self.trainOne = transpose(array(self.trainOne))
        self.trainTwo = transpose(array(self.trainTwo))
        self.trainThree = transpose(array(self.trainThree))


    def calculateMuMean(self):
        """
        This function calculates the mean average of the training data.
        :return: None.
        """
        testOne = []
        testTwo = []
        testThree = []

        if self.includeSepLength:
            testOne.append(( mean(self.trainDataSetosa['sepalLength']) ))
            testTwo.append((mean(self.trainDataVersicolor['sepalLength'])))
            testThree.append((mean(self.trainDataVirginica['sepalLength'])))

        if self.includeSepWidth:
            testOne.append(( mean(self.trainDataSetosa['sepalWidth'])) )
            testTwo.append((mean(self.trainDataVersicolor['sepalWidth'])))
            testThree.append((mean(self.trainDataVirginica['sepalWidth'])))

        if self.includePetLength:
            testOne.append(( mean(self.trainDataSetosa['petalLength'])) )
            testTwo.append((mean(self.trainDataVersicolor['petalLength'])))
            testThree.append((mean(self.trainDataVirginica['petalLength'])))

        if self.includePetWidth:
            testOne.append(( mean(self.trainDataSetosa['petalWidth'])) )
            testTwo.append((mean(self.trainDataVersicolor['petalWidth'])))
            testThree.append((mean(self.trainDataVirginica['petalWidth'])))

        self.muOne = array(testOne)

        self.muTwo = array(testTwo)

        self.muThree = array(testThree)


    def calculateSigma(self):
        """
        This function calculates the covariance matrix using the training data and the MU mean of the training data.
        If the indepFeatures boolean is on, the covariance matrix will be a matrix of diagonal.

        The matrix will be a numOfFeatures x numFeatures matrix.
        :return: None.
        """
        self.calculateMuMean()

        sumOne = 0
        sumTwo = 0
        sumThree = 0

        (m, n) = self.trainOne.shape
        for x in range(m):
            diff = subtract(self.trainOne[x], self.muOne) # This will give us a 4x1
            diff = diff.reshape((self.numOfFeatures, 1)) # This will reshape this
            diffTransposed = transpose(diff)
            if self.indepFeatures:
                a = diag(diag(dot(diff, diffTransposed)))
                sumOne += a
            else:
                sumOne += dot(diff, diffTransposed)


            diffTwo = subtract(self.trainTwo[x], self.muTwo)  # This will give us a 4x1
            diffTwo = diffTwo.reshape((self.numOfFeatures, 1))
            diffTwoTransposed = transpose(diffTwo)
            if self.indepFeatures:
                a = diag(diag(dot(diffTwo, diffTwoTransposed)))
                sumTwo += a
            else:
                sumTwo += dot(diffTwo, diffTwoTransposed)

            diffThree = subtract(self.trainThree[x], self.muThree)  # This will give us a 4x1
            diffThree = diffThree.reshape((self.numOfFeatures, 1))
            diffThreeTransposed = transpose(diffThree)
            if self.indepFeatures:
                a = diag(diag(dot(diffThree, diffThreeTransposed)))
                sumThree += a
            else:
                sumThree += dot(diffThree, diffThreeTransposed)

        self.sigma = sumOne / m  # programming language paradigms don't seem to see 4x1 as a vector but as a row.
        self.sigmaTwo = sumTwo / m  # programming language paradigms don't seem to see 4x1 as a vector but as a row.
        self.sigmaThree = sumThree / m  # programming language paradigms don't seem to see 4x1 as a vector but as a row.


    def calculateProbabilityDensityFunc(self):
        """
        This function uses the Gaussian Multivariate Probability Density Function to calculate the probability density
        of a test data point as Setosa, Versicolor or Virginica.
        :return: None.
        """

        trainErrorSetosa = 0
        trainErrorVirginica = 0
        trainErrorVersicolor = 0

        testErrorSetosa = 0
        testErrorVirginica = 0
        testErrorVersicolor = 0

        i = 1/(((2*pi)**(self.numOfFeatures/2))*sqrt(linalg.det(self.sigma)))
        z = linalg.inv(self.sigma)

        iTwo = 1/(((2*pi)**(self.numOfFeatures/2))*sqrt(linalg.det(self.sigmaTwo)))
        zTwo = linalg.inv(self.sigmaTwo)

        iThree = 1/(((2*pi)**(self.numOfFeatures/2))*sqrt(linalg.det(self.sigmaThree)))
        zThree = linalg.inv(self.sigmaThree)

        (m, n) = self.testData.shape
        for x in range(m):
            diff = subtract(self.testData[x], self.muOne)  # This will give us a 4x1
            diff = diff.reshape((self.numOfFeatures, 1))
            diffTransposed = transpose(diff)

            k = exp(-0.5 * dot(dot(diffTransposed, z), diff))
            probSetosa = i * k

            diffTwo = subtract(self.testData[x], self.muTwo)  # This will give us a 4x1
            diffTwo = diffTwo.reshape((self.numOfFeatures, 1))
            diffTwoTransposed = transpose(diffTwo)

            k = exp(-0.5 * dot(dot(diffTwoTransposed, zTwo), diffTwo))
            probVersicolor = iTwo * k

            diffThree = subtract(self.testData[x], self.muThree)  # This will give us a 4x1
            diffThree = diffThree.reshape((self.numOfFeatures, 1))
            diffThreeTransposed = transpose(diffThree)

            k = exp(-0.5 * dot(dot(diffThreeTransposed, zThree), diffThree))
            probVirginica = iThree * k

            maxVal = max([probSetosa[0], probVersicolor[0], probVirginica[0]])

            answer = ""
            if maxVal == probSetosa:
                answer = "Iris-setosa"
            elif maxVal == probVersicolor:
                answer = "Iris-versicolor"
            elif maxVal == probVirginica:
                answer = "Iris-virginica"
            trueVal = self.data["class"][x].decode("utf-8")
            if answer != trueVal:
                if trueVal == "Iris-setosa":
                    if x >= 40 and x < 50:
                        testErrorSetosa += 1
                    else:
                        trainErrorSetosa += 1
                elif trueVal == "Iris-versicolor":
                    if x >= 90 and x < 100:
                        testErrorVersicolor += 1
                    else:
                        trainErrorVersicolor += 1
                elif trueVal == "Iris-virginica":
                    if x >= 140 and x < 150:
                        testErrorVirginica += 1
                    else:
                        trainErrorVirginica += 1

        trainError = (trainErrorVirginica + trainErrorVersicolor + trainErrorSetosa) / 120
        testError = (testErrorSetosa + testErrorVersicolor + testErrorVirginica) / 30

        testErrorSetosa /= 10
        testErrorVirginica /= 10
        testErrorVersicolor /= 10

        trainErrorSetosa /= 40
        trainErrorVirginica /= 40
        trainErrorVersicolor /= 40
        print("The test error rate was {}".format(testError))
        print("The training error rate was {}".format(trainError))

        print("The Setosa test error rate was {}".format(testErrorSetosa))
        print("The Versicolor test error rate was {}".format(testErrorVersicolor))
        print("The Virginica test error rate was {}".format(testErrorVirginica))

        print("The Setosa training error rate was {}".format(trainErrorSetosa))
        print("The Versicolor training error rate was {}".format(trainErrorVersicolor))
        print("The Virginica training error rate was {}".format(trainErrorVirginica))

    def LDA(self):
        """
        This function sets up the p.d.f for LDA classification by setting all the sigmas to be the average.
        :return: None.
        """
        sigmaAvg = self.sigma + self.sigmaTwo + self.sigmaThree
        sigmaAvg /= 3

        self.sigma = sigmaAvg
        self.sigmaTwo = sigmaAvg
        self.sigmaThree = sigmaAvg

    def findLeastImportantFeauture(self):
        """
        This function turns off individual features one at a time to see the effect that the feature has on the pdf.
        :return: None.
        """
        self.extractTrainingSet()
        self.calculateMuMean()
        self.calculateSigma()
        self.LDA()
        self.calculateProbabilityDensityFunc()

        self.numOfFeatures = 3

        self.includeSepWidth = False
        self.extractTrainingSet()
        self.calculateMuMean()
        self.calculateSigma()
        self.LDA()
        self.calculateProbabilityDensityFunc()
        self.includeSepWidth = True

        self.includeSepLength = False
        self.extractTrainingSet()
        self.calculateMuMean()
        self.calculateSigma()
        self.LDA()
        self.calculateProbabilityDensityFunc()
        self.includeSepLength = True

        self.includePetWidth = False
        self.extractTrainingSet()
        self.calculateMuMean()
        self.calculateSigma()
        self.LDA()
        self.calculateProbabilityDensityFunc()
        self.includePetWidth = True

        self.includePetLength = False
        self.extractTrainingSet()
        self.calculateMuMean()
        self.calculateSigma()
        self.LDA()
        self.calculateProbabilityDensityFunc()
        self.includePetLength = True

        self.numOfFeatures = 1
        self.includeSepWidth = False
        self.includeSepLength = False
        self.includePetLength = False
        self.extractTrainingSet()
        self.calculateMuMean()
        self.calculateSigma()
        self.LDA()
        self.calculateProbabilityDensityFunc()

if __name__ == "__main__":
    t = HW2()
    #t.findLeastImportantFeauture() #Uncomment to enable finding the least important feature
    #exit() #Uncomment to stop code from running bottom

    t.extractTrainingSet()
    t.calculateMuMean()
    t.calculateSigma()
    #t.LDA() #Uncomment this to enable LDA
    t.calculateProbabilityDensityFunc()
