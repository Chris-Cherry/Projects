import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import math

class perceptron:
    """Implementation of a two later perceptron using back-propogation.

    Parameters:
    ----------
    h_nodes: int
        The number of nodes in the hidden layer
    """

    def __init__(self, h_nodes=3):
        self.h_nodes = h_nodes

    def activationFunction(self, x):
        # Activation function for the network defined here. Make sure to update the derivative function as well.
        out = 1/(1+math.exp(-x))
        return(out)

    def activationFunctionDerivative(self, x):
        # Derivative of the activation function defined here
        out = self.activationFunction(x)*(1-self.activationFunction(x))
        return(out)

    def makeNetwork(self, X, y):
        # X is training set and y is training outcomes
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        numericX = X.select_dtypes(include=numerics)
        numericX = numericX.apply(lambda x: x.fillna(x.mean()))
        # Normalize or clean data here if desired
        numericX = numericX.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        self.w1 = np.random.rand(numericX.shape[1], self.h_nodes) * 2 - 1
        self.b1 = np.zeros((numericX.shape[0],self.h_nodes))
        self.w2 = np.random.rand(self.h_nodes) * 2 - 1
        self.b2 = np.zeros(numericX.shape[0])
        self.xFit = numericX
        yFit = (2*y - 1) # Convert to -1 and 1 for live/dead
        self.yFit = np.array(yFit).astype('float64')

    def fit(self, iterations, use_b=False):
        
        self.errors = np.empty(iterations, float)
        vAF = np.vectorize(self.activationFunction)
        vAFD = np.vectorize(self.activationFunctionDerivative)
        for i in range(0, iterations):
            if i%100 == 0:
                print(i)
            # Feed forward
            z1 = vAF(np.dot(self.xFit, self.w1) + self.b1)
            yEst = vAF(np.dot(z1, self.w2) + self.b2)

            # Get error
            error = sum(np.square(self.yFit - yEst))
            self.errors[i] = error

            # Feed back to change w1, w2, b1, b2 
            db2 = 2 * (self.yFit-yEst) * vAFD(np.dot(z1, self.w2) + self.b2)
            dw2 = np.dot(z1.T, db2)
            derPart = 2*(self.yFit-yEst)*vAFD(np.dot(z1, self.w2)+self.b2)
            derPart = np.reshape(derPart, (derPart.shape[0], 1))
            w2Inv = np.reshape(self.w2, (1, self.w2.shape[0]))
            db1 = np.dot(derPart, w2Inv) * vAFD(z1)
            dw1 = np.dot(self.xFit.T, db1)

            if use_b:
                self.b1 += 0.01*db1
                self.b2 += 0.01*db2

            self.w1 += 0.01*dw1
            self.w2 += 0.01*dw2
            
        return

    def predict(self, X, use_b=False):
        # Clean input data the same as x training data
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        numericX = X.select_dtypes(include=numerics)
        numericX = numericX.apply(lambda x: x.fillna(x.mean()))
        # Normalize or clean data here if desired
        numericX = numericX.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        vAF = np.vectorize(self.activationFunction)

        # Make predictions
        if use_b:
            z1 = vAF(np.dot(numericX, self.w1)+self.b1)
            predictions = vAF(np.dot(z1, self.w2)+self.b2)

        if not use_b:
            z1 = vAF(np.dot(numericX, self.w1))
            predictions = vAF(np.dot(z1, self.w2))
        return(np.round(predictions))


def main():
    trainingSet = pd.read_csv(
        '~/Projects/MLinPython/data/train.csv',
        quotechar='"'
    )
    testSet = pd.read_csv(
        '~/Projects/MLinPython/data/test.csv',
        quotechar='"'
    )

    # Convert Sex to binary
    trainingSet['Sex'] = trainingSet['Sex'].replace('male', 1)
    trainingSet['Sex'] = trainingSet['Sex'].replace('female', -1)
    testSet['Sex'] = testSet['Sex'].replace('male', 1)
    testSet['Sex'] = testSet['Sex'].replace('female', -1)

    # Get survival data and trim redundancies
    testPassengerIds = testSet['PassengerId']
    ident = trainingSet['Survived']
    trainingSet = trainingSet.drop(columns=['PassengerId', 'Survived'])
    testSet = testSet.drop(columns=['PassengerId'])

    # Make prediciton with KNN
    perceptronModel = perceptron(h_nodes=3)
    perceptronModel.makeNetwork(X=trainingSet, y=ident)
    perceptronModel.fit(iterations=100)

    # Plot the errors of the fit
    with PdfPages('Errors.pdf') as pdf:
        fig = plt.figure()
        plt.plot(perceptronModel.errors)
        pdf.savefig(fig)
        plt.close()

    predictions = perceptronModel.predict(testSet)

    predictions = pd.DataFrame({'Survived': predictions,
                                'PassengerId': testPassengerIds})

    # Write predictions
    predictions.to_csv('~/Projects/MLinPython/perceptron/predictions.csv',
                        index=False)


if __name__ == "__main__":
    main()