import pandas as pd
import numpy as np
import math

class perceptron:
    """Implimentatin of a two later perceptron using back-propogation.

    Parameters:
    ----------
    h_nodes: int
        The number of nodes in the hidden layer
    """

    def __init__(self, h_nodes=3):
        self.h_nodes = h_nodes

    def activationFunction(self, x):
        # Activation function for the network defined here
        out = 1/(1+math.exp(-x))
        return(out)

    def makeNetwork(self, X, y):
        # X is training set and y is training outcomes
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        numericX = X.select_dtypes(include=numerics)
        numericX = numericX.apply(lambda x: x.fillna(x.mean()))
        # Don't normalize for first pass numericX = numericX.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        self.w1 = np.random.rand(numericX.shape[1], self.h_nodes) * 2 - 1
        self.b1 = np.zeros((numericX.shape[0],self.h_nodes))
        self.w2 = np.random.rand(self.h_nodes) * 2 - 1
        self.b2 = np.zeros(numericX.shape[0])
        self.xFit = numericX
        yFit = (2*y - 1) # Convert to -1 and 1 for live/dead
        self.yFit = np.array(yFit).astype('float64')

    def fit(self, iterations):
        self.errors = np.empty(iterations, float)
        vAF = np.vectorize(self.activationFunction)
        for i in range(0, iterations):
            # Feed forward
            z1 = vAF(np.dot(self.xFit, self.w1) + self.b1)
            yEst = vAF(np.dot(z1, self.w2) + self.b2)

            # Get error function
            error = sum(np.square(self.yFit - yEst))
            self.errors[i] = error

            # Feed back to change w1, w2, b1, b2
        return (error)


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
    perceptronModel.fit(iterations=1000)
    #predictions = perceptronModel.predict(testSet)

    predictions = pd.DataFrame({'Survived': predictions,
                                'PassengerId': testPassengerIds})

    # Write predictions
    predictions.to_csv('~/Projects/ClassifyTitanic/KNN/predictions.csv',
                       index=False)


if __name__ == "__main__":
    main()