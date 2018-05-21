import numpy as np
import pandas as pd
import math


class node:
    """Recursive object used to define the decision tree.

    Parameters:
    ----------
    Ltree: node
        Object defining the left side of the branch
    Rtree: node
        Object defining the right side of the branch
    classification: int
        The classification decision if 'node' is the end of a branch
    thresh: float
        Threshold used to make a decision for branch based on feature
        if node is a branch
    featIndex: int
        Index of feature used to make decision if node is a branch
    """

    def __init__(self, Ltree=None, Rtree=None, classification=None,
                 thresh=None, featIndex=None):
        self.Ltree = Ltree
        self.Rtree = Rtree
        self.classification = classification
        self.thresh = thresh
        self.featIndex = featIndex


class decisionTree:
    """Decision tree object.

    Parameters:
    ----------
    depth:  int
        The maximum depth of the tree.

    Public functions:
    ----------
    decisionTree.fit(X, y)
        X:  data.frame (float)
            The feature data that will be used to create the decision
            tree. Expects a Pandas data.frame
        y: series (binary)
            The classification data that will be used to create the
            decision tree
            Expects a Pandas series
    Returns none.

    Creates a decision tree based on feature and classification data.
    Must be called prior to decisionTree.predict .


    decisionTree.predict(X)
        X: data.frame (float)
            The feature data that will be used to generate a prediction
    Returns a Pandas series of predictions

    Make a prediction of class based on feature data with the previously
    built model (decisionTree.fit).
    """

    def __init__(self, depth=10):
        self.depth = depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self.buildTree(X, y)

    def calcInformationGain(self, X, y, iFeat, val):
        baseP = float(sum(y)) / len(y)
        baseQ = 1 - baseP
        baseEntropy = -baseP * math.log(baseP+.001, 2) - baseQ * math.log(baseQ+.001, 2)

        leftIndex = X.iloc[:, iFeat] >= val
        leftP = float(sum(y.loc[leftIndex])) / len(y.loc[leftIndex])
        leftQ = 1 - leftP
        leftEntropy = -leftP * math.log(leftP+.001, 2) - leftQ * math.log(leftQ+.001, 2)

        rightIndex = X.iloc[:, iFeat] < val
        rightP = float(sum(y.loc[rightIndex])) / len(y.loc[rightIndex])
        rightQ = 1 - rightP
        rightEntropy = -rightP * math.log(rightP+.001, 2) - rightQ * math.log(rightQ+.001, 2)

        postEntropy = leftEntropy * float(sum(leftIndex)) / len(y) \
                      + rightEntropy * float(sum(rightIndex)) / len(y)

        return (baseEntropy - postEntropy)

    def buildTree(self, X, y, level=0):
        best_split = {}
        # Check that current level is below max depth

        if level <= self.depth:
            # Step through features
            best_info_gain = 0
            for iFeat in range(X.shape[1]):
                featValues = X.iloc[:, iFeat]
                uniqueVals = featValues.unique()
                print(np.nditer(uniqueVals))
                for val in uniqueVals:
                    # If there is no split made then skip
                    leftIndex = X.iloc[:, iFeat] >= val
                    rightIndex = X.iloc[:, iFeat] < val
                    if sum(leftIndex)==0 or sum(rightIndex)==0:
                        continue

                    # If there's actually a split made, get information gain
                    info_gain = self.calcInformationGain(X, y, iFeat, val)
                    if info_gain > best_info_gain and info_gain > .05:
                        best_info_gain = info_gain
                        best_split['Feature Index'] = iFeat
                        best_split['Value'] = val


                        best_split['leftY'] = y[leftIndex]
                        best_split['leftX'] = X[leftIndex]
                        best_split['rightY'] = y[rightIndex]
                        best_split['rightX'] = X[rightIndex]
        # We now have either the best split or no split if requirements aren't
        # met. If we have the split, then we build a new branch on either side.
        # Otherwise this is a leaf.

        if len(best_split) == 0:
            best_prediction = round(float(sum(y)) / len(y))
            return (node(classification=best_prediction))
        else:
            Ltree = self.buildTree(best_split['leftX'], best_split['leftY'],
                                   level=level + 1)
            Rtree = self.buildTree(best_split['rightX'], best_split['rightY'],
                                   level=level + 1)
            return (node(Ltree=Ltree, Rtree=Rtree, thresh=best_split['Value'],
                         featIndex=best_split['Feature Index']))


def main():
    trainingSet = pd.read_csv(
        '~/Projects/ClassifyTitanic/data/train.csv',
        quotechar='"'
    )
    testSet = pd.read_csv(
        '~/Projects/ClassifyTitanic/data/test.csv',
        quotechar='"'
    )

    # Convert Sex to binary
    trainingSet['Sex'] = trainingSet['Sex'].replace('male', 1)
    trainingSet['Sex'] = trainingSet['Sex'].replace('female', -1)
    testSet['Sex'] = testSet['Sex'].replace('male', 1)
    testSet['Sex'] = testSet['Sex'].replace('female', -1)

    # Get survival data
    Train_Survival = trainingSet['Survived']

    # Get rid of non-informative columns
    trainingSet = trainingSet.drop(['PassengerId', 'Survived', 'Name', 'Ticket',
                                    'Cabin', 'Embarked'], axis=1)
    testSet = testSet.drop(['PassengerId', 'Name', 'Ticket', 'Cabin',
                            'Embarked'], axis=1)

    tree = decisionTree(depth=10)
    tree.fit(X=trainingSet, y=Train_Survival)

    print('hi')


if __name__ == "__main__":
    main()