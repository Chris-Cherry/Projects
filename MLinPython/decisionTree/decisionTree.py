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

    def predict(self, X):
        assert self.tree is not None, "Run fit before predict!"
        predictions = []
        # Step through the samples
        for sample_index in range(X.shape[0]):
            predictions.append(
                self.navigateBranch(X.iloc[sample_index, :], self.tree)
            )
        return (predictions)

    def calcInformationGain(self, X, y, iFeat, val):
        base_p = float(sum(y)) / len(y)
        base_q = 1 - base_p
        base_entropy = -base_p * math.log(base_p + .001, 2) \
                       - base_q * math.log(base_q + .001, 2)

        left_index = X.iloc[:, iFeat] >= val
        left_p = float(sum(y.loc[left_index])) / len(y.loc[left_index])
        left_q = 1 - left_p
        left_entropy = -left_p * math.log(left_p + .001, 2) \
                       - left_q * math.log(left_q + .001, 2)

        right_index = X.iloc[:, iFeat] < val
        right_p = float(sum(y.loc[right_index])) / len(y.loc[right_index])
        right_q = 1 - right_p
        right_entropy = -right_p * math.log(right_p + .001, 2) \
                        - right_q * math.log(right_q + .001, 2)

        post_entropy = left_entropy * float(sum(left_index)) / len(y) \
                       + right_entropy * float(sum(right_index)) / len(y)

        return base_entropy - post_entropy

    def buildTree(self, X, y, level=0):
        best_split = {}
        # Check that current level is below max depth

        if level <= self.depth:
            # Step through features
            best_info_gain = 0
            for iFeat in range(X.shape[1]):
                featValues = X.iloc[:, iFeat]
                uniqueVals = featValues.unique()
                for val in uniqueVals:
                    # If there is no split made then skip
                    leftIndex = X.iloc[:, iFeat] >= val
                    rightIndex = X.iloc[:, iFeat] < val
                    if sum(leftIndex) == 0 or sum(rightIndex) == 0:
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

    def navigateBranch(self, sample_vec, node):
        if node.classification is not None:
            return node.classification
        elif node.thresh is not None:
            sample_value = sample_vec[node.featIndex]
            if sample_value >= node.thresh:
                prediction = self.navigateBranch(sample_vec, node.Ltree)
                return prediction
            elif sample_value < node.thresh:
                prediction = self.navigateBranch(sample_vec, node.Rtree)
                return prediction
            else:
                print("Sample value didn't meet threshold expectation")
        else:
            print("Node classification and threshold both not defined")


def main():
    training_set = pd.read_csv(
        '~/Projects/MLinPython/data/train.csv',
        quotechar='"'
    )
    test_set = pd.read_csv(
        '~/Projects/MLinPython/data/test.csv',
        quotechar='"'
    )

    # Convert Sex to binary
    training_set['Sex'] = training_set['Sex'].replace('male', 1)
    training_set['Sex'] = training_set['Sex'].replace('female', -1)
    test_set['Sex'] = test_set['Sex'].replace('male', 1)
    test_set['Sex'] = test_set['Sex'].replace('female', -1)

    # Get survival data
    train__survival = training_set['Survived']
    test_ids = test_set['PassengerId']

    # Get rid of non-informative columns
    training_set = training_set.drop(['PassengerId', 'Survived', 'Name', 'Ticket',
                                      'Cabin', 'Embarked'], axis=1)
    test_set = test_set.drop(['PassengerId', 'Name', 'Ticket', 'Cabin',
                              'Embarked'], axis=1)

    # The data must be cleaned of Na values before running. We will
    # simply replace the missing values with the mean of the column
    # for this simple example.

    training_set = training_set.apply(lambda x: x.fillna(x.mean()))
    test_set = test_set.apply(lambda x: x.fillna(x.mean()))

    tree = decisionTree(depth=10)
    tree.fit(X=training_set, y=train__survival)
    predictions = tree.predict(test_set)
    predictions = map(int, predictions)
    predictions = pd.DataFrame({'Survived': predictions, 'PassengerId': test_ids})
    predictions.to_csv('~/Projects/MLinPython/decisionTree/predictions.csv', index=False)


if __name__ == "__main__":
    main()
