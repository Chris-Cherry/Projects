import pandas as pd
import numpy as np

class KNN:
    """Supervised learning technique which classifies data based on the
    the identity of the nearest k neighbors (k nearest neihbors)

    Parameters:
    ----------
    k: int
        The number of neighbors k to classify with
    """

    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        numericX = X.select_dtypes(include=numerics)
        numericX = numericX.apply(lambda x: x.fillna(x.mean()))
        numericX = numericX.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        self.refMatrix = numericX
        self.refIds = y

    def predict(self, X):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        numericX = X.select_dtypes(include=numerics)
        numericX = numericX.apply(lambda x: x.fillna(x.mean()))
        numericX = numericX.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        prediction= np.empty(numericX.shape[0], dtype = int)
        for i in range(0, numericX.shape[0]):
            tarVec = numericX.iloc[i, :]
            dist = self.refMatrix.apply(lambda x: sum((tarVec - x) ** 2), axis=1)
            matchInd = np.argsort(dist)[:self.k]
            prediction[i] = int(round(self.refIds.iloc[matchInd].mean()))
            print(i)
        return(prediction)


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

    # Get survival data and trim redundancies
    testPassengerIds = testSet['PassengerId']
    ident = trainingSet['Survived']
    trainingSet = trainingSet.drop(columns=['PassengerId', 'Survived'])
    testSet = testSet.drop(columns = ['PassengerId'])

    # Make prediciton with KNN
    knnModel = KNN(k=3)
    knnModel.fit(X=trainingSet, y=ident)
    predictions = knnModel.predict(testSet)

    predictions = pd.DataFrame({'Survived': predictions, 'PassengerId': testPassengerIds})

    # Write predictions
    predictions.to_csv('~/Projects/ClassifyTitanic/Predictions/KNNk3.csv', index=False)


if __name__ == "__main__":
    main()
