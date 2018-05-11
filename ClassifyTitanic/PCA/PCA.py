import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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

    # Combine data with flag for survived, dead, or test data
    trainingSet['Survival'] = 'Alive'
    trainingSet.loc[trainingSet['Survived'] == 0, 'Survival'] = 'Dead'
    testSet['Survival'] = 'Test'
    allSet = pd.concat([trainingSet, testSet])
    allSet['PassengerId'] = allSet['PassengerId'] - 1
    allSet = allSet.set_index('PassengerId')
    survival = allSet['Survival']

    # Select for relevant data and scale it
    allSet = allSet.drop(columns=['Survived', 'Parch', 'SibSp'])
    allSet = allSet.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64'])
    allSet = allSet.apply(lambda x: (x - x.min())/(x.max() - x.min()))
    allSet = allSet.apply(lambda x: x.fillna(x.mean()))

    # PCA
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(allSet)
    pcs = pd.DataFrame(data=pcs, columns=['PC1', 'PC2'])
    pcs = pcs.assign(Survival=survival)

    # Finally plot
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('PC1', fontsize=15)
    ax.set_ylabel('PC2', fontsize=15)
    ax.set_title('PCA for titanic data set (minimal cleaning)', fontsize=20)

    survival = ['Test', 'Alive', 'Dead']
    colors = ['b', 'g', 'r']
    for surv, color in zip(survival, colors):
        tarInd = pcs['Survival'] == surv
        ax.scatter(pcs.loc[tarInd, 'PC1'], pcs.loc[tarInd, 'PC2'], c=color, s=50)
    ax.legend(survival)
    ax.grid()
    fig.savefig('minCleanPCA.pdf')

if __name__ == "__main__":
    main()