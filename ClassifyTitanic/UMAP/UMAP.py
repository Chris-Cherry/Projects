import pandas as pd
import umap
import matplotlib.pyplot as plt

def main():
    trainingSet = pd.read_csv(
        '~/Documents/Chris_Cherry/Projects/ClassifyTitanic/data/train.csv',
        quotechar='"'
    )
    testSet = pd.read_csv(
        '~/Documents/Chris_Cherry/Projects/ClassifyTitanic/data/test.csv',
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

    # Run UMAP on data and plot
    embeddings = umap.UMAP(n_neighbors = 150,
                           min_dist = .5).fit_transform(allSet)
    embeddings = pd.DataFrame(data = embeddings, columns = ['UMAP 1', 'UMAP 2'],
                              index = allSet.index.values)
    embeddings = pd.concat([embeddings, survival], axis = 1)
    
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1, 1, 1,)
    ax.set_xlabel('UMAP 1', fontsize=15)
    ax.set_ylabel('UMAP 2', fontsize=15)
    ax.set_title('UMAP for titanic data set (minimal cleaning)', fontsize = 20)
    survival = ['Test', 'Alive', 'Dead']
    colors = ['b', 'g', 'r']

    for surv, color in zip(survival, colors):
        tarInd = embeddings['Survival'] == surv
        ax.scatter(embeddings.loc[tarInd, 'UMAP 1'], embeddings.loc[tarInd, 'UMAP 2'],
                   c = color, s = 50, alpha = .1)
        plt.show()
        
    ax.legend(survival)
    ax.grid()
    fig.savefig('test.pdf')
    
if __name__ == "__main__":
    main()
