import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Perceptron
from sklearn import tree





ret = os.listdir('dados/')
print(ret)
for in_file in ret:
    with open('dados/'+in_file) as f: 
        indice = 'teste'
        if(f.readline() == '@inputs'): 
            indice = f.readline()
        df = pd.read_csv(f,comment="@",header = None)

        encoder = LabelEncoder()
        df = df.apply(encoder.fit_transform)
      
        collumn_target = df.iloc[:,-1]
        collumn_data = df.iloc[:,:-1]
        df.iloc[:,:-1]
        X = collumn_data.values
        y = collumn_target.values

        for j in range(30):
            print('LDA')
            lda = LinearDiscriminantAnalysis()
            score = cross_val_score(lda.fit(X,y), X, y, cv=5)
            print(score)
            print('Perceptron')
            ppn = Perceptron(max_iter = 40, eta0=0.1, random_state=1)
            ppn.fit(X, y)
            scores = cross_val_score(ppn.fit(X, y), X, y, cv=5)
            print(scores)
            print('Decision Tree')
            nordan_tree = tree.DecisionTreeClassifier()
            scores = cross_val_score(nordan_tree.fit(X, y), X, y, cv=5)
            print(scores)

