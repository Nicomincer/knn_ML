import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as pt
import numpy as np 
from sklearn.model_selection import cross_val_score

#from imblearn.over_sampling import RandomOverSampler

df = pd.read_csv("archive/Iris.csv")

train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])


X_train = train[train.columns[:-1]].values
y_train = train[train.columns[-1]].values
X_test = valid[valid.columns[:-1]].values
y_test = valid[valid.columns[-1]].values
label_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
y_train = [label_map[label] for label in y_train]
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)
print(y_pred)
a = []
for valor in y_test:
    if valor == "Iris-setosa":
        a.append(0)
    elif valor == "Iris-versicolor":
        a.append(1)
    else:
        a.append(2)
print(a)
contagem = 0
for c in range(0, len(y_pred)):
    if y_pred[c] == a[c]:
        contagem+= 1

erros = len(a)-contagem
acertos = contagem
print(contagem)
print(f"Houve {acertos} acertos e {erros} erros, com rendimento de {(acertos/len(a))*100}%")