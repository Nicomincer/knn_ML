import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier
import numpy as np 
from sklearn.metrics import classification_report

df = pd.read_csv("archive/Iris.csv")

train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])


X_train = train[train.columns[:-1]].values
y_train = train[train.columns[-1]].values
X_test = test[test.columns[:-1]].values
y_test = test[test.columns[-1]].values
label_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
y_train = [label_map[label] for label in y_train]
y_test = [label_map[label] for label in y_test]

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

y_pred1 = knn_model.predict(X_test)

print(classification_report(y_test, y_pred1))