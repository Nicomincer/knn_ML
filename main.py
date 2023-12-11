import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier
import numpy as np 
from sklearn.metrics import precision_score

#Iris-dataset

#getting the dataset Iris
df = pd.read_csv("archive/Iris.csv")

#creating the train, valid and test datasets. 
train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])


#Normalizing and creating our vector features. 
X_train = train[train.columns[:-1]].values
y_train = train[train.columns[-1]].values
X_test = test[test.columns[:-1]].values
y_test = test[test.columns[-1]].values
label_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
y_train = [label_map[label] for label in y_train]
y_test = [label_map[label] for label in y_test]

#starting the training
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

#Our predict
y_pred1 = knn_model.predict(X_test)

#Results
print(precision_score(y_test, y_pred1, pos_label="positive", average="micro"))