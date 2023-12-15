import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np 
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import precision_score, classification_report

#Iris-dataset

#getting the dataset Iris
df = pd.read_csv("archive/Iris.csv")

#creating the train, valid and test datasets. 
train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])


#Normalizing and creating our vector features. 
# X_train = train[train.columns[:-1]].values
# y_train = train[train.columns[-1]].values
# X_test = test[test.columns[:-1]].values
# y_test = test[test.columns[-1]].values
# label_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
# y_train = [label_map[label] for label in y_train]
# y_test = [label_map[label] for label in y_test]

def scale_dataset(dataframe, oversample=False):
    X = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values
    label_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    y = [label_map[label] for label in y]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if oversample:
        ros = RandomOverSampler()
        X, y = ros.fit_resample(X, y)

    data = np.hstack((X, np.reshape(y, (-1, 1))))

    return data, X, y

train, X_train, y_train = scale_dataset(train, oversample=True)
test, X_test, y_test = scale_dataset(test, oversample=False)

#starting the training
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

nb_model = GaussianNB()
nb_model = nb_model.fit(X_train, y_train)

#Our predict
y_pred1 = knn_model.predict(X_test)
y_pred2 = nb_model.predict(X_test)

#Results
print(classification_report(y_test, y_pred1))
#print(classification_report(y_test, y_pred2))
#print(precision_score(y_test, y_pred1, average='macro'))