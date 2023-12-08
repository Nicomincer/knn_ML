import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as pt
import numpy as np 

#from imblearn.over_sampling import RandomOverSampler

df = pd.read_csv("archive/Iris.csv")
print(df.shape)
df.loc[df["Species"] == "Iris-setosa"] = 0 
df.loc[df["Species"] == "Iris-versicolor"] = 1
df.loc[df["Species"] == "Iris-virginica"] = 2


train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])

"""def scale_dataset(dataframe, oversample=False):
  X = dataframe[dataframe.columns[:-1]].values
  y = dataframe[dataframe.columns[-1]].values

  scaler = StandardScaler()
  X = scaler.fit_transform(X)

  if oversample:
    ros = RandomOverSampler()
    X, y = ros.fit_resample(X, y)

  data = np.hstack((X, np.reshape(y, (-1, 1))))

  return data, X, y



train, X_train, y_train = scale_dataset(train, oversample=True)
valid, X_valid, y_valid = scale_dataset(valid, oversample=False)
test, X_test, y_test = scale_dataset(test, oversample=False)
"""
X_train = df[df.columns[:-1]].values
y_train = df[df.columns[-1]].values
knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(X_train, y_train)
