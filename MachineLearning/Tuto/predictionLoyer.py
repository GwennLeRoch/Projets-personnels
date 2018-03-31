import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import neighbors



# Initiatisation

# Recuperation des donnees
houseData = pd.read_csv(os.path.realpath("data/house_data.csv"))
houseData=houseData.dropna(axis=0, how='any')
n = houseData.shape[0]
X = np.matrix([houseData['surface'], houseData['arrondissement']]).T
y = np.matrix(houseData['price']).T
print(n,X.shape,y.shape)

# Separation des training/testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)



# Traitement

# Regression lineaire
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
y_predict = regr.predict(X_test)
error_reg = regr.score(X_test, y_test)*100
print('Erreur regression linéaire : %f' %error_reg)

# Classifieur k-NN
errors = []
for k in range(10,20):
    knn = neighbors.KNeighborsClassifier(k)
    errors.append(100*(1-knn.fit(X_train, y_train).score(X_test, y_test)))

# Comparaison des modeles
plt.plot([10,20],[error_reg,error_reg])
plt.plot(range(10,20), errors, 'o-')
plt.show()