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
X = np.matrix([np.ones(n), houseData['surface'].as_matrix(), houseData['arrondissement'].as_matrix()]).T
y = np.matrix(houseData['price']).T
print(n,X.shape,y.shape)
plt.scatter([X[:,0].T], [y.T], c=[X[:,1].T])
plt.show()

# Separation des training/testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)



# Traitement

# Regression lineaire
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
y_predict_reg = regr.predict(X_test)
error_reg = regr.score(X_test, y_test)*100
print('Erreur regression linéaire : %f' %error_reg)

# Affichage des donnees et des erreurs
plt.title("Regression linéaire et erreurs de l'algo")
plt.xlabel("Surface")
plt.ylabel("Prix")
print(X_train.shape, y_train.shape)
plt.plot(X_train[:,1], y_train, 'ko')           #on plot les points du datatest
plt.plot(X_test[:,1], y_predict_reg, 'ro')          #on plot les predictions
plt.plot(X_test[:,1], y_test, 'go')
print(X_test[:,1].shape)
for i, Xi in enumerate(X_test[:,1]):
    plt.plot([Xi.item(0),Xi.item(0)], [y_predict_reg[i],y_test[i]], 'r')
plt.show()

# Classifieur k-NN
errors = []
knns = []
for k in range(2,15):
    knn = neighbors.KNeighborsClassifier(k)
    knn.fit(X_train, y_train)
    knns.append(knn)
    errors.append(100*(1-knn.score(X_test, y_test)))

# Choix du parametre k en comparant les precisions des modeles
plt.title("Précision des modèles de regression linéaire (line) et k-NN (courbe)")
plt.xlabel("k")
plt.ylabel("Précision")
plt.plot([2,15],[error_reg,error_reg])
plt.plot(range(2,15), errors, 'o-')
plt.show()
k = input("Entrez un k maximisant la précision :\n")
y_predict_knn = knn.predict(X_test)

# Comparaison des modeles
errors_regr = np.linalg.norm

# Affichage des resultats
plt.title('Comparaison des résultats entre régression linéaire et k-NN')
plt.xlabel('Surface')
plt.ylabel('Prix')
plt.plot(X_test[:,1], y_test, 'ko')
plt.plot(X_test[:,1], y_predict_reg, 'bx')
plt.plot(X_test[:,1], y_predict_knn, 'g+')
plt.show()