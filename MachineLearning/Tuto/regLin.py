import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path

# Chargement des données
houseData = pd.read_csv(os.path.realpath("data/house.csv"))

# X = (1, x1, x2, ..., xN), le 1 permet d'avoir l'ordonnée à l'origine
n = houseData.shape[0]
X = np.matrix([np.ones(n), houseData['surface'].as_matrix()]).T
y = np.matrix(houseData['loyer']).T

# Calcul des coefficients
theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

# Affichage
plt.title("house.csv")
plt.xlabel("Surface")
plt.ylabel("Loyer")
plt.plot(houseData['surface'], houseData['loyer'], 'ro', markersize=4)

plt.plot([0,n],[theta.item(0), theta.item(0) + n*theta.item(1)], linestyle='--', c='black')

plt.show()

# Exemple estimation
print(theta.item(0) + theta.item(1)*35)