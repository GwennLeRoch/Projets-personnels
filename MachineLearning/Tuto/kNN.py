# Preparation des donnees

## Chargement de la librairie nécessaire au téléchargement du dataset
from sklearn.datasets import fetch_mldata

import numpy as np
import matplotlib.pyplot as plt

## Telechargement du dataset MNIST : 70000 images classées de chiffres manuscrits
mnist = fetch_mldata('MNIST original')              #on peut dl dans un dossier mldata le fichier mnist-original.mat et le charger en rajoutant l'argument "data_home=location"

## Le data set des images 28*28=784 pixels NB [0(blanc),16(noir)]
print(mnist.data.shape)

## Le vecteur annotations correspondant à la valeur "lue" du chiffre
print(mnist.target.shape)

### Echantillonage : dataset trop gros pour obtenir rapidement des résultats
#sample = np.random.randint(70000, size=5000)         #ce samplage est inexact, regarder sklearn.utils.resample
#data = mnist.data[sample]
#target = mnist.target[sample]
#
### Chargement de la librairie de séparation du dataset
#from sklearn.model_selection import train_test_split
#
### X(images d'exemple), y(annotations cibles)  séparés en proportion 80/20
#Xtrain, Xtest, ytrain, ytest = train_test_split(data, target, train_size=0.8)
#
#
#
## Classifieur 5-NN (5 plus proches voisins)
#
### Chargement de le librairie de l'algorithme correspondant
#from sklearn import neighbors
#
#knn = neighbors.KNeighborsClassifier(n_neighbors=5) #declaration de l'algorithme utilise et parametrage
#knn.fit(Xtrain, ytrain)                             #fait correspondre le modele avec X en datatrain et y en target value
#
### Test de l'erreur du claffifieur
#error = 1 - knn.score(Xtest, ytest)                 #score renvoie le pourcentage de prédiction véridique
#print('Erreur : %f' % error)
#
## Optimisation du score en cherchant le k optimal [2,15]
##errors = []
##for k in range(2,15):
##    knn = neighbors.KNeighborsClassifier(k)
##    errors.append(100*(1 - knn.fit(Xtrain,ytrain).score(Xtest, ytest)))
##plt.plot(range(2,15), errors, 'o-')
##plt.show()
#
## Recuperation des predictions sur les donnees test
#predicted = knn.predict(Xtest)
#
## Redimensionnement des donnees sous forme d'images
#images = Xtest.reshape((-1,28,28))
#
### Selection d'echantillon de 12 images au hasard
##select = np.random.randint(images.shape[0], size=12)
##
### Affichage des images avec prediction associee
##for index, value in enumerate(select):
##    plt.subplot(3,4,index+1)
##    plt.axis('off')
##    plt.imshow(images[value], cmap=plt.cm.gray_r, interpolation="nearest")
##    plt.title('Predicted: %i' % predicted[value])
##plt.show()
#
## Recuperation des donnes mal predites
#missclass = (ytest!=predicted)
#print(images.shape)
#missclass_images = images[missclass,:,:]
#missclass_predicted = predicted[missclass]
#
## Selection d'un echantillon de 12 images mal predites
#select = np.random.randint(missclass_images.shape[0], size=12)
#
## Affichage de ces erreurs
#for index, value in enumerate(select):
#    plt.subplot(3,4,index+1)
#    plt.axis('off')
#    plt.imshow(missclass_images[value], cmap='Greys', interpolation="nearest")
#    plt.title('Predicted: %i' %missclass_predicted[value])
#plt.show()