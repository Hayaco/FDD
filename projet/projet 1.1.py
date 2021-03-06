# Créé par haya, le 24/12/2017
from __future__ import division
from lycee import *


import numpy as np
from numpy.linalg import inv
import math
import matplotlib.pyplot as plt

# Partie 1:
# 1)
mu0 =np.array([0,0])
mu1 = np.array([3,2])
sigma=np.array([[1 , 1/2], [1/2, 1]])


#Cas pour les données d'apprentissage

x0 , y0 = np.random.multivariate_normal(mu0,sigma,10).T
x1 , y1 = np.random.multivariate_normal(mu1,sigma,10).T
x=np.concatenate((x0, x1))
y=np.concatenate((y0, y1))
print("Jeu de données d'apprentissage", '\n' , x,'\n', y, '\n')

#Création des catégories pour avoir les couleurs

categories=[]
for i in range(10):
    categories.append(0)

for i in range(10):
    categories.append(1)
colormap = np.array(['r', 'b'])

#Affichage du nuage de point avec les couleurs et les axes

#plt.scatter(x, y, c=colormap[categories])
#plt.title("Nuage de points pour les données d\'apprentissage")
#plt.xlabel('x')
#plt.ylabel('y')
#plt.savefig('NuageApprentissage.png') #Sauvegarde du nuage dans un fichier
#plt.show()

#Cas pour le test
t0 , z0 = np.random.multivariate_normal(mu0,sigma,1000).T
t1 , z1 = np.random.multivariate_normal(mu1,sigma,1000).T
tTest=np.concatenate((t0, t1))
zTest=np.concatenate((z0, z1))
#print("Jeu de données test", '\n' , tTest,'\n', zTest, '\n') à remettre

#Création des catégories pour avoir les couleurs

categoriesTest=[]

for i in range(1000):
    categoriesTest.append(0)

for i in range(1000):
    categoriesTest.append(1)
colormap = np.array(['r', 'b'])

#Affichage du nuage de point avec les couleurs et les axes

#plt.scatter(tTest, zTest, c=colormap[categoriesTest])
#plt.title("Nuage de points pour les données de test")
#plt.xlabel('t')
#plt.ylabel('z')
#plt.savefig('NuageTest.png') #Sauvegarde du nuage dans un fichier
#plt.show()

#partie 2
#1)

#Definition de la methode pour calculer nos estimateurs

def estimation(n, n0, n1, X0, Y0, X1, Y1 ) :
    PI0 = n0 / n
    PI1 = n1 / n
    XY0= np.concatenate([[X0], [Y0]])
    MU0= XY0.mean(1)
    XY1= np.concatenate([[X1], [Y1]])
    MU1= XY1.mean(1)
    SIGMA0= np.cov(X0, Y0, bias=True) # pour diviser par n et pas n-1
    SIGMA1= np.cov(X1, Y1, bias=True)
    SIGMA=(n0*SIGMA0+n1*SIGMA1)/(n0+n1)
    SIGMA_1 = inv(SIGMA)
    # print ("Matrice inversée ", SIGMA_1)
    W=np.dot(SIGMA_1, MU0-MU1)
    # print(W)
    T=-1/2 * np.dot(np.transpose(MU0-MU1), np.dot(SIGMA_1, MU0+MU1))+ math.log(PI0/PI1)
    # print (T)
    return W, T

def resulat(W, T, X):
    if ((np.dot(np.transpose(X), W)+T)>0): return 0
    else: return 1

w, t=estimation(20, 10 , 10 , x0, y0, x1, y1)
#print (w, t)
print ("resultat: classe ", resulat(w, t, np.array([5, 1])))

#verification  avec sklearn

# X et Y sont les deux colones de donnees tests que l'on a généré ( avec 1000 de C0, 1000 de C1)
# Nous on sait que les 1000 premier sont de la classe 0 et les 1000 suivant de la classe 1
# on va tester notre programme pour voir la performance de ce dernier
# on considère la classe 0 comme yes et 1 comme No
def taux_bonne_classificationtest(w, t, Ttest, Ztest):
    tp=0
    fp=0
    fn=0
    tn=0
    for i in range (1000):
        if (resulat(w, t, np.array([Ttest[i], Ztest[i]]))==0):
            tp+=1
        else:
            fn+=1
    for i in range (1000,2000):
        if(resulat(w, t, np.array([Ttest[i], Ztest[i]]))==1):
            tn+=1
        else:
            tn+=1
    return ((tp+tn)/(tp+tn+fp+fn))

def taux_bonne_classificationapprentissage(w, t, X, Y):
    tp=0
    fp=0
    fn=0
    tn=0
    for i in range (10):
        if (resulat(w, t, np.array([X[i], Y[i]]))==0):
            tp+=1
        else:
            fn+=1
    for i in range (10,20):
        if(resulat(w, t, np.array([X[i], Y[i]]))==1):
            tn+=1
        else:
            tn+=1
    return ((tp+tn)/(tp+tn+fp+fn))

print (" taux de bonne classification en test ",taux_bonne_classificationtest(w, t, tTest, zTest))
print (" taux de bonne classification en apprentissage ",taux_bonne_classificationapprentissage(w, t, x, y))
#resultat:
#(' taux de bonne classification en test ', 0.95850000000000002)
#(' taux de bonne classification en apprentissage ', 1.0)

# 2)
x0[0]=-10
y0[0]=-10
#print (x0,y0)
w, t=estimation(20, 10 , 10 , x0, y0, x1, y1)
print ("resultat: classe ", resulat(w, t, np.array([1, 3])))

#on observe que cette donne abérente fausse la prédiction

# resultat avant modification: classe  1
# resultat apres modification: classe  0

#3)

def graph(formula, x_range):
    x = np.array(x_range)
    y = formula(x)
    plt.plot(x, y)
    plt.plot()
    plt.show()

abs=np.linspace(-10, 10, 30)
#graph(lambda a : (-t - a * w[0])/w[1] , abs)

# on oberve que le frontière de décision est bien significative car elle sépare bien les deux classes
# bien que ce ne soit pas très précis, cela reste tout de même acceptable
# car on le rappele on doit préconiser un modèle simple qui marche plutôt qu'un trop compliqué
# qui risque de ne plus marcher sur des données différentes.

#4) je pense peute etre une distribution uniforme sinon j'ai pas d'idée.

#5)Si lambda vaut 1 on retombe sur l' ADL
# on doit recalcule les bons parametres w et T
# L = lambda entre 0 et 1

def estimationNewModele(n, n0, n1, X0, Y0, X1, Y1, L ) :
    PI0 = n0 / n
    PI1 = n1 / n
    XY0= np.concatenate([[X0], [Y0]])
    MU0= XY0.mean(1)
    XY1= np.concatenate([[X1], [Y1]])
    MU1= XY1.mean(1)
    SIGMA0= np.cov(X0, Y0, bias=True) # pour diviser par n et pas n-1
    SIGMA1= np.cov(X1, Y1, bias=True)
    SIGMA=L*(n0*SIGMA0+n1*SIGMA1)/(n0+n1)+ (1-L)*np.eye(2)
    #print (SIGMA)
    SIGMA_1 = inv(SIGMA)
    # print ("Matrice inversée ", SIGMA_1)
    W=np.dot(SIGMA_1, MU0-MU1)
    # print(W)
    T=-1/2 * np.dot(np.transpose(MU0-MU1), np.dot(SIGMA_1, MU0+MU1))+ math.log(PI0/PI1)
    # print (T)
    return W, T

def traceTauxBonneClassificationApp(X0, Y0, X1, Y1):
    Lambda= np.linspace(0, 1, 50 )
    X=np.concatenate((X0, X1))
    Y=np.concatenate((Y0, Y1))
    Taux=[]
    for i in Lambda:
        w, t=estimationNewModele(20, 10 , 10 , X0, Y0, X1, Y1, i)
        a=taux_bonne_classificationapprentissage(w, t, X, Y)
        Taux.append(a)
    plt.scatter(Lambda, Taux)
    plt.title("Taux de bonne classification en fonction de Lambda Aprentissage")
    plt.show()

traceTauxBonneClassificationApp(x0, y0, x1, y1)

def traceTauxBonneClassificationtest(X0, Y0, X1, Y1):
    Lambda= np.linspace(0, 1, 100 )
    X=np.concatenate((X0, X1))
    Y=np.concatenate((Y0, Y1))
    Taux=[]
    for i in Lambda:
        w, t=estimationNewModele(2000, 1000 , 1000 , X0, Y0, X1, Y1, i)
        a=taux_bonne_classificationapprentissage(w, t, X, Y)
        Taux.append(a)
    print(Taux)
    plt.scatter(Lambda, Taux)
    plt.title("Taux de bonne classification en fonction de Lambda test")
    plt.show()

#traceTauxBonneClassificationtest(t0, z0, t1, z1)
# on s'apercoit que dans le cas du test comme de l'apprentissage, plus on est proche de lambda=1
#plus la taux est élévé
# je suis pas sur

#6)
# cette methode retourne une valeur de lambda pour la laquelle le taux moyend e bonne classification
# est max
# on travail sur les donnes d'apprentissage
def validationcroise(X0, Y0, X1, Y1):
    lambdamax=-1
    moyenTauxBCMAX=-1
    Lambda= np.linspace(0, 1, 100 )
    X= np.concatenate((X0, Y0))
    Y= np.concatenate((X1, Y1))
    for j in Lambda:
        moyenTauxBC=0
        for i in range (5):
            w, t=estimationNewModele(19, 9 , 9 , np.delete(X0, i), np.delete(Y0, i), X1, Y1, j)
            #print ("resultat: classe validation ", resulat(w, t, np.array([X0[i], Y0[i]])))
            moyenTauxBC+=taux_bonne_classificationapprentissage(w, t, X, Y)
        for i in range(5, 10):
            w, t=estimationNewModele(19, 9 , 9 ,X0, Y0, np.delete(X1, i), np.delete(Y1, i), j)
            #print ("resultat: classe validation ", resulat(w, t, np.array([X1[i], Y1[i]])))
            moyenTauxBC+=taux_bonne_classificationapprentissage(w, t, X, Y)
        moyenTauxBC= moyenTauxBC/10
        print (moyenTauxBC)
        if (moyenTauxBCMAX<moyenTauxBC):
            moyenTauxBCMAX=moyenTauxBC
            lambdamax=j
    return j
a=validationcroise(x0, y0, x1, y1)
print ("validation croise ", a)




