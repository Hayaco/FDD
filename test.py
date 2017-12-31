

import numpy as np
# import matplotlib.pyplot as plt

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

# plt.scatter(x, y, c=colormap[categories])
# plt.title("Nuage de points pour les données d\'apprentissage")
# plt.xlabel('x')
# plt.ylabel('y')
# plt.savefig('NuageApprentissage.png') #Sauvegarde du nuage dans un fichier
#plt.show() a remettre

#Cas pour le test
t0 , z0 = np.random.multivariate_normal(mu0,sigma,1000).T
t1 , z1 = np.random.multivariate_normal(mu1,sigma,1000).T
tTest=np.concatenate((t0, t1))
zTest=np.concatenate((z0, z1))
print("Jeu de données d'apprentissage", '\n' , tTest,'\n', zTest, '\n')

#Création des catégories pour avoir les couleurs

categoriesTest=[]

for i in range(1000):
    categoriesTest.append(0)

for i in range(1000):
    categoriesTest.append(1)
colormap = np.array(['r', 'b'])

#Affichage du nuage de point avec les couleurs et les axes

# plt.scatter(tTest, zTest, c=colormap[categoriesTest])
# plt.title("Nuage de points pour les données de test")
# plt.xlabel('t')
# plt.ylabel('z')
# plt.savefig('NuageTest.png') #Sauvegarde du nuage dans un fichier
#plt.show() a remettre

#Definition de la methode pour calculer nos estimateurs

print ("test")
estimation(20, 10 , 10 , x0, y0, x1, y1)

def estimation(n, n0, n1, X0, Y0, X1, Y1 ) :
    PI0 = n0 / n
    PI1 = n1 / n
    XY0= np.concatenate([[x0], [y0]])
    MU0= XY0.mean()
    XY1= np.concatenate([[x1], [y1]])
    MU1= XY1.mean()
    SIGMA0= np.cov(X0, Y0)
    print (SIGMA0)
    SIGMA1= np.cov(X1, Y1)
    #SIGMA-1 = np.linalg.inv(SIGMA)
    #print ("Matrice inversée ", SIGMA-1)