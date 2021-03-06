import numpy as np
import matplotlib.pyplot as plt

mu0 = [0,0]
mu1 = [3,2]
sigma=[[1 , 1/2], [1/2, 1]]

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

plt.scatter(x, y, c=colormap[categories])
plt.title("Nuage de points pour les données d\'apprentissage")
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('NuageApprentissage.png') #Sauvegarde du nuage dans un fichier
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

plt.scatter(tTest, zTest, c=colormap[categoriesTest])
plt.title("Nuage de points pour les données de test")
plt.xlabel('t')
plt.ylabel('z')
plt.savefig('NuageTest.png') #Sauvegarde du nuage dans un fichier
#plt.show() a remettre

#Definition de la methode pour calculer nos estimateurs

def estimation(n, n0, n1, X0, Y0, X1, Y1 ) :
    PI0 = n0 / n
    PI1 = n1 / n
    mutest= np.average(np.concatenate((x0, y0)), axis=1)
    print (mutest)
    MU0 = [1 / n0 * sum(X0) , 1/n0 * sum(Y0)]
    MU1 = [1 / n1 * sum(X1) , 1 / n1 * sum(Y1)]
    SIGMA= np.cov(x0, y0)
    print (SIGMA)
    #SIGMA-1 = np.linalg.inv(SIGMA)
    #print ("Matrice inversée ", SIGMA-1)
estimation(20, 10 , 10 , x0, y0, x1, y1)
