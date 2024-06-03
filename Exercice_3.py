import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

(epsilon,lamda) = (1,0.5)
(a,b) = (0,4)
(alpha,beta) = (8,0)

def integral1(x,i):
    return (4 + 2 * (4-x) * np.exp(x/2)) * ((x-I[i-1]) / h)

def integral2(x,i):
    return (4 + 2 * (4-x) * np.exp(x/2)) * ((I[i+1]-x) / h)

def calcul_second_membre(N):
    # Calcul de l'intégrale par la fonction quad de la bib Scpay
    vecteur_second_membre = np.zeros(N)
    Tab_1 = np.zeros(N)
    Tab_2 = np.zeros(N)
    for i in range(1,N-1):
        resultIntegral_1 = quad(integral1,I[i-1],I[i],args=(i))
        print('resultat_integral1',resultIntegral_1)
        res_1 = resultIntegral_1[0] + resultIntegral_1[1]
        Tab_1[i] = res_1   
    for i in range(1,N):
        resultIntegral_2 = quad(integral2,I[i],I[i+1],args=(i))
        res_2 = resultIntegral_2[0] + resultIntegral_2[1]
        Tab_2[i] = res_2
    vecteur_second_membre = Tab_1 + Tab_2    
    return vecteur_second_membre
    
def EF(N):
    A = np.zeros((N,N))
    A = (2 * (epsilon / h)) * np.diag(np.ones(N),0) + ((-epsilon / h) + (lamda / 2)) * np.diag(np.ones(N-1),1) - ((epsilon / h) + (lamda / 2)) * np.diag(np.ones(N-1),-1)
    Vecteur_forme_linéaire = Fh + (lamda * h) *((alpha - beta) / (b - a))
    U = np.linalg.solve(A,Vecteur_forme_linéaire) # Solution du probleme de relevement
    U=np.append([0],U)
    U=np.append(U,[0])
    U = U + alpha * ((b-I)/(b-a)) + beta * ((I-a)/(b-a)) #Solution du probleme avec la conditon non homogene
    return U
n = [100 ,200 ,500, 1000]
plt.figure(figsize=(8, 6))
for n in n:
    I = np.linspace(a ,b ,n+2)
    h = (b-a) / (n + 1)
    Fh = calcul_second_membre(n)
    print('Fh ==>',Fh)
    U = EF(n)
    plt.plot(I,U,label=f'Courbe pour la valeur de n = {n}',markevery=20)
plt.title('Solution du problème aux limites cas général')
plt.ylabel('Solution approchée pour différentes valeurs de n')
plt.legend()
plt.grid(True)
plt.savefig('Courbe_EX3.png')
plt.show()