import numpy as np
import matplotlib.pyplot as plt

# Paramètres du problème
e1 = 1
e2 = 0.1
e3 = 0.01
lamda = 1

def solve_problem(N, epsilon):
    # Construction de la matrice A et du vecteur F
    A = ((2 * epsilon) / h) * np.diag(np.ones(N), 0) + ((- epsilon / h) + (lamda / 2)) * np.diag(np.ones(N - 1), 1) + ((- epsilon / h) + (-lamda / 2)) * np.diag(np.ones(N - 1), -1)
    F = (h) * np.ones(N)
    
    # Résolution du système linéaire
    U = np.linalg.solve(A, F)
    
    # Ajout des conditions aux limites
    U = np.append([0], U)
    U = np.append(U, [0])

    return U

# Liste des valeurs de n à tester
n_values = [100,200, 500, 1000]

plt.figure(figsize=(8, 7))
# Boucle sur les différentes valeurs de n
for n in n_values:
    h = 1 / (n + 1)
    x = np.linspace(0, 1, n + 2)
    
    # Résolution pour chaque valeur de epsilon
    U_e1 = solve_problem(n, e1)
    U_e2 = solve_problem(n, e2)
    U_e3 = solve_problem(n, e3)
    # Tracé des courbes
    plt.plot(x, U_e1,'b',label=f'Courbe pour epsilon = {e1} et n = {n}', markevery=20)
    plt.plot(x, U_e2,'g',label=f'Courbe pour epsilon = {e2} et n = {n}', markevery=20)
    plt.plot(x, U_e3,'r',label=f'Courbe pour epsilon = {e3} et n = {n}', markevery=20)

# Paramètres du tracé global
plt.title('Solution du problème aux limites pour différentes valeurs de n')
plt.ylabel('Solution approchée')
plt.legend(loc='upper left')  # Déplacer la légende vers le coin supérieur gauche
plt.legend()
plt.grid(True)
plt.savefig('Courbe_EX2_P1.png')
plt.show()
