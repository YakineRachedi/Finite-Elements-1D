import numpy as np
import matplotlib.pyplot as plt

# Nombre des noueds
n_valeurs = [50, 80, 100, 250, 500, 1000]

e = 0.01
lamda = 1

def EF(N, epsilon):
    x = np.sqrt(np.linspace(0, 1, N + 2))
    A = np.zeros((N, N))
    F = np.zeros(N)
    h = x[1:] - x[:-1] #vecteur du pas
    for i in range(N):
        A[i, i] = epsilon * 1 / h[i]
        F[i] = h[i]
        if i == N - 1:
            A[i, i] = A[i, i] + epsilon * 1 / h[N] #remplissage de la case A_n_n
            break
        A[i, i] = A[i, i] + epsilon * 1 / h[i + 1]
        A[i, i + 1] = - epsilon * (1 / h[i + 1]) + (lamda / 2)
        A[i + 1, i] = - epsilon * (1 / h[i]) - (lamda / 2)

    U = np.linalg.solve(A, F)
    U = np.append([0], U)
    U = np.append(U, [0])
    return x, U

# Tracer les courbes pour différentes valeurs de n
plt.figure(figsize=(8, 6))
for n in n_valeurs:
    x, U = EF(n, e)
    plt.plot(x, U, label=f'n={n}')

plt.legend()
plt.legend(loc='upper left')
plt.grid(True)
plt.ylabel('Solution approchée')
plt.title('Courbes pour différentes valeurs de n cas non uniforme')
plt.savefig('Courbe_EX2_P2.png')
plt.show()
