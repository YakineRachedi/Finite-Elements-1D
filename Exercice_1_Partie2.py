import numpy as np
import matplotlib.pyplot as plt

def derivate_u(x):
    return np.pi*np.cos(np.pi*x)

def u(x):
    return np.sin(np.pi * x)

def calculate_l2_norm_derivative(a,b,N):    

    h_global = (b-a) / N
    h_local = h_global / N
    Integral = 0
    x_global = a
    for i in range(N):
        x_local = x_global
        j = 0 
        while j < N:
            x_next = x_global + h_global
            der_interpolate = (u(x_next) - u(x_global)) / (h_global)
            Integral += ( derivate_u(x_local) - der_interpolate) ** 2  
            x_local = x_local + h_local
            j+=1
        x_global = (i+1) * h_global
    NormeL2 = np.sqrt(h_local * Integral)   
    return NormeL2



# Générer des valeurs de N
n_values = np.arange(5, 100, 5)

# Calculer les erreurs de la dérivée en norme L2 pour différentes valeurs de N
errors_h = np.array([calculate_l2_norm_derivative(0, 1, n) for n in n_values])

# Afficher la courbe d'erreur de la dérivée en norme L2
plt.plot(n_values, errors_h, 'r', label='Courbe de l\'erreur de la dérivée en norme L2')
plt.ylabel('Erreurs')
plt.title("Courbe d'erreur de la dérivée en norme L^2")
plt.grid(True)
plt.legend()
plt.savefig('Courbe_dervRectangles_EX1.png')
plt.show()

# Afficher la courbe d'erreur de la dérivée en norme L2 en échelle logarithmique
plt.loglog(n_values, errors_h, 'b', label='Courbe de l\'erreur de la dérivée en norme L2 en échelle')
plt.ylabel('Erreurs')
plt.title("Courbe d'erreur de la dérivée en norme L^2")
plt.grid(True)
plt.legend()
plt.savefig('Courbe_dervRectanglesEchelle_EX1.png')
plt.show()

# Calculer la pente dans l'échelle logarithmique
p = np.polyfit(np.log(n_values), np.log(errors_h), 1)
p_value = p[0]
print('La valeur de P est :', p_value)