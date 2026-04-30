# dirichlet.py

import numpy as np
from scipy.sparse.linalg import spsolve


def apply_dirichlet_by_reduction(A, b, dirichlet_dofs, dirichlet_values):
    dirichlet_dofs = np.asarray(dirichlet_dofs, dtype=int)
    dirichlet_values = np.asarray(dirichlet_values, dtype=float)

    n = len(b)

    mask = np.ones(n, dtype=bool)
    mask[dirichlet_dofs] = False
    free_dofs = np.nonzero(mask)[0]


# Dirichlet_doffs c'est les indices où la temperature est connu, free_doffs c'est
# les indices où on a encore rien , les 2 sont des tableaux 
#Dirichlet values c'est les valeurs connu qu'on à aux indices Diriecht_doffs
#mask c'est list boolean, Si true alors à cet indice il est libre, si false alors il est au bord (connu)

   
    A_FF = A[free_dofs, :][:, free_dofs]
    A_FD = A[free_dofs, :][:, dirichlet_dofs]
    b_F  = b[free_dofs]
    b_red = b_F - A_FD.dot(dirichlet_values)

    U_full = np.zeros(n, dtype=float)  #on va le remplir au fur et à mesure , mais on peut 
    #déjà y mettre les CL
    U_full[dirichlet_dofs] = dirichlet_values # On rempli les conditions limites

    return A_FF, b_red, free_dofs, U_full


def solve_dirichlet(A, b, dirichlet_dofs, dirichlet_values):
   
    A_red, b_red, free_dofs, U_full = apply_dirichlet_by_reduction(A,b,dirichlet_dofs,dirichlet_values)
    U_free = spsolve(A_red.tocsr(), b_red) # le toscr() est une maniere plus adaptée pour le solve 
    #Resoudre A_red U_free = b_red
    U_full[free_dofs] = U_free # la on les met de nouveau dans la matrice complete UFULL
    U_full[dirichlet_dofs] = dirichlet_values

    return U_full

    # Ici notre U contient donc que les valeurs de dirichlet [100, 0 , 20,]


def theta_step(M, K, F_n, F_np1, U_n, dt, theta, dirichlet_dofs, dir_vals_np1):
    """
    Effectue un pas de temps pour :
        M dU/dt + K U = F(t)
oke 
    avec le schéma theta :

        (M + theta dt K) U^{n+1}
        =
        (M - (1-theta) dt K) U^n
        + dt [theta F^{n+1} + (1-theta) F^n]

    Les conditions de Dirichlet sont imposées sur U^{n+1}.
    """

    # Matrice du système à résoudre au temps n+1
    A = M + theta * dt * K

    # Partie connue venant du temps précédent
    B = M - (1.0 - theta) * dt * K

    # Second membre complet
    rhs = B.dot(U_n) + dt * (theta * F_np1 + (1.0 - theta) * F_n)

    # Application des conditions de Dirichlet
    A_red, rhs_red, free_dofs, U_full = apply_dirichlet_by_reduction(A,rhs,dirichlet_dofs,dir_vals_np1)

    U_free = spsolve(A_red.tocsr(), rhs_red) # Résolution uniquement sur les noeuds libres

    # Reconstruction du vecteur complet
    U_full[free_dofs] = U_free
    U_full[dirichlet_dofs] = dir_vals_np1

    return U_full