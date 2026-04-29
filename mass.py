# code qui va être utilisé pour calculer la masse du sol et des tubes, à partir de la géométrie créée avec gmsh_utils.py
import numpy as np
import scipy.sparse as sp
from scipy.sparse import lil_matrix


# ==============================================================================
# PASSAGE DE L'INTÉGRALE AU CODE : LA MATRICE DE MASSE (M)
# ==============================================================================
# FORMULE : M_ij = ∫_Ω (Ni * Nj) dΩ  ≈  Σ_e [ Σ_g (w_g * Ni(g) * Nj(g) * det(J)_g) ]
#
# 1. DISCRÉTISATION SPATIALE :
#    L'intégrale sur tout le domaine Ω est découpée en une somme d'intégrales 
#    sur chaque élément (triangle) 'e' du maillage : Σ_e ∫_e ...
#    + on approxime T(x,y,t) par Σ Uj(t) * Nj(x,y), où Nj sont les fonctions de forme
#    et Uj sont les valeurs aux nœuds (temporelles).
#    -> la dérivée temporelle (∂T/∂t) devient Σ (dUj/dt * Nj), et on sort la dérivée de l'intégrale.
#    la dérivée sortie sera traitée dans Dirichlet.py 
#
# 2. CHANGEMENT DE REPÈRE (Le Jacobien) :
#    Calculer une intégrale sur un triangle quelconque est complexe. On projette
#    chaque triangle réel sur un "triangle de référence".
#    - 'det(J)' (déterminant du Jacobien) : C'est le facteur de correction. 
#      Il représente le rapport de taille entre le triangle réel et le triangle 
#      de référence. Sans lui, l'ordinateur ignorerait la taille réelle des mailles.
#    
#
# 3. INTÉGRATION NUMÉRIQUE (Quadrature de Gauss) :
#    L'ordinateur ne calcule pas d'intégrales, il fait des sommes pondérées aux 
#    "points de Gauss" (g).
#    - 'w' (weights) : Ce sont les poids de chaque point de Gauss.
#    - 'N' : Les valeurs des fonctions de forme évaluées à ces points précis.
#
# 4. ASSEMBLAGE GLOBAL :
#    - 'tag_to_dof' : Permet de placer la contribution locale du triangle 'e' 
#      à la bonne "adresse" (ligne Ia, colonne Ib) dans la grande matrice globale.
#
# 5. CE QUE CE FICHIER NE CALCULE PAS :
#    - La PHYSIQUE (ρ * cp) : Multiplié dans le MAIN.
#    - Le TEMPS (∂T/∂t) : Géré dans DIRICHLET.PY (Schéma en thétâ).
# ==============================================================================


def compute_mass_matrix(elemTags, conn, det, w, N, tag_to_dof):
    #données d'entrée 
    ne = len(elemTags) #nombre d'éléments (triangles dans le maillage)
    ngp = len(w) #nombre de points de gauss par élément
    nloc = int(len(conn) // ne) #nombre de noeuds par élément (3 pour des triangles)
    nn = int(np.max(tag_to_dof) + 1) #nombre total de degrés de liberté (noeuds totaux)

    #on redimensionne les listes plates de Gmsh pour aller dedans plus facilement
    det = np.asarray(det, dtype=np.float64).reshape(ne, ngp) #det(J) pour chaque élément et chaque point de gauss
    conn = np.asarray(conn, dtype=np.int64).reshape(ne, nloc) #listes de noeuds pour chaque élément
    N = np.asarray(N, dtype=np.float64).reshape(ngp, nloc) #valeurs des Ni aux points de gauss -> pas trop bien compris

    #on vietn cr&er notre matrice vide 
    M = lil.matrix((nn, nn), dtype=np.float64)

    #on assemble la matrice de masse

    #on boucle sur chaque élément du maillage (triangles)
    for e in range(ne):
        element_tags = conn[e, :] #noeuds de l'élément e
        dof_indices = tag_to_dof[element_tags] #traduit les "tags de Gmsh" en indices de la matrice M (1, 1, 2,...)


        #on boucle sur les points de gauss de l'élément
        for g in range(ngp): #on boucle sur les points de gauss de l'élément
            wg = w[g] #poids du point de gauss g -> c'est quoi dans notre calcul ?
            detg = det[e, g] #taille du triangle (élément) à cet endroit

            #double boucle sur les noeuds du triangle (a et b)
            for a in range(nloc):
                Ia = int(dof_indices[a]) #ligne dans la matrice
                Na = N[g, a] #valeur de la fonction Ni au point g

                for b in range(nloc):
                    Ib = int(dof_indices[b]) #indice du noeud b dans la matrice M
                    Nb = N[g, b] #valeur de la fonction Nj au point g

                    #formule finale
                    #poids * valeur de Ni * valeur de Nj * det(J)
                    # notre rho * cp n'est pas là main on le rajoute dans le main donc ça va.
                    M[Ia, Ib] += wg * Na * N[g, b] * detg
