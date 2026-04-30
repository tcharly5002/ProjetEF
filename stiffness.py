import numpy as np
import scipy.sparse as sp
from scipy.sparse import lil_matrix
import gmsh


def assemble_stiffness_and_robin (elemTags, conn, jac, det, xphys, w, N, gN, kappa, h_coef, boudaries, tag_to_dof, order):
    """
    Assemblage de la matrice de rigidité (k)
    1. partie conduction :
    2. partie Robin 
    """

    ne = len(elemTags) #nombre d'éléments (triangles dans le maillage)
    ngp = len(w) #nombre de points de gauss par élément
    nloc = int(len(conn) // ne) #nombre de noeuds par élément (3 pour des triangles)
    nn = int(np.max(tag_to_dof) + 1) #nombre total de

    K = lil_matrix((nn, nn), dtype=np.float64) #matrice de rigidité vide

    #===============+ partie conduction +=================
    #FORMULE : k[i,j] = ∫(k * grad(Ni) . grad(Nj)) dΩ
    #=================

    #redimensionnement des listes plates de Gmsh pour aller dedans plus facilement

    det = np.asarray(det, dtype=np.float64).reshape(ne, ngp) #det(J) pour chaque élément et chaque point de gauss
    jac = np.asarray(jac, dtype=np.float64).reshape(ne, ngp, 3, 3) #jacobien pour chaque élément et chaque point de gauss
    conn = np.asarray(conn, dtype=np.int64).reshape(ne, nloc) #listes de noeuds pour chaque élément
    gN = np.asarray(gN, dtype=np.float64).reshape(ngp, nloc, 3) #gradients de Ni aux points de gauss -> pas trop bien compris

    for e in range(ne):
        nodes = conn[e, :] #noeuds de l'élément e
        dof_indices = tag_to_dof[nodes] #traduit les "tags de Gmsh" en indices de la matrice K (1, 1, 2,...)

        for g in range(ngp): #on boucle sur les points de gauss de l'élément
            wg = w[g] #poids du point de gauss g -> ??
            detg = det[e, g] #taille du triangle (élément) à cet endroit
            invjac = np.linalg.inv(jac[e, g]) #inverse du jacobien pour ce point de gauss 

            for a in range(nloc):
                Ia = int(dof_indices[a]) #ligne dans la matrice
                gradNa = gN[g, a, :] @ invjac #gradient de Ni dans les coordonnées physiques

                for b in range(nloc):
                    Ib = int(dof_indices[b]) #indice du noeud b dans la matrice K
                    gradNb = gN[g, b, :] @ invjac #gradient de Nj dans les coordonnées physiques

                    #formule finale
                    #poids * kappa * (grad(Ni) . grad(Nj)) * det(J)
                    K[Ia, Ib] += wg * kappa * np.dot(gradNa, gradNb) * detg

    #===============+ partie Robin +=================
    #FORMULE : k[i,j] += ∫(h * Ni * Nj) dΓ
    #=================

    #on récupère les éléments de bordure pour les tubes in et out
    for tube_key in ["in", "out"]:
        #on demande à Gmsh les éléments de la ligne (1D) sur ces frontières
        #on suppose que noundary_in/out sont des physical groups de dimension 1 (ligne) dans Gmsh
        dim_bnd = 1
        try :
            #Ici on demande à Gmsh :"Donne moi tout les petits segments de droite qui forme le cercle du tube d'entrée/sortie"
            phys_tag = gmsh.model.getEntitiesForPhysicalName(f"boundary_{tube_key}")[0][1]
            elemTypesBnd, elemTagsBnd, elemNodeTagsBnd = gmsh.model.mesh.getElements(dim_bnd, phys_tag)

            eTypeBnd = elemTypesBnd[0] 
            eNodesBnd = elemNodeTagsBnd[0].reshape(-1, order + 1)

            xiBnd , wBnd = gmsh.model.mesh.getIntegrationPoints(eTypeBnd, f"Gauss{2*order}") #points de gauss pour les éléments de bordure
            _, NBnd, _ = gmsh.model.mesh.getBasisFunctions(eTypeBnd, xiBnd, "Lagrange")
            NBnd =NBnd.reshape(len(wBnd), -1)

            #xiBnd : La position des points de calcul sur le segment.
            #wBnd : Le poids (l'importance) de chaque point.
            #NBnd : La valeur des fonctions de forme sur ces points.



            jacBnd, detBnd, _ = gmsh.model.mesh.getJacobians(eTypeBnd, xiBnd, tag=phys_tag)
            detBnd = detBnd.reshape(-1, len(wBnd))

            # Assemblage de la matrice de masse de bordure (Robin)
            for i_el in range(len(eNodesBnd)):
                nodes = eNodesBnd[i_el]
                dofs = tag_to_dof[nodes]
                for g in range(len(wBnd)):
                    for a in range(len(nodes)):
                        Ia = int(dofs[a])
                        for b in range(len(nodes)):
                            Ib = int(dofs[b])
                            # Formule : h * Ni * Nj * detJ_1D
                            K[Ia, Ib] += wBnd[g] * h_coef * NBnd[g,a] * NBnd[g,b] * detBnd[i_el, g]
        except:
            print(f"Erreur ou absence de la frontière boundary_{tube_key}")




"""
Maintenant on va faire la partie de droite de notre équation c-a-d le vecteur de charge (f) 
dû à la température du fluide dans les tubes d'entrée et de sortie.
Formule : f[i] = ∫(h * T_fluid * Ni) dΓ
"""


#=============+
#celui là j'avais la flemme de tout vérif je pense qu'il est juste mais peut-être à revoir
#=============+


def assemble_rhs_robin(num_dofs, tag_to_dof, h_coef, T_f_in, T_f_out, order):

    F = np.zeros(num_dofs)
    dim_bnd = 1

    # On boucle sur les deux tubes car ils peuvent avoir des températures différentes
    fluid_temps = {"in": T_f_in, "out": T_f_out}

    for tube_key, T_fluid in fluid_temps.items():
        try:
            phys_tag = gmsh.model.getEntitiesForPhysicalName(f"boundary_{tube_key}")[0][1]
            elemTypesBnd, elemTagsBnd, elemNodeTagsBnd = gmsh.model.mesh.getElements(dim_bnd, phys_tag)
            
            eTypeBnd = elemTypesBnd[0]
            eNodesBnd = elemNodeTagsBnd[0].reshape(-1, order + 1)
            
            xiBnd, wBnd = gmsh.model.mesh.getIntegrationPoints(eTypeBnd, f"Gauss{2*order}")
            _, NBnd, _ = gmsh.model.mesh.getBasisFunctions(eTypeBnd, xiBnd, "Lagrange")
            NBnd = NBnd.reshape(len(wBnd), -1)
            
            _, detBnd, _ = gmsh.model.mesh.getJacobians(eTypeBnd, xiBnd, tag=phys_tag)
            detBnd = detBnd.reshape(-1, len(wBnd))

            # Assemblage du vecteur
            for i_el in range(len(eNodesBnd)):
                nodes = eNodesBnd[i_el]
                dofs = tag_to_dof[nodes]
                for g in range(len(wBnd)):
                    for a in range(len(nodes)):
                        Ia = int(dofs[a])
                        # Formule : h * T_fluide * Ni * detJ_1D
                        F[Ia] += wBnd[g] * h_coef * T_fluid * NBnd[g, a] * detBnd[i_el, g]
        except:
            continue 

    return F