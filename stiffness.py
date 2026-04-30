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
            #on récupère les tags des éléments de bordure pour ce tube
            phys_tag = gmsh.model.getEntitiesForPhysicalName(f"boundary_{tube_key}")[0][1]
            elemTypesBnd, elemTagsBnd, elemNodeTagsBnd = gmsh.model.mesh.getElements(dim_bnd, phys_tag)

            eTypeBnd = elemTypesBnd[0] 
            eNodesBnd = elemNodeTagsBnd[0].reshape(-1, order + 1)
            