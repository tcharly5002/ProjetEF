# Interface avec Gmsh. C'est ici qu'on charge la géométrie (le sol avec les deux trous pour les tubes).

import gmsh
import numpy as np
import sys

def border_dofs_from_tags(l_tags, tag_to_dof):
    # Convertit les tags Gmsh (non consécutifs) en indices 0-based dans la matrice globale
    l_tags = np.asarray(l_tags, dtype=int)
    # tag_to_dof[tag] vaut -1 si le tag n'a pas de DoF associé (ex: points géométriques purs)
    valid_mask = (tag_to_dof[l_tags] != -1)
    l_dofs = tag_to_dof[l_tags[valid_mask]]
    return l_dofs


def prepare_quadrature_and_basis(elemType, order):
    """
    Returns:
      xi (flattened uvw), w (ngp), N (flattened bf), gN (flattened gbf)
    """
    rule = f"Gauss{2 * order}"
    xi, w = gmsh.model.mesh.getIntegrationPoints(elemType, rule)
    _, N, _ = gmsh.model.mesh.getBasisFunctions(elemType, xi, "Lagrange")
    _, gN, _ = gmsh.model.mesh.getBasisFunctions(elemType, xi, "GradLagrange")
    return xi, np.asarray(w, dtype=float), N, gN

def get_jacobians(elemType, xi, tag=-1):
    """
    Wrapper around gmsh.getJacobians.
    Returns (jacobians, dets, coords)
    """
    jacobians, dets, coords = gmsh.model.mesh.getJacobians(elemType, xi, tag=tag)
    return jacobians, dets, coords


def load_geometry(d_p=2.0):   ## d_p = distance entre les deux tubes, on peut la changer pour voir l'effet sur le maillage et les résultats
    gmsh.initialize()

    #param géométrie de notre sonde
    L_sol = 10.0
    R_tube = 0.5


    #taille des mailles 
    lc_tube = 0.02 # Taille de la maille pour les tubes (peut être changée selon les besoins)
    lc_sol = 0.4 # Taille de la maille pour le sol (peut être changée selon les besoins)

    #création des surfaces du sol et des tubes
    #rectangle centré en (0,0) de longueur L_sol et de largeur L_sol
    rectangle = gmsh.model.occ.addRectangle(-L_sol/2, -L_sol/2, 0, L_sol, L_sol)

    #les deux tubes
    tube1 = gmsh.model.occ.addDisk(-d_p/2, 0, 0, R_tube, R_tube)
    tube2 = gmsh.model.occ.addDisk(d_p/2, 0, 0, R_tube, R_tube)

    # en gros ici on stock dans outsurface la surface résultante de [(2, rectangle)] moins ceci : [(2, tube1), (2, tube2)] où les 2 veulent simplement dire 2D
    #puis out_map on va pas s'en servir mais c'est nécessaire de le mettre car la fonction renvoie 2 trucs
    out_surface, out_map = gmsh.model.occ.cut([(2, rectangle)], [(2, tube1), (2, tube2)])
    gmsh.model.occ.synchronize() #synchronisation pour que les changements soient pris en compte
   

    # création des frontières
    curves = gmsh.model.getEntities(1)  # Récupérer les courbes (lignes)
    boundary_in = []
    boundary_out = []
    boundary_ext = []

    tolerance = 0.1
    tube_threshold = 0.01

    for dim, tag in curves:
        #là je crois on récupère les centre de masse de chaque courbe (= curves -> entité de dimension 1)
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        x, y = com[0], com[1]

        #vérif si c'est la frontière ext
        if abs(x) > L_sol/2 - tolerance or abs(y) > L_sol/2 - tolerance:
            boundary_ext.append(tag)

        #vérif quel tube est quel tube
        elif x < -tube_threshold:
            boundary_in.append(tag)
        elif x > tube_threshold:
            boundary_out.append(tag)

        #si c'est pas ça, c'est qu'on a pas classé la courbe
        else:
            print(f"attention !!! Courbe {tag} non classée: ({x:.3f}, {y:.3f})")

    #si on veut des infos sur les frontières, on peut les afficher
    #print(f"✓ Frontière ext: {len(boundary_ext)} courbes")
    #print(f"✓ Tube in: {len(boundary_in)} courbes")
    #print(f"✓ Tube out: {len(boundary_out)} courbes")

    #création des groupes physiques pour le code FEM
    gmsh.model.addPhysicalGroup(1, boundary_ext, name="boundary_ext")
    gmsh.model.addPhysicalGroup(1, boundary_in, name="boundary_in")
    gmsh.model.addPhysicalGroup(1, boundary_out, name="boundary_out")

    #pour le sol on peut faire pareil
    gmsh.model.addPhysicalGroup(2, [out_surface[0][1]], name="sol")



    #rien compris en desous


    #maillage
    # Gmsh utilise des "champs" pour contrôler la taille des mailles.
    # On crée deux champs qui travaillent ensemble : Distance puis Threshold.
    field = gmsh.model.mesh.field

    # Champ 1 : Distance
    # Calcule, pour chaque point du domaine, sa distance aux courbes des tubes.
    dist = field.add("Distance")
    field.setNumbers(dist, "CurvesList", boundary_in + boundary_out)

    # Champ 2 : Threshold (seuil)
    # Utilise les distances calculées par le champ Distance pour fixer la taille de maille :
    # - si distance < DistMin (0.1) → taille = SizeMin (lc_tube, maille fine)
    # - si distance > DistMax (1.0) → taille = SizeMax (lc_sol, maille grossière)
    # - entre les deux → interpolation linéaire
    thresh = field.add("Threshold")
    field.setNumber(thresh, "InField", dist)   # on branche le champ Distance en entrée
    field.setNumber(thresh, "SizeMin", lc_tube)
    field.setNumber(thresh, "SizeMax", lc_sol)
    field.setNumber(thresh, "DistMin", 0.1)
    field.setNumber(thresh, "DistMax", 1.0)

    # On dit à Gmsh d'utiliser ce champ comme référence pour générer le maillage
    field.setAsBackgroundMesh(thresh)

    gmsh.model.mesh.generate(2)
    #gmsh.write("sonde.msh") #si on veut sauvegarder le maillage

    #Ectraction des données du maillage pour le reste du projet
    
    ## IMPORTANT!
    elemType2D = gmsh.model.mesh.getElementType("triangle", 1)
    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    elemTags, elemNodeTags = gmsh.model.mesh.getElementsByType(elemType2D)

    boundaries = {
        "ext": gmsh.model.mesh.getNodesForPhysicalGroup(1, gmsh.model.getEntitiesForPhysicalName("boundary_ext")[0][1])[0],
        "in": gmsh.model.mesh.getNodesForPhysicalGroup(1, gmsh.model.getEntitiesForPhysicalName("boundary_in")[0][1])[0],
        "out": gmsh.model.mesh.getNodesForPhysicalGroup(1, gmsh.model.getEntitiesForPhysicalName("boundary_out")[0][1])[0]
    }


### juste avoir le visu du maillage dans gmsh
    if 'visu' in sys.argv:
        gmsh.fltk.run()

    # === ATTENTION ===
    # J'ai retiré gmsh.finalize() car si on ferme ici, les données nodeTags, etc. 
    # disparaissent de la mémoire avant que le main.py puisse les lire.
    # On fermera Gmsh à la toute fin du projet.
    # ==================pas oublié de faire ça dans le main===
    
    return elemType2D, nodeTags, nodeCoords, elemTags, elemNodeTags, boundaries

if __name__ == "__main__":
    load_geometry()
    gmsh.finalize() #ici on peut le mettre parce que c'est juste un test pour voir la géométrie, pas besoin de garder les données en mémoire après.
    



### POUR LANCER LE MAILLAGE, FAIRE LA COMMANDE DANS LE TERMINAL :
# python gmsh_utils.py visu