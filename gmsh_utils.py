# Interface avec Gmsh. C'est ici qu'on charge la géométrie (le sol avec les deux trous pour les tubes).

import gmsh
import numpy as np
import sys

def load_geometry(filename = "sonde.msh"):
    gmsh.initialize()
    gmsh.open(filename)

    #param géométrie de notre sonde
    L_sol = 10.0 # Longueur du sol (peut être changée selon les besoins)
    R_tube = 0.5 # Rayon des tubes (peut être changé selon les besoins)
    d_p = 2.0 # Distance entre les centres des tubes (peut être changée selon les besoins)


    #taille des mailles 
    lc_tube = 0.005 # Taille de la maille pour les tubes (peut être changée selon les besoins)
    lc_sol = 0.5 # Taille de la maille pour le sol (peut être changée selon les besoins)

    #création des surfaces du sol et des tubes
    #rectangle centré en (0,0) de longueur L_sol et de largeur L_sol
    rectangle = gmsh.model.occ.addRectangle(-L_sol/2, -L_sol/2, 0, 2*L_sol, L_sol)

    #les deux tubes
    tube1 = gmsh.model.occ.addDisk(-d_p/2, 0, 0, R_tube, R_tube)
    tube2 = gmsh.model.occ.addDisk(d_p/2, 0, 0, R_tube, R_tube)

    #chat -> soustraction booléene : sol = rectangle - tube1 - tube2
    out_surface, out_map = gmsh.model.occ.cut([(2, rectangle)], [(2, tube1), (2, tube2)])
    # -> out_surface, out_map je ne comprends pas bien.

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

        #vérif si c'est les tubes
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
    #on force une taille de maille plus fine autour des tubes
    gmsh.model.occ.synchronize()
    points = gmsh.model.getEntities(0)  # Récupérer les points
    tube_centers = [(-d_p / 2, 0.0), (d_p / 2, 0.0)]
    tube_influence_radius = R_tube + 0.1

    for dim, tag in points:
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        x, y = com[0], com[1]

        if any((x - cx) ** 2 + (y - cy) ** 2 <= tube_influence_radius ** 2 for cx, cy in tube_centers):
            gmsh.model.mesh.setSize([(dim, tag)], lc_tube)
        else:
            gmsh.model.mesh.setSize([(dim, tag)], lc_sol)

    gmsh.model.mesh.generate(2)
    #gmsh.write("sonde.msh") #si on veut sauvegarder le maillage

    if 'close' not in sys.argv:
        gmsh.fltk.run()

    gmsh.finalize()

if __name__ == "__main__":
    generate_geothermal_mesh()