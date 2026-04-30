#pas ou blié de rajouter rho*cp du mass dans le main
import numpy as np
import gmsh
import gmsh_utils
import mass
import stiffness
import physics_temp
import dirichlet

# ==============================================================================
# On va mettre toutes les données pour le moment c'est des données au pif mais on doit trouver des valeurs réalistes pour notre problème
#pas oublié de trouver des valeurs réalistes et de les prouver et de les explic-quer dans notre rapport
# ==============================================================================

# SOL mais quel sol pierre, terre, ... ?

RHO = 2500 #masse volumique du sol en kg/m^3
CP = 800 #capacité thermique massique du sol en J/kg/K
K_SOL = 2.5 #conductivité thermique du sol en W/m.K
T_INIT = 15 #température initiale du sol en °C


#ROBIN
H_COEF = 50.0 #coefficient de transfert de chaleur (h) en W/m^2.K


#FLUIDE param
ALPHA = 5.0 #coefficient de refroidissement du fluide en °C/sqrt(s)
T_START_OUT = 10.0 #temps de démarrage du refroidissement à la sortie du tube en secondes

# Param de la simulation
DT = 600 #pas de temps (10 minutes = 600 secondes)
N_STEPS = 24 * 6 #nombre de pas de temps pour une journée (24 heures * 6 pas par heure)
ORDER = 1 #ordre des éléments finis 