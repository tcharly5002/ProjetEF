import numpy as np
import gmsh
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.sparse import csr_matrix

import gmsh_utils
import mass
import stiffness
import physics_temp
import dirichlet
from stiffness import assemble_rhs_robin

#############################################################################

#BON ICI C EST ENCORE BCP L IA MAIS LE BUT C4EST DE COMPRENDRE ET MODIFIER CE QUI VA PAS
#PUIS ON REND LE TRUC UN PEU PLUS "HUMAIN" 

###############################################################################


# ==============================================================================
# PARAMÈTRES PHYSIQUES
# ==============================================================================
RHO    = 2500.0   # kg/m³
CP     = 800.0    # J/kg/K
K_SOL  = 30    # W/m/K
H_COEF = 50.0     # W/m²/K  (Robin)
ALPHA = 0.027       # descend progressivement vers ~0°C en 24h
T_START_OUT = 3600  # le tube retour démarre 1h après
T_INIT = 8.0        # température initiale du sol
T_BORD = 8.0        # même chose au bord

# Paramètres simulation
DT      = 600.0        # pas de temps en secondes (10 min)
N_STEPS = 24 * 6       # 1 journée
THETA   = 1.0          # Euler implicite (stable)
ORDER   = 1

# ==============================================================================
# 1. CHARGEMENT DU MAILLAGE
# ==============================================================================
elemType2D, nodeTags, nodeCoords, elemTags, elemNodeTags, boundaries = gmsh_utils.load_geometry()

# Construction de tag_to_dof : tag Gmsh -> indice dans la matrice
max_tag = int(np.max(nodeTags))
tag_to_dof = -np.ones(max_tag + 1, dtype=int)
for dof_idx, tag in enumerate(nodeTags):
    tag_to_dof[int(tag)] = dof_idx

nn = len(nodeTags)  # nombre total de degrés de liberté

# Coordonnées des noeuds pour le plot (x, y)
nodeCoords = np.array(nodeCoords).reshape(-1, 3)
x_nodes = nodeCoords[:, 0]
y_nodes = nodeCoords[:, 1]

# ==============================================================================
# 2. QUADRATURE ET FONCTIONS DE FORME
# ==============================================================================
xi, w, N, gN = gmsh_utils.prepare_quadrature_and_basis(elemType2D, ORDER)
jac, det, xphys = gmsh_utils.get_jacobians(elemType2D, xi)

# ==============================================================================
# 3. ASSEMBLAGE M ET K (une seule fois, hors boucle temps)
# ==============================================================================
print("Assemblage de la matrice de masse M...")
M_raw = mass.compute_mass_matrix(elemTags, elemNodeTags, det, w, N, tag_to_dof)
M = csr_matrix(RHO * CP * M_raw)

print("Assemblage de la matrice de rigidité K...")
K = csr_matrix(stiffness.assemble_stiffness_and_robin(
    elemTags, elemNodeTags, jac, det, xphys,
    w, N, gN,
    kappa=K_SOL,
    h_coef=H_COEF,
    boudaries=boundaries,
    tag_to_dof=tag_to_dof,
    order=ORDER
))

# ==============================================================================
# 4. CONDITIONS AUX LIMITES DIRICHLET (bord extérieur)
# ==============================================================================
ext_dofs = gmsh_utils.border_dofs_from_tags(boundaries["ext"], tag_to_dof)
dir_vals = np.full(len(ext_dofs), T_BORD)

# ==============================================================================
# 5. CONDITION INITIALE
# ==============================================================================
U_n = np.full(nn, T_INIT, dtype=float)
U_n[ext_dofs] = T_BORD  # cohérence avec Dirichlet

# ==============================================================================
# 6. BOUCLE EN TEMPS
# ==============================================================================
print("Début de la simulation...")

# F au temps 0
T_in_0, T_out_0 = physics_temp.get_fluid_temperature(0.0, T_INIT, ALPHA, T_START_OUT)
F_n = assemble_rhs_robin(nn, tag_to_dof, H_COEF, T_in_0, T_out_0, ORDER)

# Stockage pour visualisation
T_history = [U_n.copy()]

for step in range(1, N_STEPS + 1):
    t = step * DT
    T_in, T_out = physics_temp.get_fluid_temperature(t, T_INIT, ALPHA, T_START_OUT)

    # Assemblage F au temps n+1
    F_np1 = assemble_rhs_robin(nn, tag_to_dof, H_COEF, T_in, T_out, ORDER)

    # Pas de temps theta
    U_np1 = dirichlet.theta_step(
        M, K, F_n, F_np1, U_n,
        dt=DT, theta=THETA,
        dirichlet_dofs=ext_dofs,
        dir_vals_np1=dir_vals
    )

    U_n = U_np1
    F_n = F_np1

    if step % 12 == 0:  # affichage toutes les 2 heures
        print(f"  t = {t/3600:.1f}h  |  T_in = {T_in:.2f}°C  |  T_out = {T_out:.2f}°C"
              f"  |  T_min = {U_n.min():.2f}°C  |  T_max = {U_n.max():.2f}°C")
    T_history.append(U_n.copy())

# ==============================================================================
# 7. VISUALISATION — animation temporelle
# ==============================================================================
import matplotlib.animation as animation

conn_plot = np.array(elemNodeTags, dtype=int).reshape(-1, 3)
conn_dof = np.vectorize(lambda t: tag_to_dof[t])(conn_plot)
triang = mtri.Triangulation(x_nodes, y_nodes, conn_dof)

# Calcul des limites globales pour que la colorbar soit fixe
T_min_global = min(T.min() for T in T_history)
T_max_global = max(T.max() for T in T_history)

fig, ax = plt.subplots(figsize=(8, 7))
tpc = ax.tripcolor(triang, T_history[0], shading='gouraud', cmap='RdYlBu_r',
                   vmin=T_min_global, vmax=T_max_global)
cbar = plt.colorbar(tpc, ax=ax, label='Température (°C)')
ax.set_aspect('equal')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
title = ax.set_title('t = 0h00')

def update(frame):
    ax.cla()
    tpc = ax.tripcolor(triang, T_history[frame], shading='gouraud', cmap='RdYlBu_r',
                       vmin=T_min_global, vmax=T_max_global)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    t_h = (frame * DT) / 3600
    ax.set_title(f't = {t_h:.1f}h')
    return tpc,

ani = animation.FuncAnimation(fig, update, frames=len(T_history),
                               interval=200, blit=False)

plt.tight_layout()
gmsh.finalize()
plt.show()