#MON PROJET


# Simulation d'une Sonde Géothermique par Éléments Finis (FEM)

# Présentation du projet
Ce projet a été réalisé dans le cadre du cours LEPL1110 - Méthodes des éléments finis(Année 2025-2026).

L'objectif est de modéliser la diffusion de chaleur dans le sol autour d'une sonde géothermique composée de deux tubes (entrée et sortie). La simulation étudie l'évolution de la température sur 24 heures en utilisant une approche numérique par éléments finis en 2D.

# ->Caractéristiques du modèle
* Physique : Équation de la chaleur (diffusion thermique).
* Conditions aux limites : 
    * Robin : Échange convectif entre le fluide des tubes et le sol (coefficient h).
    * Dirichlet : Température constante imposée aux limites  du domaine.
* Schéma temporel : Schéma en thêta (Euler implicite utilisé pour la stabilité).
* Maillage : Généré via Gmsh, on a resserré le maillage autour des tubes pour être plus précis là où les échanges de chaleur sont les plus forts.

# Installation

1. Créer un environnement virtuel :
   python -m venv .venv

2. Activer l'environnement :
   Windows : .venv\Scripts\activate

3. Installer les dépendances :
   pip install numpy scipy matplotlib gmsh

# Utilisation

# ->Visualisation du maillage
Pour inspecter la géométrie et la finesse du maillage avant de lancer le calcul :
python gmsh_utils.py visu

# ->Lancer la simulation
Pour effectuer les calculs et afficher l'animation de l'évolution thermique :
python main.py

# Structure du code

* main.py : Script principal gérant l'initialisation, la boucle temporelle et la visualisation.
* gmsh_utils.py : Interface avec Gmsh pour la création de la géométrie et du maillage.
* mass.py : Calcul et assemblage de la matrice de masse (capacité thermique).
* stiffness.py : Assemblage de la matrice de rigidité (conduction) et des termes de bord de Robin.
* physics_temp.py : Définition de l'évolution de la température du fluide circulant dans les tubes.
* dirichlet.py : Fonctions pour l'application des conditions de Dirichlet et le calcul du pas de temps (schéma thêta).

# Modèle Mathématique

L'équation de base est l'équation de diffusion :
rho * cp * dT/dt - div(k * grad(T)) = 0

Sur les parois des tubes, la condition de Robin est appliquée :
-k * grad(T) . n = h * (T - T_fluide)

La température du fluide (T_fluide) diminue selon une loi en racine carrée du temps pour simuler l'extraction d'énergie.

# Auteurs (Groupe 54)
* Quentin GILLAIN
* Charles-Henry STÉVENART
* Simon GEORGE