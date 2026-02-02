# Multi-Agent Path Finding with Graph Neural Networks (MAPF-GNN)

Ce dépôt contient une implémentation d'un solveur de recherche de chemin multi-agents (MAPF) exploitant les réseaux de neurones graphiques (GNN). L'objectif est de permettre à un ensemble d'agents d'atteindre leurs destinations respectives dans un environnement 2D tout en évitant les collisions, de manière décentralisée.

## Présentation du projet

La résolution du problème MAPF par des méthodes traditionnelles (comme CBS ou Prioritized Planning) devient coûteuse en calcul à mesure que le nombre d'agents augmente. Cette approche utilise l'apprentissage par imitation (Imitation Learning) pour entraîner un modèle GNN capable de prédire les actions optimales en se basant sur les observations locales et les communications entre agents voisins.

### Caractéristiques principales

* **Architecture décentralisée** : Les décisions sont prises localement par chaque agent.
* **Communication par graphe** : Utilisation des GNN pour modéliser les interactions entre agents à proximité.
* **Passage à l'échelle** : Capacité à gérer des scénarios avec une densité d'agents élevée là où les algorithmes classiques saturent.

## Structure du répertoire

* `models/` : Définitions des architectures PyTorch (GNN, couches de convolution, etc.).
* `data/` : Scripts de génération d'environnements et de datasets d'entraînement.
* `train.py` : Script principal pour l'entraînement du modèle.
* `test.py` : Évaluation des performances sur des cartes inédites.
* `utils/` : Fonctions utilitaires pour le traitement des graphes et la détection de collisions.

## Installation

### Prérequis

* Python 3.8+
* PyTorch
* PyTorch Geometric (pour les opérations sur graphes)

### Entraînement

Pour lancer un entraînement avec les paramètres par défaut :

```bash
python train.py --config config/default.yaml

```

### Évaluation

Pour tester le modèle entraîné sur un scénario spécifique :

```bash
python test.py --model_path checkpoints/model_best.pth --num_agents 10

```
