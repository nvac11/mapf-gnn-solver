import torch
from dataset import pos_to_idx, idx_to_pos, build_edges
from model import ImprovedLevel1Policy
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------
# Affichage du labyrinthe + agents + goals
# -------------------------------------------------
def draw_maze(maze, positions, goals, step):
    H, W = len(maze), len(maze[0])
    grid = np.array(maze)

    plt.clf()
    plt.title(f"Étape {step}")
    plt.imshow(grid, cmap="gray_r", interpolation="nearest")

    # Affichage des goals (étoiles jaunes)
    for i, g in enumerate(goals):
        plt.scatter(g[0], g[1], c="yellow", s=200, marker="*",
                    edgecolors="black", linewidths=1.5)

    # Affichage des agents (ronds rouges avec index)
    for i, p in enumerate(positions):
        plt.scatter(p[0], p[1], c="red", s=200, marker="o")
        plt.text(p[0] - 0.15, p[1] + 0.15, f"{i}",
                 color="white", fontsize=12, weight="bold")

    plt.xticks(range(W))
    plt.yticks(range(H))
    plt.grid(color="black", linewidth=0.5)
    plt.pause(0.4)


# -------------------------------------------------
# Actions → mouvement (dx, dy) EN COORDONNÉES CBS
# CBS: (0,0) est TOP-LEFT, Y augmente vers le BAS
# -------------------------------------------------
ACTION_TO_DELTA = {
    0: (0, +1),   # down  (y augmente)
    1: (0, -1),   # up    (y diminue)
    2: (+1, 0),   # right (x augmente)
    3: (-1, 0),   # left  (x diminue)
    4: (0, 0),    # stay
}


# -------------------------------------------------
# Construction du graphe PyG pour un agent + 1 goal
# -------------------------------------------------
def build_single_graph(maze, ego, goal):
    """
    ego et goal sont en coordonnées CBS: (x, y) avec (0,0) top-left
    """
    H, W = len(maze), len(maze[0])
    n = H * W

    # --- node features ---
    x = torch.zeros((n, 6), dtype=torch.float)

    # obstacles (parcours en coordonnées array Python)
    for y in range(H):
        for x_pos in range(W):
            if maze[y][x_pos] == 1:
                # Convertir (x_pos, y) CBS vers index PyG
                idx = pos_to_idx(x_pos, y, W, H)
                x[idx, 0] = 1.0

    # ego
    ego_i = pos_to_idx(ego[0], ego[1], W, H)
    x[ego_i, 1] = 1.0

    # goal
    goal_i = pos_to_idx(goal[0], goal[1], W, H)
    x[goal_i, 2] = 1.0

    # Coordonnées : utiliser idx_to_pos pour retrouver les coords CBS
    # C'est EXACTEMENT comme dans dataset.py
    for idx in range(n):
        pos = idx_to_pos(idx, W, H)
        x[idx, 4:6] = torch.tensor([pos.x, pos.y], dtype=torch.float)

    # masks
    ego_mask = torch.zeros(n, dtype=torch.bool)
    ego_mask[ego_i] = True

    goal_mask = torch.zeros(n, dtype=torch.bool)
    goal_mask[goal_i] = True

    edge_index = build_edges(H, W)

    return x, edge_index, ego_mask, goal_mask, H, W


# -------------------------------------------------
# Multi-agents + multi-goals + collisions + Matplotlib
# -------------------------------------------------
def simulate_multi_agents_collisions_goals(model, maze, starts, goals, max_steps=50):
    """
    Multi-agents + multi-goals + gestion complète des collisions entre agents,
    avec affichage Matplotlib.

    IMPORTANT: starts et goals doivent être en coordonnées CBS (0,0 = top-left)
    
    starts : liste [(x,y), ...]
    goals  : liste [(x,y), ...] même longueur que starts
    """
    H, W = len(maze), len(maze[0])
    positions = starts[:]
    num_agents = len(positions)

    assert len(goals) == num_agents, "⚠️ Il faut un goal par agent !"

    # Vérification des positions initiales
    print("\n=== Vérification initiale ===")
    for i, (pos, goal) in enumerate(zip(positions, goals)):
        print(f"Agent {i}: start={pos}, goal={goal}")
        # Vérifier que start et goal sont valides
        if not (0 <= pos[0] < W and 0 <= pos[1] < H):
            print(f"  ⚠️ ERREUR: Position de départ hors limites!")
        elif maze[pos[1]][pos[0]] == 1:
            print(f"  ⚠️ ERREUR: Position de départ sur un obstacle!")
        
        if not (0 <= goal[0] < W and 0 <= goal[1] < H):
            print(f"  ⚠️ ERREUR: Goal hors limites!")
        elif maze[goal[1]][goal[0]] == 1:
            print(f"  ⚠️ ERREUR: Goal sur un obstacle!")

    plt.figure(figsize=(6, 6))

    for step in range(max_steps):
        # Affichage de l'état courant
        draw_maze(maze, positions, goals, step)

        print(f"\n===== ÉTAPE {step} =====")
        proposed_moves = []

        for i, pos in enumerate(positions):
            goal = goals[i]
            print(f" Agent {i} — Position : {pos}  Goal : {goal}")
            if pos == goal:
                print(f"   Agent {i} a atteint son objectif → il reste sur place.")
                proposed_moves.append(pos)
                continue
            # Construire le graphe
            x, edge_index, ego_mask, goal_mask, H, W = build_single_graph(maze, pos, goal)
            
            # DEBUG: Vérifier ce que le modèle "voit"
            ego_idx = ego_mask.nonzero(as_tuple=True)[0].item()
            goal_idx = goal_mask.nonzero(as_tuple=True)[0].item()
            ego_coords_in_graph = x[ego_idx, 4:6]
            goal_coords_in_graph = x[goal_idx, 4:6]
            
            print(f"   DEBUG: ego CBS={pos}, ego_idx={ego_idx}, coords_dans_graphe={ego_coords_in_graph.tolist()}")
            print(f"   DEBUG: goal CBS={goal}, goal_idx={goal_idx}, coords_dans_graphe={goal_coords_in_graph.tolist()}")
            
            # Prédiction du modèle
            with torch.no_grad():
                logits = model(x, edge_index, ego_mask, goal_mask)
                action = logits.argmax(dim=1).item()
                print(f"   Logits: {logits.squeeze().tolist()}")
            
            print(f"   Action choisie: {action} {['down','up','right','left','stay'][action]}")

            # Calculer nouvelle position en coordonnées CBS
            dx, dy = ACTION_TO_DELTA[action]
            new_pos = (pos[0] + dx, pos[1] + dy)

            # Vérification collision mur/obstacle
            if not (0 <= new_pos[0] < W and 0 <= new_pos[1] < H):
                print(f"  ⚠️ Agent {i} hors limites → reste en place")
                new_pos = pos
            elif maze[new_pos[1]][new_pos[0]] == 1:
                print(f"  ⚠️ Agent {i} collision obstacle à {new_pos} → reste en place")
                new_pos = pos

            proposed_moves.append(new_pos)

        # --- 2) Résolution des collisions entre agents ---
        final_positions = positions[:]

        # a) Collisions multiples (plusieurs agents vers même case)
        for i, new_pos in enumerate(proposed_moves):
            if proposed_moves.count(new_pos) > 1:
                print(f"  ❌ Collision multiple sur {new_pos} → agent {i} reste en place")
                continue
            final_positions[i] = new_pos

        # b) Cross-collisions (A ↔ B simultanément)
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                if positions[i] == final_positions[j] and positions[j] == final_positions[i]:
                    print(f"  ❌ Cross-collision entre agents {i} et {j} → tous deux restent en place")
                    final_positions[i] = positions[i]
                    final_positions[j] = positions[j]

        positions = final_positions

        # --- 3) Logs texte ---
        for i, pos in enumerate(positions):
            print(f" → Agent {i} nouvelle position : {pos}")
            if pos == goals[i]:
                print(f"   🎉 Agent {i} a atteint son objectif !")

        # Tous les agents au but ?
        all_at_goal = all(positions[i] == goals[i] for i in range(num_agents))
        print(f"\n📊 Status: Agent 0 at goal: {positions[0] == goals[0]}, Agent 1 at goal: {positions[1] == goals[1]}")
        
        if all_at_goal:
            print("\n🎯 Tous les agents ont atteint leur objectif !")
            draw_maze(maze, positions, goals, step + 1)
            break

    plt.show()
    return positions


# -------------------------------------------------
# Exemple d'utilisation
# -------------------------------------------------
if __name__ == "__main__":
    model = ImprovedLevel1Policy(hidden=128)
    model.load_state_dict(torch.load("level1_policy.pth"))
    model.eval()

    # Maze: maze[y][x] où y=0 est en haut
    maze = [
        [0,0,0,0,1],  # y=0 (haut)
        [1,1,0,1,0],  # y=1
        [0,0,0,0,0],  # y=2
        [0,1,1,1,0],  # y=3
        [0,0,0,0,0]   # y=4 (bas)
    ]
    # Dimensions: H=5, W=5

    print("\n=== Configuration du maze ===")
    print("Maze (avec indices Y):")
    for y, row in enumerate(maze):
        print(f"y={y}: {row}")

    # Positions en coordonnées CBS: (x, y) avec (0,0) = top-left
    # Agent 0: va de coin haut-gauche vers coin bas-droite
    # Agent 1: va de coin bas-gauche vers position (0,3)
    
    starts = [(0, 0), (0, 4)]  # (x, y) en CBS
    goals  = [(4, 4), (0, 3)]  # (x, y) en CBS
    
    # Vérification manuelle
    print("\n=== Vérification manuelle des positions ===")
    print(f"start[0]=(0,0) -> maze[0][0]={maze[0][0]} (devrait être 0)")
    print(f"goal[0]=(4,4) -> maze[4][4]={maze[4][4]} (devrait être 0)")
    print(f"start[1]=(0,4) -> maze[4][0]={maze[4][0]} (devrait être 0)")
    print(f"goal[1]=(0,3) -> maze[3][0]={maze[3][0]} (devrait être 0)")

    simulate_multi_agents_collisions_goals(model, maze, starts, goals)