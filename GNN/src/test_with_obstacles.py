import torch
from dataset import pos_to_idx, idx_to_pos, build_edges
from model import ImprovedLevel1Policy

def test_exact_scenario():
    """
    Reproduit EXACTEMENT les deux situations problématiques de votre simulation.
    """
    model = ImprovedLevel1Policy(hidden=128)
    model.load_state_dict(torch.load("level1_policy.pth"))
    model.eval()
    
    # Votre maze exact
    maze = [
        [0,0,0,0,1],  # y=0
        [1,1,0,1,0],  # y=1
        [0,0,0,0,0],  # y=2
        [0,1,1,1,0],  # y=3
        [0,0,0,0,0]   # y=4
    ]
    H, W = 5, 5
    n = H * W
    
    print("\n" + "="*70)
    print("TEST DES SITUATIONS PROBLÉMATIQUES")
    print("="*70)
    
    # Les deux cas qui bloquent
    test_cases = [
        # (ego, goal, maze, description)
        ((4, 3), (4, 4), "Agent 0: devrait aller DOWN"),
        ((0, 4), (0, 3), "Agent 1: devrait aller UP"),
    ]
    
    for ego, goal, description in test_cases:
        print(f"\n{'='*70}")
        print(description)
        print(f"Ego: {ego} → Goal: {goal}")
        print(f"{'='*70}")
        
        # Construire le graphe AVEC obstacles
        x = torch.zeros((n, 6), dtype=torch.float)
        
        # Obstacles
        for y in range(H):
            for x_pos in range(W):
                if maze[y][x_pos] == 1:
                    idx = pos_to_idx(x_pos, y, W, H)
                    x[idx, 0] = 1.0
        
        # Ego
        ego_i = pos_to_idx(ego[0], ego[1], W, H)
        x[ego_i, 1] = 1.0
        
        # Goal
        goal_i = pos_to_idx(goal[0], goal[1], W, H)
        x[goal_i, 2] = 1.0
        
        # Coordonnées
        for idx in range(n):
            pos = idx_to_pos(idx, W, H)
            x[idx, 4:6] = torch.tensor([pos.x, pos.y], dtype=torch.float)
        
        # Masks
        ego_mask = torch.zeros(n, dtype=torch.bool)
        ego_mask[ego_i] = True
        
        goal_mask = torch.zeros(n, dtype=torch.bool)
        goal_mask[goal_i] = True
        
        edge_index = build_edges(H, W)
        
        # Prédiction
        with torch.no_grad():
            logits = model(x, edge_index, ego_mask, goal_mask)
            pred_action = logits.argmax(dim=1).item()
        
        action_names = ["DOWN", "UP", "RIGHT", "LEFT", "STAY"]
        
        # Déterminer l'action attendue
        dx = goal[0] - ego[0]
        dy = goal[1] - ego[1]
        
        if dx > 0:
            expected_action = 2  # RIGHT
        elif dx < 0:
            expected_action = 3  # LEFT
        elif dy > 0:
            expected_action = 0  # DOWN
        elif dy < 0:
            expected_action = 1  # UP
        else:
            expected_action = 4  # STAY
        
        print(f"\nAction attendue: {expected_action} ({action_names[expected_action]})")
        print(f"Action prédite:  {pred_action} ({action_names[pred_action]})")
        print(f"\nLogits: {logits.squeeze().tolist()}")
        
        # Analyser le graphe
        print(f"\n--- Analyse du graphe ---")
        print(f"Ego index: {ego_i}, coords dans graphe: {x[ego_i, 4:6].tolist()}")
        print(f"Goal index: {goal_i}, coords dans graphe: {x[goal_i, 4:6].tolist()}")
        
        # Vérifier les voisins de ego
        print(f"\nVoisins de ego (index {ego_i}):")
        for action, (dx_move, dy_move) in enumerate([(0,1), (0,-1), (1,0), (-1,0)]):
            neighbor_pos = (ego[0] + dx_move, ego[1] + dy_move)
            if 0 <= neighbor_pos[0] < W and 0 <= neighbor_pos[1] < H:
                neighbor_idx = pos_to_idx(neighbor_pos[0], neighbor_pos[1], W, H)
                is_obstacle = (x[neighbor_idx, 0] == 1)
                print(f"  {action_names[action]}: {neighbor_pos} (idx {neighbor_idx}) - Obstacle: {is_obstacle}")
        
        # Afficher les obstacles autour
        print(f"\nObstacles dans le graphe:")
        obstacle_indices = (x[:, 0] == 1).nonzero(as_tuple=True)[0]
        print(f"  Indices: {obstacle_indices.tolist()}")
        print(f"  Positions CBS:")
        for obs_idx in obstacle_indices:
            obs_pos = idx_to_pos(obs_idx.item(), W, H)
            print(f"    idx {obs_idx.item()} → ({obs_pos.x}, {obs_pos.y})")
        
        if pred_action != expected_action:
            print(f"\n❌ ERREUR: Le modèle se trompe!")
            print(f"   Il choisit {action_names[pred_action]} au lieu de {action_names[expected_action]}")
        else:
            print(f"\n✓ Le modèle prédit correctement!")


if __name__ == "__main__":
    test_exact_scenario()