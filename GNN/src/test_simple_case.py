import torch
from dataset import pos_to_idx, idx_to_pos, build_edges
from model import ImprovedLevel1Policy

def test_simple_case():
    """
    Teste le modèle sur un cas ultra-simple : aller de (0,0) à (0,1)
    dans un maze vide 3x3.
    """
    model = ImprovedLevel1Policy(hidden=128)
    model.load_state_dict(torch.load("level1_policy.pth"))
    model.eval()
    
    # Maze 3x3 vide
    maze = [
        [0, 0, 0],  # y=0
        [0, 0, 0],  # y=1
        [0, 0, 0],  # y=2
    ]
    H, W = 3, 3
    n = H * W
    
    # Test plusieurs cas simples
    test_cases = [
        # (ego, goal, expected_action, action_name)
        ((0, 0), (0, 1), 0, "DOWN"),   # Aller vers le bas
        ((0, 1), (0, 0), 1, "UP"),     # Aller vers le haut
        ((0, 0), (1, 0), 2, "RIGHT"),  # Aller vers la droite
        ((1, 0), (0, 0), 3, "LEFT"),   # Aller vers la gauche
        ((1, 1), (1, 1), 4, "STAY"),   # Rester sur place (déjà au goal)
    ]
    
    print("\n" + "="*70)
    print("TEST DU MODÈLE SUR CAS SIMPLES")
    print("="*70)
    
    correct = 0
    total = len(test_cases)
    
    for ego, goal, expected, name in test_cases:
        # Construire le graphe
        x = torch.zeros((n, 6), dtype=torch.float)
        
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
        
        # Affichage
        action_names = ["DOWN", "UP", "RIGHT", "LEFT", "STAY"]
        status = "✓" if pred_action == expected else "✗"
        correct += (pred_action == expected)
        
        print(f"\n{status} Ego: {ego} → Goal: {goal}")
        print(f"  Attendu: {expected} ({name})")
        print(f"  Prédit:  {pred_action} ({action_names[pred_action]})")
        print(f"  Logits:  {logits.squeeze().tolist()}")
        
        if pred_action != expected:
            print(f"  ❌ ERREUR: Le modèle prédit {action_names[pred_action]} au lieu de {name}")
    
    print("\n" + "="*70)
    print(f"RÉSULTAT: {correct}/{total} corrects ({correct/total*100:.1f}%)")
    print("="*70)
    
    if correct < total:
        print("\n⚠️ Le modèle échoue sur des cas simples !")
        print("Causes possibles:")
        print("1. Problème dans les coordonnées du graphe")
        print("2. Modèle pas bien entraîné sur ce type de situations")
        print("3. Bug dans build_edges ou pos_to_idx")
        
        # Test supplémentaire : afficher les coordonnées
        print("\n=== VÉRIFICATION DES COORDONNÉES ===")
        ego = (0, 0)
        goal = (0, 1)
        ego_i = pos_to_idx(ego[0], ego[1], W, H)
        goal_i = pos_to_idx(goal[0], goal[1], W, H)
        
        print(f"Ego CBS {ego} → idx PyG {ego_i} → coords dans graphe {idx_to_pos(ego_i, W, H)}")
        print(f"Goal CBS {goal} → idx PyG {goal_i} → coords dans graphe {idx_to_pos(goal_i, W, H)}")
        
        print("\nToutes les coordonnées du graphe (idx → CBS coords):")
        for idx in range(n):
            pos = idx_to_pos(idx, W, H)
            print(f"  idx {idx} → CBS ({pos.x}, {pos.y})")


if __name__ == "__main__":
    test_simple_case()