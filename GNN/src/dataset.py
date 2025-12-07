from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from torch import Tensor
from torch_geometric.data import Data

from itertools import product




@dataclass(frozen=True)
class Pos:
    x: int
    y: int  # CBS uses (0,0) top-left


@dataclass(frozen=True)
class Agent:
    id: int
    start: Pos
    goal: Pos


@dataclass(frozen=True)
class Instance:
    id: int
    width: int
    height: int
    obstacles: List[Pos]
    agents: List[Agent]
    paths: Dict[int, List[Pos]]


# ================================================================
# ===============  Utility Functions ==============================
# ================================================================

def pos_to_idx(x: int, y: int, W: int, H: int) -> int:
    """
    Convert (x,y) from CBS coords (0,0 top-left) to Pyg index (0,0 bottom-left).
    """
    flipped_y = H - 1 - y
    return flipped_y * W + x


def idx_to_pos(idx: int, W: int, H: int) -> Pos:
    """
    Inverse of pos_to_idx (useful for debug).
    """
    y = idx // W
    x = idx % W
    flipped_y = H - 1 - y
    return Pos(x, flipped_y)


def build_edges(H: int, W: int) -> Tensor:
    """
    Return edge_index for the 4-connected grid with y-axis flipped.
    """
    edges = []

    for y, x in product(range(H), range(W)):
        idx = pos_to_idx(x, y, W, H)

        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            xx, yy = x + dx, y + dy
            if 0 <= xx < W and 0 <= yy < H:
                nbr = pos_to_idx(xx, yy, W, H)
                edges.append((idx, nbr))

    return torch.tensor(edges, dtype=torch.long).T.contiguous()


def action_label(p0: Pos, p1: Pos) -> int:
    """
    Compute action from CBS agent positions.
    Directions in CBS coordinate system (0,0 top-left):
        down  = (0, +1)
        up    = (0, -1)
        right = (+1, 0)
        left  = (-1, 0)
        stay  = (0, 0)
    """
    dx = p1.x - p0.x
    dy = p1.y - p0.y

    match (dx, dy):
        case (0,  1): return 0  # down
        case (0, -1): return 1  # up
        case (1,  0): return 2  # right
        case (-1, 0): return 3  # left
        case (0,  0): return 4  # stay
    raise ValueError(f"Invalid move: {p0}->{p1}")


# ================================================================
# ===============  JSON   ===============
# ================================================================

def load_instance(raw: Dict[str, Any]) -> Instance:
    obstacles = [Pos(x, y) for (x, y) in raw["obstacles"]]

    agents = [
        Agent(a["id"], Pos(*a["start"]), Pos(*a["goal"]))
        for a in raw["agents"]
    ]

    paths = {
        int(k): [Pos(*p) for p in v]
        for k, v in raw["paths"].items()
    }

    return Instance(
        id=raw["id"],
        width=raw["width"],
        height=raw["height"],
        obstacles=obstacles,
        agents=agents,
        paths=paths,
    )


# ================================================================
# ===============  INSTANCE → LIST[Data] ==========================
# ================================================================

def instance_to_data(inst: Instance, skip_goal_stay: bool = True) -> List[Data]:
    """
    Convert instance to list of PyG Data objects.
    
    Args:
        inst: Instance to convert
        skip_goal_stay: If True, skip timesteps where agent is already at goal
                       and action is STAY (reduces bias toward STAY action)
    """
    H, W = inst.height, inst.width
    n = H * W

    edge_index = build_edges(H, W)
    obs_idx = {pos_to_idx(o.x, o.y, W, H) for o in inst.obstacles}

    all_graphs: List[Data] = []

    for agent in inst.agents:
        aid = agent.id
        goal = agent.goal
        path = inst.paths[aid]

        T = len(path)

        for t in range(T):
            p0 = path[t]
            p1 = path[t+1] if t+1 < T else p0

            # FILTRAGE: Skip si agent déjà au goal et action = STAY
            if skip_goal_stay and p0 == goal and p1 == goal:
                continue

            ego_i = pos_to_idx(p0.x, p0.y, W, H)
            goal_i = pos_to_idx(goal.x, goal.y, W, H)

            # =======================================================
            #         NODE FEATURES WITH POSITIONAL ENCODING
            # =======================================================
            x = torch.zeros((n, 6), dtype=torch.float)    # channels: 4 flags + 2 coords

            # --- static features ---
            if obs_idx:
                x[list(obs_idx), 0] = 1.0

            x[ego_i, 1] = 1.0
            x[goal_i, 2] = 1.0

            # --- other agents ---
            for other in inst.agents:
                if other.id != aid:
                    op = inst.paths[other.id][min(t, len(inst.paths[other.id])-1)]
                    oi = pos_to_idx(op.x, op.y, W, H)
                    x[oi, 3] = 1.0

            # --- positional encoding (x,y coordinates) ---
            coords = torch.zeros((n, 2))
            for idx in range(n):
                pos = idx_to_pos(idx, W, H)
                coords[idx] = torch.tensor([pos.x, pos.y], dtype=torch.float)
            x[:, 4:6] = coords  # append to feature map

            # =======================================================
            #                     LABEL & MASKS
            # =======================================================
            y = torch.tensor([action_label(p0, p1)], dtype=torch.long)

            ego_mask = torch.zeros(n, dtype=torch.bool)
            ego_mask[ego_i] = True

            goal_mask = torch.zeros(n, dtype=torch.bool)
            goal_mask[goal_i] = True

            # =======================================================
            #                   Build Data object
            # =======================================================
            d = Data(
                x=x,
                edge_index=edge_index,
                y=y,
                ego_mask=ego_mask,
                goal_mask=goal_mask,
                H=torch.tensor([H]),
                W=torch.tensor([W]),
            )

            all_graphs.append(d)

    return all_graphs


# ================================================================
# ===============  PROCESS DIRECTORY =============================
# ================================================================

def preprocess_dir(source: str, destination: str, skip_goal_stay: bool = True) -> None:
    """
    Load all JSON instances, convert to Data, and save one dataset.pt.
    
    Args:
        source: Directory containing JSON files
        destination: Directory to save processed dataset
        skip_goal_stay: If True, skip examples where agent is at goal with STAY action
    """
    dst = Path(destination)
    dst.mkdir(exist_ok=True)

    all_graphs: List[Data] = []
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}  # Count actions

    for json_file in Path(source).glob("*.json"):
        raw_list = json.load(open(json_file))

        for raw in raw_list:
            inst = load_instance(raw)
            graphs = instance_to_data(inst, skip_goal_stay=skip_goal_stay)
            all_graphs.extend(graphs)
            
            # Count actions
            for g in graphs:
                action_counts[g.y.item()] += 1

    # Save dataset
    torch.save(all_graphs, dst / "dataset.pt")
    
    # Print statistics
    total = sum(action_counts.values())
    print(f"\n=== Dataset Statistics ===")
    print(f"Total graphs: {len(all_graphs)}")
    print(f"Action distribution:")
    print(f"  0 (down):  {action_counts[0]:6d} ({action_counts[0]/total*100:.1f}%)")
    print(f"  1 (up):    {action_counts[1]:6d} ({action_counts[1]/total*100:.1f}%)")
    print(f"  2 (right): {action_counts[2]:6d} ({action_counts[2]/total*100:.1f}%)")
    print(f"  3 (left):  {action_counts[3]:6d} ({action_counts[3]/total*100:.1f}%)")
    print(f"  4 (stay):  {action_counts[4]:6d} ({action_counts[4]/total*100:.1f}%)")
    print(f"\nSaved to {dst/'dataset.pt'}")