import torch
from torch import nn
from torch_geometric.nn import SAGEConv


class ImprovedLevel1Policy(nn.Module):
    """
    Level-1 MAPF policy:

      - Input node features: 6 dims per node
           [obstacle, ego, goal, other, node_x, node_y]

      - MLP feature encoder: 6 -> hidden
      - 3× GraphSAGE layers + LayerNorm + Dropout
      - Explicit ego/goal fusion (concat embeddings)
      - Outputs logits over 5 actions:
           0: down, 1: up, 2: right, 3: left, 4: stay
    """

    def __init__(self, in_dim: int = 6, hidden: int = 64, out_dim: int = 5):
        super().__init__()

        # 1) Feature embedding
        self.embed = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )

        # 2) GNN stack
        self.convs = nn.ModuleList([
            SAGEConv(hidden, hidden),
            SAGEConv(hidden, hidden),
            SAGEConv(hidden, hidden),
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden),
            nn.LayerNorm(hidden),
            nn.LayerNorm(hidden),
        ])

        self.dropout = nn.Dropout(0.1)

        # 3) Policy head: [ego_emb, goal_emb] -> logits
        self.policy = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x, edge_index, ego_mask, goal_mask):
        """
        x:         [N, 6]
        edge_index:[2, E]
        ego_mask:  [N] bool
        goal_mask: [N] bool

        Returns:
            logits: [B, 5] where B = number of graphs in batch
        """

        # embed raw node features
        h = self.embed(x)

        # GNN layers
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index)
            h = torch.relu(h)
            h = norm(h)
            h = self.dropout(h)

        # masks select ego / goal node per graph
        ego_h = h[ego_mask]     # [B, hidden]
        goal_h = h[goal_mask]   # [B, hidden]

        fused = torch.cat([ego_h, goal_h], dim=1)  # [B, 2*hidden]
        logits = self.policy(fused)                # [B, 5]
        return logits
