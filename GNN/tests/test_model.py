import torch
from src.model import PolicyNet

def test_model_forward():
    model = PolicyNet()

    x = torch.randn(9, 4)
    edge_index = torch.tensor([[0,1,2], [1,2,0]])
    ego_mask = torch.zeros(9, dtype=torch.bool)
    ego_mask[3] = True

    out = model(x, edge_index, ego_mask)
    assert out.shape == (1, 5)
