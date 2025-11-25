import pytest
from src.graph import pos_to_idx

def test_pos_to_idx():
    assert pos_to_idx(0, 0, 5) == 0
    assert pos_to_idx(1, 0, 5) == 1
    assert pos_to_idx(0, 1, 5) == 5
    assert pos_to_idx(4, 3, 5) == 19


from src.graph import build_edges

def test_build_edges_shape():
    H, W = 3, 3
    edge_index = build_edges(H, W)
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] > 0
from src.graph import action_label

def test_action_label():
    assert action_label([0,0],[0,1]) == 0  # down
    assert action_label([0,1],[0,0]) == 1  # up
    assert action_label([0,0],[1,0]) == 2  # right
    assert action_label([1,0],[0,0]) == 3  # left
    assert action_label([2,2],[2,2]) == 4  # stay
from src.graph import instance_to_data

def test_instance_to_data():
    inst = {
        "height": 3, "width": 3,
        "obstacles": [],
        "agents": [{"id":0, "start":[0,0], "goal":[2,2]}],
        "paths": {"0": [[0,0],[1,0],[2,0],[2,1],[2,2]]},
    }

    data_list = instance_to_data(inst)
    assert len(data_list) == 5
    d = data_list[0]
    assert d.x.shape == (9, 4)
    assert d.edge_index.shape[0] == 2
