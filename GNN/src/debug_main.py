import torch
from torch_geometric.loader import DataLoader

from dataset import *
from model import ImprovedLevel1Policy


def idx_to_xy(idx: int, W: int):
    return (idx % W, idx // W)


def debug_one_batch(dataset):
    print("\n=== Loading Dataset ===")
    print(f"Total graphs: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))

    print("\n=== Batch Dump ===")
    print(batch)
    print("x:", batch.x.shape)
    print("edge_index:", batch.edge_index.shape)
    print("y:", batch.y)

    print("\nbatch.ptr:", batch.ptr)
    return batch


def debug_graph(batch, graph_id: int):
    print(f"\n=========== Graph {graph_id} ===========")

    start = batch.ptr[graph_id].item()
    end   = batch.ptr[graph_id + 1].item()

    H = batch.H[graph_id].item()
    W = batch.W[graph_id].item()
    N = H * W

    print(f"Dimensions: H={H}, W={W}, Nodes={N}")
    print(f"Slice: start={start}, end={end}")

    xg = batch.x[start:end]
    eg = batch.ego_mask[start:end]
    gg = batch.goal_mask[start:end]

    ego_local = eg.nonzero(as_tuple=True)[0]
    goal_local = gg.nonzero(as_tuple=True)[0]

    print("Ego local index:", ego_local.tolist())
    print("Goal local index:", goal_local.tolist())
    print("Ego coords:", [idx_to_xy(i.item(), W) for i in ego_local])
    print("Goal coords:", [idx_to_xy(i.item(), W) for i in goal_local])

    others = (xg[:, 3] == 1).nonzero(as_tuple=True)[0]
    print("Other agents indices:", others.tolist())
    print("Other coords:", [idx_to_xy(i.item(), W) for i in others])

    obstacles = (xg[:, 0] == 1).nonzero(as_tuple=True)[0]
    print("Obstacles:", obstacles.tolist())


def debug_forward(batch):
    print("\n=== Forward Pass ===")
    model = ImprovedLevel1Policy()
    model.eval()

    logits = model(
        batch.x,
        batch.edge_index,
        batch.ego_mask,
        batch.goal_mask,
    )
    pred = logits.argmax(dim=1)

    print("Pred:", pred.tolist())
    print("GT:", batch.y.tolist())

    acc = (pred == batch.y).float().mean().item()
    print(f"Batch accuracy: {acc * 100:.2f}%")


if __name__ == "__main__":
    print("\n===== DEBUG MODE START =====")

    dataset = torch.load("../data_processed/dataset.pt")

    batch = debug_one_batch(dataset)
    for gid in range(batch.num_graphs):
        debug_graph(batch, gid)

    debug_forward(batch)

    print("\n===== DEBUG COMPLETE =====\n")
