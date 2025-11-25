import torch
from torch import nn
from torch_geometric.loader import DataLoader


def train_level1(model, loader: DataLoader, epochs: int = 10, lr: float = 1e-3, device: str = "cuda"):
    """
    Basic supervised training loop for Level-1 policy.

    Args:
        model: ImprovedLevel1Policy
        loader: PyG DataLoader over list[Data]
        epochs: number of epochs
        lr: learning rate
        device: "cuda" or "cpu"
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in loader:
            batch = batch.to(device)

            # Forward pass (note: goal_mask is required)
            logits = model(
                batch.x,
                batch.edge_index,
                batch.ego_mask,
                batch.goal_mask,
            )

            loss = criterion(logits, batch.y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss)

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch:02d} | loss={avg_loss:.4f}")


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: str = "cuda") -> float:
    """
    Simple accuracy evaluation.

    Returns:
        accuracy in [0,1]
    """
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0

    for batch in loader:
        batch = batch.to(device)

        logits = model(
            batch.x,
            batch.edge_index,
            batch.ego_mask,
            batch.goal_mask,
        )

        pred = logits.argmax(dim=1)  # [B]
        correct += (pred == batch.y).sum().item()
        total += batch.y.size(0)

    acc = correct / total if total > 0 else 0.0
    print(f"Accuracy: {acc * 100:.2f}%")
    return acc
