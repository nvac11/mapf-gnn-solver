import torch
from torch_geometric.loader import DataLoader

from dataset import preprocess_dir
from model import ImprovedLevel1Policy
from training import train_level1, evaluate

if __name__ == "__main__":
    #preprocess_dir("../data", "../data_processed")
    dataset = torch.load("../data_processed/dataset.pt")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = ImprovedLevel1Policy(in_dim=6, hidden=64, out_dim=5)
    train_level1(model, loader, epochs=20, lr=1e-3, device="cuda")
    evaluate(model, loader, device="cuda")
