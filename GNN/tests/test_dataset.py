from src.dataset import MAPFDataset

def test_dataset_len(tmp_path):
    (tmp_path / "a.pt").touch()
    (tmp_path / "b.pt").touch()

    ds = MAPFDataset(str(tmp_path))
    assert len(ds) == 2
