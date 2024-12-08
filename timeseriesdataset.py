from torch.utils.data import Dataset, DataLoader
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, pred_len, target_features):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_features = target_features

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len

    def __getitem__(self, idx):
        seq = self.data[idx:idx + self.seq_len]
        target = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len]
        target = target[:, -len(self.target_features):]
        return torch.tensor(seq, dtype=torch.float32).to(device), torch.tensor(target, dtype=torch.float32).to(device)