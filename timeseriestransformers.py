import torch
import torch.nn as nn
import torch.optim as optim
from math import log
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TimeSeriesTransformer(nn.Module):
    def __init__(self, num_features, d_model, num_heads, num_layers, pred_len, output_dim, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.src_embedding = nn.Linear(num_features, d_model)
        self.tgt_embedding = nn.Linear(output_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, src, tgt):
        src = self.src_embedding(src)
        tgt = self.tgt_embedding(tgt)
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)
        output = self.transformer(src, tgt)
        return self.fc_out(output)