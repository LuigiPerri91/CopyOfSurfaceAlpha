"""
Deep time-series baselines trained on the returns stream only.
No surface, no vol-history, no market state.
These isolate the marginal contribution of the ViT surface encoder.
"""

import torch 
import torch.nn as nn

class LSTMBaseline(nn.Module):
    """2-layer LSTM -> last hidden -> scalar log(RV) forecast"""

    def __init__(self, input_dim: int = 8, hidden_dim: int =64, num_layers: int = 2, drouput: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, drouput=drouput if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_dim,1)

    def forward(self, returns: torch.Tensor) -> torch.Tensor:
        # returns: (batch, L, F_ret)
        _, (h_n, _) = self.lstm(returns)
        return self.head(h_n[-1]).squeeze(-1) # (batch,)
    
class GRUBaseline(nn.Module):
    """2-layer bidirectional GRU -> last hidden -> scalar log(RV) forecast."""

    def __init__(self, input_dim: int = 8, hidden_dim: int =64, num_layers: int = 2, drouput: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, drouput=drouput if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_dim * 2,1)

    def forward(self, returns: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(returns)
        fwd = h_n[-2]
        bwd = h_n[-1]
        return self.head(torch.cat([fwd,bwd], dim=-1)).squeeze(-1) # (batch,)

class _TCNBlock(nn.Module):
    """Dilated casual residual block"""

    def __init__(self, in_ch: int, out_ch:int, kernel_size:int, dilation:int , dropout:float = 0.2):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.utils.parametrize(nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=self.padding))
        self.conv2 = nn.utils.parametrize(nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation, padding=self.padding))
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.conv1(x)[:,:,:-self.padding])
        out = self.dropout(out)
        out = self.act(self.conv2(out)[:,:,:-self.padding])
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.act(out + res)

class TCNBaseline:
    """
    Dilated causal TCN -> global average pool -> scalar log(RV) forecast.
    """
    def __init__(self, input_dim: int = 8, num_channels: list[int] | None = None, kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        if num_channels is None:
            num_channels = [64, 64, 64]
        layers, in_ch = [], input_dim
        for i, out_ch in enumerate(num_channels):
            layers.append(_TCNBlock(in_ch, out_ch, kernel_size, dilation=2**i, dropout=dropout))
            in_ch = out_ch
        self.network = nn.Sequential(*layers)
        self.head = nn.Linear(num_channels[-1],1)

    def forward(self, returns: torch.Tensor) -> torch.Tensor:
        x = returns.permute(0,2,1) # (batch, F_ret, L)
        x = self.network(x).mean(dim=-1)
        return self.head(x).squeeze(-1)



