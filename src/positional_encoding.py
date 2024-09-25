import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt

class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding for non-recurrent neural networks.

    Args:
        d_model: Embedding dimension
        dropout: Dropout value (default=0.1)
        max_len: Maximum input length (default=5000)
        use_learned: If True, use learned positional embeddings instead of sinusoidal (default=False)
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000, use_learned=False):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        if use_learned:
            self.pe = nn.Parameter(torch.randn(1, max_len, d_model))
        else:
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)
            self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def visualize_positional_encoding(pe, n_positions=100, n_dims=32):
    """
    Visualizes the positional encodings.

    Args:
        pe: PositionalEncoding module
        n_positions: Number of positions to visualize
        n_dims: Number of dimensions to visualize
    """
    encodings = pe.pe.squeeze(1)[:n_positions, :n_dims].detach().numpy()
    plt.figure(figsize=(15, 5))
    plt.imshow(encodings, cmap='viridis', aspect='auto')
    plt.title("Positional Encodings")
    plt.xlabel("Encoding dimension")
    plt.ylabel("Position")
    plt.colorbar()
    plt.savefig("positional_encodings.png")
    plt.close()

# Usage
if __name__ == "__main__":
    d_model = 512
    pe = PositionalEncoding(d_model)
    visualize_positional_encoding(pe)
    print("Positional encoding visualization saved as 'positional_encodings.png'")