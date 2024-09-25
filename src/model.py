import torch
import torch.nn as nn
from positional_encoding import PositionalEncoding

class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, batch_first=True)
        self.attention_weights = None

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, self.attention_weights = self.self_attn(src, src, src, attn_mask=src_mask,
                                                      key_padding_mask=src_key_padding_mask,
                                                      need_weights=True)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class CircuitFormer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, max_seq_length, num_classes, dropout=0.1):
        super(CircuitFormer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim).to(torch.float32)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout, max_seq_length)
        encoder_layers = [CustomTransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout) for _ in range(num_layers)]
        self.transformer_layers = nn.ModuleList(encoder_layers)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.embedding(x.unsqueeze(1))  # Add sequence dimension
        x = self.pos_encoder(x)
        
        attention_weights = []
        for layer in self.transformer_layers:
            x = layer(x)
            attention_weights.append(layer.attention_weights)
        
        x = self.layer_norm(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)
        return x, attention_weights
