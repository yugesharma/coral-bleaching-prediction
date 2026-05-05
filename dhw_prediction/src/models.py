# src/models.py

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        """
        input_dim:   number of tabular features (12)
        """
        super().__init__()

        layers = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(1)


class AttentionPool(nn.Module):
    def __init__(self, input_dim, attention_hidden_dim):
        """
        input_dim: number of channels coming out of conv layers
        """
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(attention_hidden_dim, 1)
        )

    def forward(self, x):
        """
        x shape: (batch, channels, depth)
        returns:
          context: (batch, channels) — weighted sum over depth
          weights: (batch, depth)    — attention weights, for visualization
        """
        x_t = x.permute(0, 2, 1)

        scores = self.attention(x_t)          #(batch, depth, 1)
        weights = torch.softmax(scores, dim=1) #(batch, depth, 1)

        context = (weights * x_t).sum(dim=1)  #(batch, channels)

        return context, weights.squeeze(-1)    #weights: (batch, depth)


class CNN1DWithAttention(nn.Module):
    def __init__(self, n_depths, n_channels, n_filters, kernel_size,
                 n_conv_layers, attention_hidden_dim, tab_input_dim,
                 dropout):
   
        super().__init__()

        conv_layers = []
        in_channels = n_channels

        for _ in range(n_conv_layers):
            conv_layers.append(
                nn.Conv1d(in_channels, n_filters, kernel_size,
                          padding=kernel_size // 2)
            )
            conv_layers.append(nn.BatchNorm1d(n_filters))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.Dropout(dropout))
            in_channels = n_filters

        self.conv = nn.Sequential(*conv_layers)

        self.attention_pool = AttentionPool(n_filters, attention_hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(n_filters + tab_input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x_seq, x_tab):
        """
        x_seq: (batch, n_channels, n_depths)
        x_tab: (batch, tab_input_dim)
        """
        h = self.conv(x_seq)                        # (batch, n_filters, n_depths)

        context, attn_weights = self.attention_pool(h)  # (batch, n_filters)

        # Combine with tabular features
        combined = torch.cat([context, x_tab], dim=1)   # (batch, n_filters + 12)

        out = self.classifier(combined)                  # (batch, 1)

        return out.squeeze(1), attn_weights
