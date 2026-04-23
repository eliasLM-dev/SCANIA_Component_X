import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------------------------------------------------------------------
# --------------------------------------- LSTM Model ---------------------------------------
# -----------------------------------------------------------------------------------------
class LSTMModel(nn.Module):
    """
    LSTM-based binary classifier for sequential vehicle data.

    Args:
        input_size (int): Number of input features per timestep.
        hidden_size (int): Number of LSTM hidden units.
        num_layers (int): Number of stacked LSTM layers.
        dropout (float): Dropout rate. Only applied between LSTM layers if num_layers > 1.
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out


# -----------------------------------------------------------------------------------------
# --------------------------------------- TCN Model ---------------------------------------
# -----------------------------------------------------------------------------------------
class TemporalBlock(nn.Module):
    """
    Single dilated causal convolutional block with residual connection.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Convolutional kernel size.
        dilation (int): Dilation factor for the convolution.
        dropout (float): Dropout rate.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()

        # 1. Convultional 1D Layer
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation
        )

        # 2. Batch Normalisation
        self.bn = nn.BatchNorm1d(out_channels)

        # 3. ReLU Activation
        self.relu = nn.ReLU()

        # 4. Dropout
        self.dropout = nn.Dropout(dropout)

        # 5. Residual Computation
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual = None

    def forward(self, x):
        out = self.conv1d(x)
        out = out[:, :, :x.size(2)]
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out + (x if self.residual is None else self.residual(x))


class TCNModel(nn.Module):
    """
    Temporal Convolutional Network for binary classification of sequential vehicle data.
    Stacks TemporalBlocks with exponentially increasing dilation to capture long-range dependencies.

    Args:
        input_size (int): Number of input features per timestep.
        num_channels (int): Number of channels in each TemporalBlock.
        num_layers (int): Number of stacked TemporalBlocks.
        kernel_size (int): Convolutional kernel size.
        dropout (float): Dropout rate.
    """
    def __init__(self, input_size, num_channels, num_layers, kernel_size, dropout):
        super().__init__()

        # 1. Create the temporalBlocks
        blocks = []
        for i in range(num_layers):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else num_channels
            blocks.append(TemporalBlock(
                in_channels=in_channels,
                out_channels=num_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout
            ))

        self.network = nn.Sequential(*blocks)

        # 2. Linear Layer
        self.fc = nn.Linear(num_channels, 1)

    def forward(self, x):
        # 1. Transpose Input
        X = x.transpose(1, 2)

        # 3. Use the last timestep
        out = self.network(X)[:, :, -1]

        # 4. Linear functuion
        out = self.fc(out)

        return out


# -----------------------------------------------------------------------------------------
# --------------------------------------- TapNet Encoder ----------------------------------
# -----------------------------------------------------------------------------------------
class TapNetEncoder(nn.Module):
    """
    TapNet-style encoder (Zhang et al., 2020): LSTM + Conv1d → low-dimensional embedding.
    Used for dimensionality reduction before SMOTE, following Lindgren & Steinert (2022).
    Zhang, J., Xie, Y., & Qian, Y. (2020). TapNet: Multivariate Time Series Classification with Attentional Prototypical Networks. Proceedings of the AAAI Conference on Artificial Intelligence, 34(04), 6674–6681. https://ojs.aaai.org/index.php/AAAI/article/view/6017
    Lindgren, M., & Steinert, M. (2022). SMOTE for Time Series: Synthetic Minority Oversampling Technique for Multivariate Time Series Classification. Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 353–361. https://doi.org/10.1145/3534678.3539277

    Args:
        input_size (int): Number of input features per timestep.
        embed_dim (int): Dimensionality of the output embedding. Default 64.
    """
    def __init__(self, input_size, embed_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 128, batch_first=True)
        self.conv1 = nn.Conv1d(128, 256, kernel_size=8, stride=1, padding=4)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu = nn.LeakyReLU()
        self.fc = nn.Linear(128, embed_dim)
        self.classifier = nn.Linear(embed_dim, 1)

    def get_embedding(self, x):
        """Returns the low-dimensional embedding for input x, without classification head."""
        out, _ = self.lstm(x)
        out = out.transpose(1, 2)
        out = self.relu(self.bn1(self.conv1(out)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        out = out.mean(dim=2)
        return self.fc(out)

    def forward(self, x):
        return self.classifier(self.get_embedding(x))