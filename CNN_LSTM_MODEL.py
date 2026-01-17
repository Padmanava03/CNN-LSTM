import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGCNN(nn.Module):
    """
    CNN for spatial–spectral EEG feature extraction
    Input : [B, C, F, T]
    Output: [B, D, T]
    """
    def __init__(self, in_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=(3, 3),
            padding=(1, 1)
        )

        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            padding=(1, 1)
        )

        self.bn2 = nn.BatchNorm2d(64)

        # Pool only over frequency dimension
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, x):
        # x: [B, C, F, T]

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Pool over frequency → [B, 64, 1, T]
        x = self.freq_pool(x)

        # Remove frequency dim → [B, 64, T]
        x = x.squeeze(2)

        return x

class EEGLSTM(nn.Module):
    """
    LSTM for temporal depth-perception modeling
    Input : [B, T, D]
    Output: [B, H]
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, x):
        # x: [B, T, D]
        _, (h_n, _) = self.lstm(x)

        # Last layer hidden state → [B, H]
        return h_n[-1]

class DepthClassifier(nn.Module):
    """
    Final classification head
    """
    def __init__(self, input_dim, num_classes=3):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

class CNNLSTMDepthModel(nn.Module):
    """
    End-to-end CNN–LSTM model for EEG-based depth perception
    """
    def __init__(self, num_channels):
        super().__init__()

        self.cnn = EEGCNN(in_channels=num_channels)
        self.lstm = EEGLSTM(input_dim=64, hidden_dim=128)
        self.classifier = DepthClassifier(input_dim=128, num_classes=3)

    def forward(self, x):
        # x: [B, C, F, T]

        # CNN → [B, 64, T]
        x = self.cnn(x)

        # Prepare for LSTM → [B, T, 64]
        x = x.permute(0, 2, 1)

        # LSTM → [B, 128]
        x = self.lstm(x)

        # Class logits → [B, 3]
        logits = self.classifier(x)

        return logits

    def encode(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        z = self.lstm(x)          # latent embedding
        return z                  # [B, 128]