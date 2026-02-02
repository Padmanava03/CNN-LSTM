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

        # Pool only across frequency (retain time)
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, x):
        # x: [B, C, F, T]

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Pool frequency → [B, 64, 1, T]
        x = self.freq_pool(x)

        # Remove freq dim → [B, 64, T]
        x = x.squeeze(2)

        return x

class EEGBiLSTM(nn.Module):
    """
    Bi-LSTM for temporal neural-state modeling
    Input : [B, T, D]
    Output:
        - hidden_states: [B, T, 2H]
        - final_embedding: [B, 2H]
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=1):
        super().__init__()

        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x):
        # x: [B, T, D]

        output, (h_n, _) = self.bilstm(x)

        # output → [B, T, 2H] (state sequence)
        # h_n → [2*num_layers, B, H]

        # Concatenate last forward & backward hidden states
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        final_embedding = torch.cat([h_forward, h_backward], dim=1)

        return output, final_embedding

class CNNBiLSTMEncoder(nn.Module):
    """
    Encoder-only model for unlabeled EEG depth perception
    """
    def __init__(self, num_channels):
        super().__init__()

        self.cnn = EEGCNN(in_channels=num_channels)
        self.bilstm = EEGBiLSTM(input_dim=64, hidden_dim=128)

    def forward(self, x):
        """
        Forward pass returning neural state sequence + embedding
        """
        # x: [B, C, F, T]

        # CNN → [B, 64, T]
        x = self.cnn(x)

        # Prepare for LSTM → [B, T, 64]
        x = x.permute(0, 2, 1)

        # Bi-LSTM
        state_seq, embedding = self.bilstm(x)

        return state_seq, embedding

    def encode(self, x):
        """
        Return only the trial-level latent embedding
        """
        _, embedding = self.forward(x)
        return embedding
    
class DepthClassifier(nn.Module):
    """
    Final classification head
    """
    def __init__(self, input_dim, num_classes=3):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

class CNNBiLSTMDepthModel(nn.Module):
    def __init__(self, num_channels, num_classes=3):
        super().__init__()

        self.encoder = CNNBiLSTMEncoder(num_channels)
        self.classifier = DepthClassifier(input_dim=256, num_classes=num_classes)

    def forward(self, x):
        _, embedding = self.encoder(x)
        logits = self.classifier(embedding)
        return logits