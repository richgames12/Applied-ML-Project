import torch.nn as nn
import torch.nn.functional as F


class MultiheadEmotionCNN(nn.Module):
    def __init__(self, num_emotions=8, num_intensity=2, in_channels=1):
        """Initialize the CNN blocks."""
        super(MultiheadEmotionCNN, self).__init__()

        # Shared feature extraction blocks
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),  # Expects 2D
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Shared fully connected layer
        self.dropout = nn.Dropout(0.3)
        self.fc_shared = nn.Linear(64 * 16 * 16, 256)

        # Emotion classification head
        self.fc_emotion = nn.Linear(256, 8)

        # Intensity classification head
        self.fc_intensity = nn.Linear(256, 2)

    def forward(self, x):
        # Shared Conv + BN + ReLU + Pool
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        # Shared fully connected
        x = x.view(x.size(0), -1)  # Flatten the convolutional output
        x = self.dropout(F.relu(self.fc_shared(x)))

        # Two separate fully connected heads
        emotion_logits = self.fc_emotion(x)
        intensity_logits = self.fc_intensity(x)
        return emotion_logits, intensity_logits
