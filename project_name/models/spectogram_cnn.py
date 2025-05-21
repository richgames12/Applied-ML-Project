import torch.nn as nn
import torch.nn.functional as F


class EmotionCNN(nn.Module):
    def __init__(self, num_classes=8, in_channels=1):
        """Initialize the CNN blocks.

        Args:
            num_classes (int, optional): The number of emotions in the dataset.
                Defaults to 8.
            in_channels (int, optional): The number of layers of the
                spectogram. 1 if 2D, 3 if RGB. Defaults to 1.
        """
        super(EmotionCNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
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

        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv_block1(x)  # Conv + BN + ReLU + Pool
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x = x.view(x.size(0), -1)  # Flatten the convolutional output
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
