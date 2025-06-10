import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


class MultiheadEmotionCNN(nn.Module):
    """
    Create a CNN for multitask classification of emotion and intensity from
    spectrograms.

    This model uses shared feature extraction convolutional blocks followed by
    a shared fully connected layer. Then uses 2 separate fully connected heads
    to predict the emotion and intensity classes.
    """
    def __init__(self, num_emotions=8, num_intensity=2, in_channels=1) -> None:
        """Initialize the CNN with the right convolutional blocks and fully
            connected layers.

        Args:
            num_emotions (int, optional): Number of emotions to classify.
                Defaults to 8.
            num_intensity (int, optional): Number of intensities to classify.
                Defaults to 2.
            in_channels (int, optional): The number of layers of the
                spectogram. Normal 2D specteogram has 1, RGB spectrogram has 3.
                Defaults to 1.
        """
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
        self.fc_shared = nn.Linear(64 * 16 * 16, 512)

        # Emotion classification head
        self.fc_emotion = nn.Linear(512, num_emotions)

        # Intensity classification head
        self.fc_intensity = nn.Linear(512, num_intensity)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass through the CNN.

        Args:
            x (torch.Tensor): Input batch of spectrograms, has the following
                shape: [batch_size, in_channels, height, width]. where
                batch_size is the number of spectrograms in the batch,
                in_channels the number of layers of the spectrogram (1 by
                default) and height and width the sizes of the spectrogram.

        Returns:
            tuple[ torch.Tensor, torch.Tensor ]: Emotion logits and intensity
                logits. Emotion logits have size [batch_size, num_emotions]
                and intensity logits size [batch_size, num_intensity].
        """
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

    def predict(self, x: torch.Tensor):
        """
        Predict/classify a batch of spectograms. 
        Do this using argmax on the logits

        Args:
            x (torch.Tensor): Input batch of spectrograms, has the following
                shape: [batch_size, in_channels, height, width]. where
                batch_size is the number of spectrograms in the batch,
                in_channels the number of layers of the spectrogram (1 by
                default) and height and width the sizes of the spectrogram.

        Returns:
            index of most likely and thus predicted class for each spectogram in the batch,
            for both tasks
            tuple[ torch.Tensor, torch.Tensor ]: each tensor has size [batch_size]

        """
        self.eval()
        with torch.no_grad():
            emotion_logits, intensity_logits = self(x)

        emotion_predictions = torch.argmax(emotion_logits, dim=1)
        intensity_predictions = torch.argmax(intensity_logits, dim=1)

        return emotion_predictions, intensity_predictions


    def fit_model(self, dataloader, epochs: int):
        """
        We fit the model to dual-task dataset.

        Args:
            dataloader: pytorch dataloader that has train data and batch size
            model: pytorch model
            epochs: int number of epochs to train for

        Returns:

        """
        # We can pass these as parameters to the train method but im setting them 
        # here for simplicity
        optimizer = optim.Adam(self.parameters())
        loss_func = nn.CrossEntropyLoss()
        self.train()
        train_loss = []
        val_loss = []
        for epoch in range(epochs):
            total_loss = 0
            for features, emotion_labels, intensity_labels in dataloader:
                emotion_logits, intensity_logits = self(features)
                loss_sum = loss_func(emotion_logits, emotion_labels) + loss_func(intensity_logits, intensity_labels)
                loss_sum.backward()
                optimizer.step()
        return None

    def cross_val_fit(self, test_dataloader: DataLoader, train_dataloader: DataLoader, writer: None | SummaryWriter, epochs: int) -> None:
            """
            We fit the model to dual-task dataset.

            Args:
                dataloader: pytorch dataloader that has train data and batch size
                model: pytorch model
                writer: tesnorboard writer (if applicable, if not: None)
                epochs: int number of epochs to train for

            Returns:
                None
            """
            # We can pass these as parameters to the train method but im setting them
            # here for simplicity
            optimizer = optim.Adam(self.parameters())
            loss_func = nn.CrossEntropyLoss()
            self.train()
            for epoch in range(epochs):
                total_train_loss = 0
                for features, emotion_labels, intensity_labels in train_dataloader:
                    emotion_logits, intensity_logits = self(features)
                    loss_sum = loss_func(emotion_logits, emotion_labels) + loss_func(intensity_logits, intensity_labels)
                    loss_sum.backward()
                    optimizer.step()
                    total_train_loss += loss_sum.item()

                with torch.no_grad():
                    total_test_loss = 0
                    for features, emotion_labels, intensity_labels in test_dataloader:
                        emotion_logits, intensity_logits = self(features)
                        loss_sum = loss_func(emotion_logits, emotion_labels) + loss_func(intensity_logits, intensity_labels)
                        total_test_loss += loss_sum.item()

                if writer is not None:
                    writer.add_scalars('Loss', {
                        'train': (total_train_loss / len(train_dataloader)),
                        'test': (total_test_loss / len(test_dataloader))
                    }, epoch)

            return None
