import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import os


class MultiheadEmotionCNN(nn.Module):
    """
    Create a CNN for multitask classification of emotion and intensity from
    spectrograms.

    This model uses shared feature extraction convolutional blocks followed by
    a shared fully connected layer. Then uses 2 separate fully connected heads
    to predict the emotion and intensity classes.
    """
    def __init__(self, num_emotions=8, num_intensity=2, in_channels=1, dropout_rate=0.3) -> None:
        """
        Initialize the CNN with the right convolutional blocks and fully
        connected layers.

        Args:
            num_emotions (int, optional): Number of emotions to classify.
                Defaults to 8.
            num_intensity (int, optional): Number of intensities to classify.
                Defaults to 2.
            in_channels (int, optional): The number of layers of the
                spectogram. Normal 2D specteogram has 1, RGB spectrogram has 3.
                Defaults to 1.
            dropout_rate (float, optional): The dropout rate to use in the
                fully connected layer. Defaults to 0.3.
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
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_shared = nn.Linear(64 * 16 * 16, 256)

        # Emotion classification head
        self.fc_emotion = nn.Linear(256, num_emotions)

        # Intensity classification head
        self.fc_intensity = nn.Linear(256, num_intensity)

        self.trained = False  # To track the number of features after training

        self.type = "CNN"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

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
        x = x.to(self.device)
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
        if not self.trained:
            raise ValueError(
                "Model has not yet been trained, call fit() first."
            )
        self.eval()
        x = x.to(self.device)
        with torch.no_grad():
            emotion_logits, intensity_logits = self(x)

        emotion_predictions = torch.argmax(emotion_logits, dim=1)
        intensity_predictions = torch.argmax(intensity_logits, dim=1)

        return emotion_predictions.cpu(), intensity_predictions.cpu()

    def fit(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,  # It can be None if no validation is needed
        writer: None | SummaryWriter = None,
        epochs: int = 20,
        learning_rate: float = 1e-3,
        return_val_score: bool = False
    ) -> float | None:
        """
        Train the CNN model on the training data and optionally validate it on
        the validation data.

        Args:
            train_dataloader (DataLoader): DataLoader for the training data.
            val_dataloader (DataLoader | None, optional): DataLoader for the validation data.
                If None, no validation will be performed. Defaults to None.
            writer (SummaryWriter | None, optional): TensorBoard writer for logging.
                If None, no logging will be performed. Defaults to None.
            epochs (int, optional): Number of epochs to train the model for.
                Defaults to 20.
            learning_rate (float, optional): The learning rate for the optimizer.
                Defaults to 1e-3.
            return_val_score (bool, optional): If True, returns the average validation loss.
                If False, returns None. Defaults to False.

        Returns:
            float | None: If `return_val_score` is True, returns the average validation loss.
                If `return_val_score` is False, returns None.
        """
        self.trained = True
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        loss_func = nn.CrossEntropyLoss()
        self.train()
        for epoch in range(epochs):
            total_train_loss = 0
            for features, emotion_labels, intensity_labels in train_dataloader:
                features = features.to(self.device)
                emotion_labels = emotion_labels.to(self.device)
                intensity_labels = intensity_labels.to(self.device)

                optimizer.zero_grad()
                emotion_logits, intensity_logits = self(features)
                loss_sum = loss_func(emotion_logits, emotion_labels) + loss_func(intensity_logits, intensity_labels)
                loss_sum.backward()
                optimizer.step()
                total_train_loss += loss_sum.item()

            # Only run validation if val_dataloader is provided
            if val_dataloader is not None:
                with torch.no_grad():
                    total_val_loss = 0
                    for features, emotion_labels, intensity_labels in val_dataloader:
                        features = features.to(self.device)
                        emotion_labels = emotion_labels.to(self.device)
                        intensity_labels = intensity_labels.to(self.device)
                        emotion_logits, intensity_logits = self(features)
                        loss_sum = loss_func(emotion_logits, emotion_labels) + loss_func(intensity_logits, intensity_labels)
                        total_val_loss += loss_sum.item()

                if writer is not None:
                    writer.add_scalars('Loss', {
                        'train': (total_train_loss / len(train_dataloader)),
                        'val': (total_val_loss / len(val_dataloader))
                    }, epoch)
            else:
                if writer is not None:
                    writer.add_scalar('Loss/train', (total_train_loss / len(train_dataloader)), epoch)

        # Only return validation loss if validation was performed
        if val_dataloader is not None:
            avg_val_loss = total_val_loss / len(val_dataloader)
            if return_val_score:
                return -avg_val_loss  # Negative loss for maximization in tuning
        return None

    def save(self, model_name: str = "emotion_cnn") -> None:
        """
        Save the model to a file.

        Args:
            model_name (str): The name of the file to save the model to.
        """
        if not self.trained:
            raise ValueError(
                "Model has not yet been trained, call fit() first."
            )
        folder = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "saved_models"
        )
        if not os.path.exists(folder):
            os.makedirs(folder)
        file_path = os.path.join(folder, f"{model_name}.pth")

        torch.save(self, file_path)

    @classmethod
    def load(cls, file_path: str) -> "MultiheadEmotionCNN":
        """
        Load the model from a file.

        Args:
            file_path (str): The full filepath to load the model from.

        Returns:
            MultiheadEmotionCNN: The loaded model.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file {file_path} does not exist.")

        model = torch.load(file_path, map_location=torch.device("cpu"))

        if not isinstance(model, MultiheadEmotionCNN):
            raise TypeError(
                f"Loaded model is not of type MultiheadEmotionCNN, "
                f"got {type(model)} instead."
            )
        return model
