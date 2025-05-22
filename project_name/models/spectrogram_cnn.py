import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.fc_shared = nn.Linear(64 * 16 * 16, 256)

        # Emotion classification head
        self.fc_emotion = nn.Linear(256, num_emotions)

        # Intensity classification head
        self.fc_intensity = nn.Linear(256, num_intensity)

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
            emotion_logits, intensity_logits = self.forward(x)
        
        emotion_predictions = torch.argmax(emotion_logits, dim=1)
        intensity_predictions = torch.argmax(intensity_logits, dim=1)

        return emotion_predictions, intensity_predictions


#Keeping this outisde of the class because it makes more sense to add this to thre pipeline, 
#seeing as we may want to use different types of training method, this uses stochastic gradient descent
def fit(self, x: torch.Tensor, classes: tuple[torch.Tensor, torch.Tensor]):
    """
    We fit the model to dual-task dataset. 

    Args:
        x (torch.Tensor): Input batch of spectrograms, has the following
            shape: [batch_size, in_channels, height, width]. where
            batch_size is the number of spectrograms in the batch,
            in_channels the number of layers of the spectrogram (1 by
            default) and height and width the sizes of the spectrogram.
        classes (tuple): of (torch.Tensor, torch.Tensor) 
        The first tensor has size [batch, num of classes for task 1]
        The second tensor has size [batch, num of classes for task 2]
        

    Returns:

    """
        