from sklearn.multiclass import OneVsRestClassifier
from project_name.models.audio_feature_svm import AudioFeatureSVM
import numpy as np
import joblib
import os

class OneVsRestAudioFeatureSVM(OneVsRestClassifier):
    """
    A wrapper for OneVsRestClassifier to save and load SVM model.
    """

    def __init__(self, **kwargs):
        super().__init__(AudioFeatureSVM(**kwargs))
        self._n_features = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "OneVsRestAudioFeatureSVM":
        """Fit the model to the training data.

        Args:
            features (np.ndarray): The features of the audio should be of
                shape (n_samples, n_features).
            labels (np.ndarray): The labels corresponding to the features
                should be of shape (n_samples).

        Returns:
            OneVsRestAudioFeatureSVM: The fitted model.
        """
        if features.ndim != 2:
            raise ValueError(
                f"Expected input with 2 dimensions but got {features.ndim} "
                f"instead.")
        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                f"The number of features: {features.shape[0]} should be equal "
                f"to the number of labels: {labels.shape[0]}."
            )
        super().fit(features, labels)
        self._n_features = features.shape[1]

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict labels for the given features.

        Args:
            features (np.ndarray): The features of the audio should be of
                shape (n_samples, n_features).

        Returns:
            np.ndarray: The predicted labels for the features.
        """
        if self._n_features is None:
            raise ValueError(
                "Model has not yet been trained, call fit() first."
            )
        elif features.shape[1] != self._n_features:
            raise ValueError(
                f"Expected input with {self._n_features} features but got "
                f"{features.shape[1]} features instead."
            )

        return super().predict(features)
    
    def save(self, model_name: str = "emotion_svm") -> None:
        """Save the model to a file.

        Args:
            model_name (str): The name of the file to save the model to.
        """
        if self._n_features is None:
            raise ValueError(
                "Model has not yet been trained, call fit first."
            )
        folder = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "saved_models"
        )
        if not os.path.exists(folder):
            os.makedirs(folder)
        file_path = os.path.join(folder, f"{model_name}.joblib")

        joblib.dump(self, file_path)

    @classmethod
    def load(self, file_path: str) -> "OneVsRestAudioFeatureSVM":
        """Load the model from a file.

        Args:
            file_path (str): The full filepath to load the model from.

        Returns:
            OneVsRestAudioFeatureSVM: The loaded model.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file {file_path} does not exist.")

        model = joblib.load(file_path)

        if not isinstance(model, OneVsRestAudioFeatureSVM):
            raise TypeError(
                f"Loaded model is not of type OneVsRestAudioFeatureSVM, "
                f"got {type(model)} instead."
            )
        return model