from sklearn.svm import SVC
import numpy as np


class AudioFeatureSVM:
    """
    Fit and predict labels based on features using a SVM.
    """
    def __init__(
            self,
            kernel: str = "rbf",
            regularization_parameter: float = 1.0,
            gamma: str = "scale"
    ) -> None:
        """
        Initialize the SVM classifier for audio features.

        Args:
            kernel (str, optional): The kernel type to be used. Defaults to
                "rbf".
            regularization_parameter (float, optional): The regularization
                parameter to be used. Defaults to 1.0.
            gamma (str, optional): The kernel coefficient to be used. Defaults
                to "scale".
        """
        self._model = SVC(
            kernel=kernel,
            C=regularization_parameter,
            gamma=gamma
        )
        self._n_features = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """
        Fit the SVM with the given features and labels.

        Args:
            features (np.ndarray): The features of the audio should be of
                shape (n_samples, n_features).
            labels (np.ndarray): The labels corresponding to the features
                should be of shape (n_samples).
        """
        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                f"The number of features: {features.shape[0]} should be equal",
                f"to the number of labels: {labels.shape[0]}."
            )
        self._model.fit(features, labels)
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
                f"Expected input with {self._n_features} features but got"
                f"{features.shape[1]} features instead."
            )

        return self._model.predict(features)

    def score(self, features: np.ndarray, labels: np.ndarray) -> float:
        """
        Determine the accuracy of the classifier on the given test data.

        Args:
            features (np.ndarray): The features of the audio should be of
                shape (n_samples, n_features).
            labels (np.ndarray): The labels corresponding to the features
                should be of shape (n_samples).

        Returns:
            float: The accuracy of the model on the test data.
        """
        return self._model.score(features, labels)
