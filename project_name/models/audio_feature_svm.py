from sklearn.svm import SVC
import numpy as np
import joblib
import os


class AudioFeatureSVM:
    """
    Fit and predict labels based on features using a SVM.
    """
    def __init__(
            self,
            kernel: str = "rbf",
            regularization_parameter: float = 1.0,
            gamma: str = "scale",
            probability: bool = False,
            seed: int | None = None
    ) -> None:
        """
        Initialize the SVM classifier for audio features.

        Args:
            kernel (str, optional): Kernel type for the SVM ('linear', 'rbf',
                'poly', etc.). Defaults to "rbf".
            regularization_parameter (float, optional): The regularization
                parameter to be used. Defaults to 1.0.
            gamma (str, optional): The kernel coefficient to be used. Defaults
                to "scale".
            probability (bool, optional): Whether to support probabilities
                over hard class predictions. Is only needed when using
                OneVsRest. Defaults to False
            seed (int | None, optional): The seed for sklearn functions.
                Defaults to None.
        """
        self.kernel = kernel
        self.regularization_parameter = regularization_parameter
        self.gamma = gamma
        self._probability = probability  # Should not be changed
        self.seed = seed
        self.type = "SVM"

        self._model = SVC(
            kernel=self.kernel,
            C=self.regularization_parameter,
            gamma=self.gamma,
            probability=self._probability,
            random_state=self.seed
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

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for the given features.

        Args:
            features (np.ndarray): Feature matrix of shape
                (n_samples, n_features).

        Returns:
            np.ndarray: Probability estimates of shape (n_samples, n_classes).
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
        return self._model.predict_proba(features)

    def get_params(self, deep=True) -> dict:
        """
        Return hyperparameter values as a dictionary. Required to work well
        with sklearn.

        Args:
            deep (bool): Whether to return the parameters of sub-objects.

        Returns:
            dict: Dictionary of parameter names mapped to their values.
        """
        return {
            "kernel": self.kernel,
            "regularization_parameter": self.regularization_parameter,
            "gamma": self.gamma,
            "probability": self._probability,
            "seed": self.seed
        }

    def set_params(self, **params) -> "AudioFeatureSVM":
        """
        Set hyperparameters from the provided dictionary. Required to work well
        with sklearn.

        Args:
            **params: Named hyperparameters to set.

        Returns:
            AudioFeatureSVM: The updated model.
        """
        for key, value in params.items():
            setattr(self, key, value)
        self._model = SVC(
            kernel=self.kernel,
            C=self.regularization_parameter,
            gamma=self.gamma,
            probability=self._probability,
            random_state=self.seed
        )
        return self

    def save(self, model_name: str = "audio_svm.joblib") -> None:
        """
        Save a trained AudioFeatureSVM model.

        Args:
            filepath (str): The location where the model should go. Defaults
                to "audio_svm.joblib".
        """
        if self._n_features is None:
            raise ValueError("Model has not been trained yet. Can't save.")
        # Determine where the models should be stored
        folder = folder = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "saved_models"
        )
        if not os.path.exists(folder):
            # Only create the folder if it does not yet exist
            os.makedirs(folder)

        filepath = os.path.join(folder, model_name)

        # Save the whole class instance in the folder
        joblib.dump(self, filepath)

    @classmethod  # Bound to class instead of specific object
    def load(cls, filepath: str) -> "AudioFeatureSVM":
        """
        Load a saved AudioFeaturesSVM model.

        Args:
            filepath (str): The location where the model is stored.

        Returns:
            AudioFeatureSVM: The loaded model instance.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model found at {filepath}")
        loaded_model = joblib.load(filepath)
        if not isinstance(loaded_model, cls):
            raise TypeError(f"Loaded object is not of type {cls.__name__}")
        return loaded_model
