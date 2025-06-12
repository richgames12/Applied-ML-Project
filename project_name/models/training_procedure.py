from project_name.models.spectrogram_cnn import MultiheadEmotionCNN
from project_name.models.audio_feature_svm import AudioFeatureSVM
from project_name.models.one_vs_rest import OneVsRestAudioFeatureSVM

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from project_name.evaluation.model_evaluation import ModelEvaluator
from datetime import datetime
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

EMOTION_LABELS = {
    0: 'neutral', 1: 'calm', 2: 'happy', 3: 'sad',
    4: 'angry', 5: 'fearful', 6: 'disgust', 7: 'surprised'
}
INTENSITY_LABELS = {0: 'normal', 1: 'strong'}


class TrainAndEval():
    def __init__(
        self,
        aug_features: np.ndarray,
        aug_labels: tuple[np.ndarray, np.ndarray],
        n_augmentations: int,
        model: MultiheadEmotionCNN | AudioFeatureSVM | OneVsRestAudioFeatureSVM,
        n_epochs: int = 20,
        task: str | None = None,  # 'emotion' or 'intensity' for SVM; ignored for CNN
        k_fold_splits: int = 5,
        seed: int | None = None
    ) -> None:
        # Store features and shift labels to start from 0
        self.aug_features = aug_features
        self.aug_labels = (aug_labels[0] - 1, aug_labels[1] - 1)
        self.n_augmentations = n_augmentations
        self.n_samples = int(aug_features.shape[0] / (n_augmentations + 1))
        self.seed = seed

        # Model and training setup
        self.model = model
        self.n_epochs = n_epochs
        self.k_fold_splits = k_fold_splits
        self.task = task

        self.best_params = None
        self.best_score = -np.inf  # Initialize to negative infinity for maximization

        # PCA for SVMs
        self.PCA = PCA(n_components=200, random_state=self.seed)

        # TensorBoard writer with unique log directory
        logdir = f"logs/run_{datetime.now():%Y%m%d_%H%M%S}"
        self.writer = SummaryWriter(logdir)

    def train_no_split(self, params: dict | None = None) -> None:
        """
        Train the model on all available data (no validation or cross-validation).
        """
        if params is None:
            # If no model parameters are provided, use best parameters if available
            if self.best_params is not None:
                params = self.best_params
            else:
                # Default model parameters if none are provided
                params = {}

        model_params = {
            param_name: param_value
            for param_name, param_value in params.items()
            if param_name in [
                "dropout_rate",
                "regularization_parameter",
                "kernel"
            ]
        }
        train_params = {
            param_name: param_value
            for param_name, param_value in params.items()
            if param_name not in model_params
        }

        if self.model.type == "CNN":
            self._train_CNN_split(
                train_features=self.aug_features,
                val_features=None,
                emotion_train_labels=self.aug_labels[0],
                emotion_val_labels=None,
                intensity_train_labels=self.aug_labels[1],
                intensity_val_labels=None,
                n_epoch=train_params.get("n_epoch", self.n_epochs),
                learning_rate=train_params.get("learning_rate", 1e-3),
                batch_size=train_params.get("batch_size", 32),
                writer=self.writer,
                return_val_score=False,
                model_params=model_params
            )
        elif self.model.type == "SVM":
            self._train_SVM_split(
                train_features=self.aug_features,
                train_labels=self.aug_labels[0] if self.task == "emotion" else self.aug_labels[1],
                model_params=model_params
            )
        else:
            raise ValueError("Unknown model type.")

    def _k_fold(
        self,
        k_folds: int = 5,
        n_epoch: int = 20,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        return_val_score: bool = False,
        model_params=None
    ) -> float | None:
        """Evaluate using k-fold cross-validation"""
        n_total = self.n_samples
        n_aug = self.n_augmentations
        # Generate indices for k-fold cross-validation
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=self.seed)
        total_val_score = 0.0

        for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(n_total))):
            print(f"Fold {fold+1}/{k_folds}")

            # Get all augmented indices for training
            train_aug_idx = []
            for idx in train_idx:
                start = idx * (n_aug + 1)
                end = start + (n_aug + 1)
                train_aug_idx.extend(range(start, end))
            train_aug_idx = np.array(train_aug_idx)

            # Validation indices: only the original (first) sample of each group
            val_orig_idx = val_idx * (n_aug + 1)

            train_features = self.aug_features[train_aug_idx]
            val_features = self.aug_features[val_orig_idx]

            if self.model.type == "CNN":
                # For MultiheadEmotionCNN, we have two sets of labels
                train_emotion_labels = self.aug_labels[0][train_aug_idx]
                train_intens_labels = self.aug_labels[1][train_aug_idx]
                val_emote_labels = self.aug_labels[0][val_orig_idx]
                val_intens_labels = self.aug_labels[1][val_orig_idx]
            elif self.model.type == "SVM":
                # For AudioFeatureSVM, we have one set of labels depending on the task
                if self.task == "emotion":
                    train_labels = self.aug_labels[0][train_aug_idx]
                    val_labels = self.aug_labels[0][val_orig_idx]
                elif self.task == "intensity":
                    train_labels = self.aug_labels[1][train_aug_idx]
                    val_labels = self.aug_labels[1][val_orig_idx]
                else:
                    raise ValueError("Unknown task type: must be 'emotion' or 'intensity'")

            if self.model.type == "CNN":
                val_score = self._train_CNN_split(
                    train_features=train_features,
                    val_features=val_features,
                    emotion_train_labels=train_emotion_labels,
                    emotion_val_labels=val_emote_labels,
                    intensity_train_labels=train_intens_labels,
                    intensity_val_labels=val_intens_labels,
                    n_epoch=n_epoch,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    writer=self.writer,
                    return_val_score=return_val_score,
                    model_params=model_params
                )
                self.writer.close()

            elif self.model.type == "SVM":
                val_score = self._train_SVM_split(
                    train_features=train_features,
                    train_labels=train_labels,
                    val_features=val_features,
                    val_labels=val_labels,
                    return_val_score=return_val_score,
                    model_params=model_params
                )

            if return_val_score and val_score is not None:
                total_val_score = total_val_score + val_score
            print(val_score)
        if return_val_score:
            # If we want to return the validation score, we can return the last one
            return total_val_score / k_folds

        # If we don't want to return the validation score, we just return None
        return None

    def _holdout(self, val_proportion: float = 0.2, n_epoch=20, learning_rate=1e-3, batch_size=32, return_val_score=False, model_params=None) -> None:
        """
        Evaluate using holdout data split with shuffling.
        Args:
            val_proportion: float, proportion of data to use for validation (default 0.2)
        """
        n_aug = self.n_augmentations

        # Shuffle original indices using np.random
        orig_indices = np.arange(self.n_samples)
        np.random.shuffle(orig_indices)

        # Compute augmented indices for shuffled originals
        aug_indices = np.concatenate([
            np.arange(idx * (n_aug + 1), (idx + 1) * (n_aug + 1)) for idx in orig_indices
        ])

        # Only split into train and validation
        val_length = int(self.n_samples * val_proportion)
        val_begin = self.n_samples - val_length
        val_end = self.n_samples

        # Reorder features and labels according to shuffled indices
        shuffled_features = self.aug_features[aug_indices]
        val_features, train_features = self._data_split(shuffled_features, val_begin, val_end)

        # Split labels according to the same indices
        # If the model is MultiheadEmotionCNN, we have two sets of labels
        if self.model.type == "CNN":
            shuffled_emotion_labels = self.aug_labels[0][aug_indices]
            shuffled_intensity_labels = self.aug_labels[1][aug_indices]
            val_emotion_labels, train_emotion_labels = self._data_split(shuffled_emotion_labels, val_begin, val_end)
            val_intensity_labels, train_intensity_labels = self._data_split(shuffled_intensity_labels, val_begin, val_end)
        elif self.model.type == "SVM":
            # For AudioFeatureSVM, we have one set of labels depending on the task
            if self.task == "emotion":
                shuffled_labels = self.aug_labels[0][aug_indices]
            elif self.task == "intensity":
                shuffled_labels = self.aug_labels[1][aug_indices]
            else:
                raise ValueError("Unknown task type: must be 'emotion' or 'intensity'")
            val_labels, train_labels = self._data_split(shuffled_labels, val_begin, val_end)

        if self.model.type == "CNN":
            val_score = self._train_CNN_split(
                train_features=train_features,
                val_features=val_features,
                emotion_train_labels=train_emotion_labels,
                emotion_val_labels=val_emotion_labels,
                intensity_train_labels=train_intensity_labels,
                intensity_val_labels=val_intensity_labels,
                n_epoch=n_epoch,
                learning_rate=learning_rate,
                batch_size=batch_size,
                writer=self.writer,
                return_val_score=return_val_score,
                model_params=model_params
            )
            self.writer.close()

        elif self.model.type == "SVM":
            val_score = self._train_SVM_split(
                train_features=train_features, 
                val_features=val_features,
                train_labels=train_labels,
                val_labels=val_labels,
                return_val_score=return_val_score,
                model_params=model_params
            )

        if return_val_score:
            return val_score

    def _data_split(self, data: np.ndarray, val_begin: int, val_end: int) -> tuple:
        """Split data into test and validation sets
        args:
        data: np.ndarray, data to be split
        val
        """
        val_begin *= (self.n_augmentations + 1)
        val_end *= (self.n_augmentations + 1)
        val_data = data[val_begin:val_end]
        val_data = val_data[::(self.n_augmentations + 1)]
        train_data = np.concatenate((data[:val_begin], data[val_end:]))
        return val_data, train_data

    def _train_CNN_split(
        self,
        train_features: np.ndarray,
        emotion_train_labels: np.ndarray,
        intensity_train_labels: np.ndarray,
        # Validation data can be None if no validation is needed
        val_features: np.ndarray | None = None,
        emotion_val_labels: np.ndarray | None = None,
        intensity_val_labels: np.ndarray | None = None,
        n_epoch=20,
        learning_rate=1e-3,
        batch_size=32,
        writer=None,
        return_val_score=False,
        model_params: dict = None
    ) -> float | None:
        if model_params is None:
            model_params = {}

        # Convert numpy arrays to torch tensors and create DataLoaders
        train_dataset = TensorDataset(
            torch.tensor(train_features, dtype=torch.float32),
            torch.tensor(emotion_train_labels, dtype=torch.long),
            torch.tensor(intensity_train_labels, dtype=torch.long)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if val_features is not None and emotion_val_labels is not None and intensity_val_labels is not None:
            val_dataset = TensorDataset(
                torch.tensor(val_features, dtype=torch.float32),
                torch.tensor(emotion_val_labels, dtype=torch.long),
                torch.tensor(intensity_val_labels, dtype=torch.long)
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None

        self.model = MultiheadEmotionCNN(**model_params)

        # Call the model's training method and return validation score if required
        # Pass val_loader to the model's fit method (it should be able to handle None)
        val_score = self.model.fit(
            val_dataloader=val_loader,
            train_dataloader=train_loader,
            writer=writer,
            epochs=n_epoch,
            learning_rate=learning_rate,
            return_val_score=return_val_score
        )
        return val_score

    def _train_SVM_split(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,  # Now generic: can be emotion or intensity
        val_features: np.ndarray = None,
        val_labels: np.ndarray = None,
        return_val_score: bool = False,
        model_params: dict = None
    ):
        if model_params is None:
            model_params = {}

        # Flatten spectrograms for SVM input
        spec_train_flat = train_features.reshape(train_features.shape[0], -1)
        self.pca = PCA(n_components=200, random_state=self.seed)
        spec_train_reduced = self.pca.fit_transform(spec_train_flat)

        # Train SVM for the current task
        if isinstance(self.model, OneVsRestAudioFeatureSVM):
            self.model = OneVsRestAudioFeatureSVM(**model_params, seed=self.seed)
        else:
            self.model = AudioFeatureSVM(**model_params, seed=self.seed)

        self.model.fit(spec_train_reduced, train_labels)
        print("Spectrogram-based SVM trained.")

        # Validation
        if return_val_score and val_features is not None and val_labels is not None:
            val_flat = val_features.reshape(val_features.shape[0], -1)
            val_reduced = self.pca.transform(val_flat)
            predicted = self.model.predict(val_reduced)
            score = f1_score(val_labels, predicted, average="weighted")
            return score

        return None

    def _mc_uncertainty(self, eval_data, n=100):
        # currently for a single sample
        self.model.train()  # to enable dropout, because no grad or step this wont update weights
        outcomes = []
        with torch.no_grad():
            for _ in range(n):
                output = self.model.forward(eval_data)
                prob = torch.nn.functional.softmax(output[0], dim=1)
                outcomes.append(prob)
        print(outcomes)
        outcomes = torch.stack(outcomes)
        mean_class_probs = outcomes.mean(dim=0)
        class_std = outcomes.std(dim=0)
        print(mean_class_probs)
        print(class_std)

    def _filter_params(self, params: dict) -> tuple[dict, dict]:
        """
        Filter model and training parameters from the provided dictionary.
        Only parameters that are relevant for the model and training
        are returned. This is useful for hyperparameter tuning.

        Args:
            params (dict): Dictionary of parameters to filter.

        Returns:
            tuple[dict, dict]: Filtered model parameters and training parameters.
        """
        if self.model.type == "CNN":
            model_param_names = [
                "dropout_rate",
            ]
        elif self.model.type == "SVM":
            model_param_names = [
                "regularization_parameter",
                "kernel",
                "gamma",
                "max_iter",
            ]
        train_param_names = [
            "n_epoch",
            "learning_rate",
            "batch_size"
        ]

        model_params = {
            param_name: param_value
            for param_name, param_value in params.items()
            if param_name in model_param_names
        }
        train_params = {
            param_name: param_value
            for param_name, param_value in params.items()
            if param_name in train_param_names
        }
        return model_params, train_params

    def hyperparameter_tune(self, param_grid: list[dict], cross_val="k_fold", model_name: str | None = "best_model") -> None:
        """
        Perform hyperparameter tuning using the provided parameter grid.
        This method iterates over the parameter grid and evaluates the model.

        Args:
            param_grid (list[dict]): List of dictionaries, where each dictionary
                contains a set of parameters to evaluate. Each dictionary should 
                contain keys for both model and training parameters, e.g.:
                {dropout_rate: 0.5, regularization_parameter: 0.01, kernel: 'rbf',
                gamma: 0.1, n_epoch: 20, learning_rate: 0.001, batch_size: 32}
                and so on for other parameters. Default values are used if not provided.
            cross_val (str, optional): The type of cross validation to use.
                "k_fold" or "holdout" Defaults to "k_fold".
            model_name (str, optional): Name of the model to save after tuning. If None,
                the model will not be saved. Defaults to "best_model".

        Raises:
            ValueError: If an unknown cross-validation type is provided.
        """
        for params in tqdm(param_grid, desc="Hyperparameter tuning"):

            # Split parameters into model and training parameters
            model_params, train_params = self._filter_params(params)
            print(model_params, train_params)
            # Model parameters are used to initialize the model while training parameters
            # are used to fit the model
            if cross_val == "k_fold":
                val_score = self._k_fold(return_val_score=True, **train_params, model_params=model_params)
            elif cross_val == "holdout":
                val_score = self._holdout(return_val_score=True, **train_params, model_params=model_params)
            else:
                raise ValueError("Unknown cross-validation type.")
            if val_score > self.best_score:
                self.best_score = val_score
                self.best_params = params

        print(f"Best validation score: {self.best_score} with parameters: {self.best_params}")

        # Retrain best model on all training data
        print("Retraining best model on all training data...")
        self.train_no_split()
        if model_name is not None:
            # Save the best model
            print(f"Saving best model to {model_name}.")
            self.model.save("best_model")

    def evaluate_on_testset(self, test_features, emotion_test_labels: np.ndarray = None, intensity_test_labels: np.ndarray = None, title_suffix: str = "") -> None:
        """
        Evaluate the model on the test set and write metrics.
        Args:
            test_features: np.ndarray or torch.Tensor, test features
            emotion_test_labels: np.ndarray, true emotion labels
            intensity_test_labels: np.ndarray, true intensity labels
        """
        emotion_evaluator = ModelEvaluator(EMOTION_LABELS)
        intensity_evaluator = ModelEvaluator(INTENSITY_LABELS)

        if self.model.type == "CNN":
            if emotion_test_labels is None or intensity_test_labels is None:
                raise ValueError("Both emotion and intensity test labels must be provided for MultiheadEmotionCNN.")

            # Convert test features to torch tensor
            test_features_tensor = torch.tensor(test_features, dtype=torch.float32)
            self.model.eval()
            with torch.no_grad():
                predictions = self.model.predict(test_features_tensor)
                emotion_predictions = predictions[0].numpy()
                intensity_predictions = predictions[1].numpy()
            emotion_evaluator.evaluate_from_predictions(emotion_test_labels, emotion_predictions, title_suffix=title_suffix)
            intensity_evaluator.evaluate_from_predictions(intensity_test_labels, intensity_predictions, title_suffix=title_suffix)

        elif self.model.type == "SVM":
            test_flat = test_features.reshape(test_features.shape[0], -1)
            test_reduced = self.pca.transform(test_flat)
            predictions = self.model.predict(test_reduced)

            # SVMs can only predict one task at a time, so we need to handle both tasks separately
            if self.task == "emotion":
                if emotion_test_labels is None:
                    raise ValueError("Emotion test labels must be provided for emotion SVM.")

                emotion_evaluator.evaluate_from_predictions(emotion_test_labels, predictions, title_suffix=title_suffix)
            elif self.task == "intensity":
                if intensity_test_labels is None:
                    raise ValueError("Intensity test labels must be provided for intensity SVM.")

                intensity_evaluator.evaluate_from_predictions(intensity_test_labels, predictions, title_suffix=title_suffix)
            else:
                raise ValueError("Unknown task type: must be 'emotion' or 'intensity'")
        else:
            raise ValueError("Unknown model type. Cannot evaluate on test set.")
