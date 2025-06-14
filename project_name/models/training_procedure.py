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
import random as rnd
import heapq
import math
import joblib
import os

from torch.utils.tensorboard import SummaryWriter

EMOTION_LABELS = {
    0: 'neutral', 1: 'calm', 2: 'happy', 3: 'sad',
    4: 'angry', 5: 'fearful', 6: 'disgust', 7: 'surprised'
}
INTENSITY_LABELS = {0: 'normal', 1: 'strong'}


class TrainAndEval():
    """
    Class to handle training and evaluation of audio models.
    This class supports training with or without validation, hyperparameter tuning,
    and evaluation on test sets. It can handle both CNN and SVM models for audio
    classification tasks, specifically for emotion and intensity recognition.
    """
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
        """
        Initialize the training and evaluation class.
        This method sets up the training and evaluation environment.

        Args:
            aug_features (np.ndarray): The augmented features of the audio data. Also includes the original
                features, so the shape is (n_samples * (n_augmentations + 1), n_features).
                The original features are before the augmented features. Like this:
                [[original_feature_1], [augmented_feature_1_1], [augmented_feature_1_2],
                [original_feature_2], [augmented_feature_2_1], [augmented_feature_2_2]]]
            aug_labels (tuple[np.ndarray, np.ndarray]): A tuple containing two numpy arrays:
                - The first array contains emotion labels (shape: (n_samples * (n_augmentations + 1),)).
                - The second array contains intensity labels (shape: (n_samples * (n_augmentations + 1),)).
            n_augmentations (int): The number of augmentations applied to each original sample.
            model (MultiheadEmotionCNN | AudioFeatureSVM | OneVsRestAudioFeatureSVM): The model to be trained.
            n_epochs (int, optional): The number of training cycles for CNNs. Defaults to 20.
            task (str | None, optional): The task type for SVMs, either 'emotion' or 'intensity'.
                If None, the task is ignored (only relevant for SVMs). Defaults to None.
            k_fold_splits (int, optional): Number of splits for k-fold cross-validation.
                Defaults to 5.
            seed (int | None, optional): The random seed for reproducibility.
                If None, no seed is set. Defaults to None.
        """
        # Store features and shift labels to start from 0
        self.aug_features = aug_features
        self.aug_labels = (aug_labels[0] - 1, aug_labels[1] - 1)
        self.n_augmentations = n_augmentations
        self.n_samples = int(aug_features.shape[0] / (n_augmentations + 1))

        # Set random seed for reproducibility
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

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
        Train the model without splitting the data.
        This method trains the model on the entire dataset without validation.

        Args:
            params (dict | None, optional): A dictionary of model and training parameters.
                If None, the best parameters are used if available, otherwise default parameters are used.
        """
        if params is None:
            # If no model parameters are provided, use best parameters if available
            if self.best_params is not None:
                params = self.best_params
            else:
                # Default model parameters if none are provided
                params = {}

        model_params, train_params = self._filter_params(params)

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
        write_val_loss: bool = True,
        model_params: dict | None = None
    ) -> float | None:
        """
        Train the model using k-fold cross-validation.

        Args:
            k_folds (int, optional): The number of folds for k-fold cross-validation.
                Defaults to 5.
            n_epoch (int, optional): The number of training epochs for CNNs.
                Defaults to 20.
            learning_rate (float, optional): The learning rate of the optimizer for CNNs.
                Defaults to 1e-3.
            batch_size (int, optional): The batch size for training CNNs.
                Defaults to 32.
            return_val_score (bool, optional): Return the average validation score across all folds.
                If True, the method returns the average validation score. Defaults to False.
            model_params (dict | None, optional): The model parameters to use for training. If None,
                the default parameters are used. Defaults to None.

        Returns:
            float | None: Returns the average validation score across all folds
                if `return_val_score` is True, otherwise returns None.
        """
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
            if(write_val_loss is True):
                local_writer = self.writer
            else:
                local_writer = None

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
                    writer=local_writer,
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

    def _holdout(
        self,
        val_proportion: float = 0.2,
        n_epoch: int = 20,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        return_val_score: bool = False,
        write_val_loss: bool = True,
        model_params: dict | None = None
    ) -> float | None:
        """
        Train the model using a holdout validation strategy.

        Args:
            val_proportion (float, optional): The proportion of the dataset to be used for validation.
            n_epoch (int, optional): The number of training epochs for CNNs. Defaults to 20.
            learning_rate (float, optional): The learning rate of the optimizer for CNNs. Defaults to 1e-3.
            batch_size (int, optional): The batch size for training CNNs. Defaults to 32.
            return_val_score (bool, optional): The flag to return the average validation score.
                If True, the method returns the average validation score. Defaults to False.
            model_params (dict | None, optional): The model parameters to use for training. If None,
                the default parameters are used. Defaults to None.

        Returns:
            float | None: If `return_val_score` is True, returns the average validation score.
        """
        n_aug = self.n_augmentations

        if self.seed is not None:
            np.random.seed(self.seed)
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
        if(write_val_loss is True):
            local_writer = self.writer
        else:
            local_writer = None
            
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
                writer=local_writer,
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

    def _data_split(self, data: np.ndarray, val_begin: int, val_end: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Split the data into validation and training sets.
        Makes sure to handle the augmentation correctly by
        considering the number of augmentations.

        Args:
            data (np.ndarray): The shuffled data to be split.
            val_begin (int): The starting index for the validation set.
            val_end (int): The ending index for the validation set.

        Returns:
            tuple[np.ndarray, np.ndarray]: The validation and training data.
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
        n_epoch: int = 20,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        writer: SummaryWriter | None = None,
        return_val_score: bool = False,
        model_params: dict = None
    ) -> float | None:
        """
        Train the MultiheadEmotionCNN model with the provided features and labels.

        Args:
            train_features (np.ndarray): The training features of the audio data.
            emotion_train_labels (np.ndarray): The emotion labels for the training data.
            intensity_train_labels (np.ndarray): The intensity labels for the training data.
            val_features (np.ndarray | None, optional): The validation features of the audio data.
                If None, no validation is performed. Defaults to None.
            emotion_val_labels (np.ndarray | None, optional): The emotion labels for the validation data.
                If None, no validation is performed. Defaults to None.
            intensity_val_labels (np.ndarray | None, optional): The intensity labels for the validation data.
                If None, no validation is performed. Defaults to None.
            n_epoch (int, optional): The number of training epochs for the CNN model. Defaults to 20.
            learning_rate (float, optional): The learning rate of the optimizer for the CNN model.
                Defaults to 1e-3.
            batch_size (int, optional): The batch size for training the CNN model.
                Defaults to 32.
            writer (SummaryWriter | None, optional): The TensorBoard writer for logging training metrics.
                If None, no logging is performed. Defaults to None.
            return_val_score (bool, optional): The flag to return the average validation score.
                If True, the method returns the average validation score. Defaults to False.
            model_params (dict, optional): The model parameters to use for training. If None,
                the default parameters are used. Defaults to None.

        Returns:
            float | None: If `return_val_score` is True, returns the average validation score.
        """
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
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            writer=writer,
            epochs=n_epoch,
            learning_rate=learning_rate,
            return_val_score=return_val_score
        )
        return val_score

    def _train_SVM_split(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,  # Can be emotion or intensity
        val_features: np.ndarray = None,
        val_labels: np.ndarray = None,
        return_val_score: bool = False,
        model_params: dict = None
    ) -> float | None:
        """
        Train the SVM model with the provided features and labels.
        Can train both AudioFeatureSVM and OneVsRestAudioFeatureSVM models.

        Args:
            train_features (np.ndarray): The training features of the audio data.
            train_labels (np.ndarray): The labels for the training data. Can be emotion or intensity labels.
            val_features (np.ndarray, optional): The validation features of the audio data. Defaults to None.
            val_labels (np.ndarray, optional): The labels for the validation data. Defaults to None.
            return_val_score (bool, optional): If True, returns the validation score.
                Defaults to False.
            model_params (dict, optional): The model parameters to use for training. If None,
                the default parameters are used. Defaults to None.

        Returns:
            float | None: If `return_val_score` is True, returns the validation score.
        """
        if model_params is None:
            model_params = {}

        # Flatten spectrograms for SVM input
        spec_train_flat = train_features.reshape(train_features.shape[0], -1)
        self.pca = PCA(n_components=200, random_state=self.seed)
        spec_train_reduced = self.pca.fit_transform(spec_train_flat)
        file_path = os.path.dirname(os.path.dirname(__file__))
        if isinstance(self.model, OneVsRestAudioFeatureSVM):
            joblib.dump(self.pca, os.path.join(file_path, "data", f"pca_emotion_svm_ovr.joblib"))
        elif self.task == "emotion":
            joblib.dump(self.pca, os.path.join(file_path, "data", f"pca_emotion_svm.joblib"))
        elif self.task == "intensity":
            joblib.dump(self.pca, os.path.join(file_path, "data", f"pca_intensity_svm.joblib"))
        else:
            raise ValueError("Unknown task type: must be 'emotion' or 'intensity'")

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

    def _mc_uncertainty(self, eval_data: torch.Tensor, n: int = 100) -> None:
        """
        Determine the uncertainty of the model using Monte Carlo dropout.
        This method performs multiple forward passes through the model with dropout enabled
        to estimate the uncertainty of the model's predictions.

        Args:
            eval_data (torch.Tensor): The evaluation data to be used for uncertainty estimation.
            n (int, optional): The number of forward passes to perform for uncertainty estimation.
                Defaults to 100.
        """
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

    def hyperparameter_tune(self, param_grid: list[dict], cross_val="holdout", model_name: str | None = "best_model") -> list[tuple]:
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
        Returns: 
            List with val_scores for each evaluated parameter grid point
        """
        val_scores = []
        for i, params in tqdm(enumerate(param_grid), desc="Hyperparameter tuning"):
            # Split parameters into model and training parameters
            model_params, train_params = self._filter_params(params)
            print(model_params, train_params)
            # Model parameters are used to initialize the model while training parameters
            # are used to fit the model
            if cross_val == "k_fold":
                val_score = self._k_fold(return_val_score=True, write_val_loss=False **train_params, model_params=model_params)
            elif cross_val == "holdout":
                val_score = self._holdout(return_val_score=True, write_val_loss=False,  **train_params, model_params=model_params)
            else:
                raise ValueError("Unknown cross-validation type.")
            if val_score > self.best_score:
                self.best_score = val_score
                self.best_params = params
            val_scores.append(val_score)

        print(f"Best validation score: {self.best_score} with parameters: {self.best_params}")
        # Retrain best model on all training data
        print("Retraining best model on all training data...")
        self.train_no_split()
        if model_name is not None:
            # Save the best model
            print(f"Saving best model to {model_name}.")
            self.model.save(model_name)

        return val_scores

    def evolutionary_hyper_search(
        self,
        param_template: dict[list[list]],
        cross_val:str ="holdout",
        n: int = 10,
        pop_size: int = 15,
        repro_rate: float = 0.4,
        model_name: str | None = "best_model"
    ):
        """
        Perform hyperparameter search using evolutionary optimization. 
        This method will generate an initial population of size pop_size 
        of random dictionaries and test them using the hyperparameter tune 
        method, it will then take the best performing repro_rate (reproduction)
        rate portion and reproduce those dictionaries through the _reproduce_population
        with crossover and mutation to produce a new set of dictionaries with higher 
        fitness scores. 
        Args:
            param_template: dict[list[list]], template with constraints for each parameter 
            of the dicts 
            cross_val:str, type of crossvalidation to use for evaluation. 
            n: int, number of generations to run the evolution process for
            pop_size: int, population size
            repro_rate: float, proportion of fittest parameter dicts that get 
            to reproduce
            model_name: str, name to save the best model with
        """
        population = self.generate_random_points(param_template, pop_size)

        for _ in range(n):
            fitness_scores = self.hyperparameter_tune(population, cross_val=cross_val, model_name=model_name)
            population = self._reproduce_population(population, fitness_scores, param_template, pop_size, repro_rate)
        
    def _reproduce_population(self, population: list, fitness_scores: list, param_template: dict[list[list]]
                              , goal_size: int, repro_rate: float, elitsm: bool = True) -> list[dict]:
        """Perform the crossover and mutation procedures on the selected breeding population. 
        Args:
            goal_size: int, The desired population size to be reached after reproduction through crossover
            population 
        Returns:
            list[dict]: List with newly generated population
            """
        n_best = int(repro_rate * goal_size)
        fittest_dictionaries = self.select_fitness_prob(population, fitness_scores, n_best)
        new_population = []
        mutated_population = []

        if elitsm is True: #Perform elitism (with mutation if pop size is above 10)
            mutated_population.append(fittest_dictionaries[0]) #skip mutating fittest individual
            if(goal_size >= 20):
                for _ in range(int(goal_size*0.1) - 1):
                    new_population.append(fittest_dictionaries[0].copy())
                goal_size = goal_size - int(goal_size*0.1)
            else:
                goal_size = goal_size - 1

        rnd.shuffle(population)
        
        for _ in range(int(goal_size / 2)):
            parents = rnd.sample(population, 2)
            children = self._crossover(parents[0], parents[1])
            new_population.extend(children)
            
        if goal_size % 2 != 0:
            parents = rnd.sample(population, 2)
            children = self._crossover(parents[0], parents[1])
            new_population.append(children[0])      
        
        for individual in new_population:
            mutated_population.append(self._mutate(param_template, individual))
 
        

        return new_population
    
    def _crossover(self, parent_one:dict, parent_two:dict) -> list[dict]:
        """Perform crossover operation on the parameters (genomes)
        of two parameter containing dicts (parents)
        Args:
            parent_one:dict: it is important that both parent dicts have the same keys
            parent_two:dict

        Returns:
            tuple[dict], of two dicts are identical to the parent dicts 
            except that they have the values at one location swapped
            """
        child_one = parent_one.copy()
        child_two = parent_two.copy()
        keys = list(child_one.keys())
        param_to_swap = rnd.choice(keys)
        temp_value_store = child_one[param_to_swap]
        child_one[param_to_swap] = child_two[param_to_swap]
        child_two[param_to_swap] = temp_value_store

        return [child_one, child_two]

    def _mutate(self, param_template: dict[list[list]],  parameters:dict, mutation_factor:int = 0.4) -> dict[list[list]]:
        """
        Apply mutation to one of the parameters of a dict.
        Args:
            param_template: dict[list[list]]: parameter template 
            parameters: dict: parameters to mutate
            mutation_factor: int: proportion of value mutated
        """
        parameters = parameters.copy()
        keys = list(parameters.keys())
        param_to_mutate = rnd.choice(keys)

        param_constraint = param_template[param_to_mutate][0]
        constraint_spec = param_template[param_to_mutate][1]
        #Repeated code to allow for furthe adjustment per constrait type if needed
        if param_constraint == "float_range":
            value = parameters[param_to_mutate]
            range_begin = constraint_spec[0]
            range_end = constraint_spec[1]
            sigma = (range_end + range_begin) * 0.5 * mutation_factor
            delta = rnd.uniform(-1*sigma, sigma) 
            value += delta
            value = max(range_begin, min(value, range_end))
            parameters[param_to_mutate] = value   

        elif param_constraint == "int_range":
            value = parameters[param_to_mutate]
            range_begin = constraint_spec[0]
            range_end = constraint_spec[1]
            sigma = (range_end + range_begin) * 0.5 * mutation_factor
            delta = rnd.uniform(-1*sigma, sigma) 
            value = int(value + delta)
            value = max(range_begin, min(value, range_end))
            parameters[param_to_mutate] = int(value)    

        elif param_constraint == "int_power_range": 
            base = constraint_spec[0]
            value = math.log(parameters[param_to_mutate]) / math.log(base)
            range_begin = constraint_spec[1]
            range_end = constraint_spec[2]
            sigma = (range_end + range_begin) * 0.5 * mutation_factor
            delta = rnd.uniform(-1*sigma, sigma)  
            value = int(value + delta)
            value = max(range_begin, min(value, range_end))
            parameters[param_to_mutate] = int(base ** value)

        elif param_constraint == "float_range_log": 
            #Note that the range of this one reflects the actual value of the param 
            #not the base like in power range
            value = math.log(parameters[param_to_mutate])
            range_begin = math.log(constraint_spec[0])
            range_end = math.log(constraint_spec[1])
            sigma = (range_end + range_begin) * 0.5 * mutation_factor
            delta = rnd.uniform(0, sigma) 
            value = math.exp(value + delta)
            value = max(math.exp(range_begin), min(value, math.exp(range_end)))
            parameters[param_to_mutate] = value

        return parameters 

    def generate_random_points(self, param_template: dict[list[list]], n: int = 50, sd_scale: float = 0.9) -> list[dict]:
        """Generate random points in the parameter grid 
        Args:
            param_template: dict[list[list]], contains a dictionary with constraints for each parameter
        n: int, the number of random points to generate
        Returns:
            grid_subset: list[dict] with random points on prameter grid"""
        #Note that the largly repeated code is to allow for adjustments to be made to each constraint 
        #type without the use of complex helper functions 
        grid_subset = []
        for _ in range(n):
            grid_point = {}
            for parameter in param_template.items():
                param_constraint = parameter[1][0]
                constraint_spec = parameter[1][1]
                param_name = parameter[0]
                if param_constraint == "float_range":
                    range_begin = constraint_spec[0]
                    range_end = constraint_spec[1]
                    mean = (range_begin + range_end) / 2
                    param_value = np.random.normal(mean, mean * sd_scale)
                    param_value = max(range_begin, min(param_value, range_end))
                    grid_point[param_name] = param_value
                    
                elif param_constraint == "int_range":
                    range_begin = constraint_spec[0]
                    range_end = constraint_spec[1]
                    mean = (range_begin + range_end) / 2
                    param_value = int(np.random.normal(mean, mean * sd_scale))
                    param_value = max(range_begin, min(param_value, range_end))
                    grid_point[param_name] = param_value     

                elif param_constraint == "int_power_range": 
                    base = constraint_spec[0]
                    range_begin = constraint_spec[1]
                    range_end = constraint_spec[2]
                    mean = (range_begin + range_end) / 2
                    power = int(np.random.normal(mean, mean * sd_scale))
                    power = max(range_begin, min(power, range_end))                    
                    param_value = base ** power
                    grid_point[param_name] = param_value   

                elif param_constraint == "float_range_log": 
                    #Note that the range of this one reflects the actual value of the param 
                    #not the base like in power range
                    range_begin = math.log(constraint_spec[0])
                    range_end = math.log(constraint_spec[1])
                    mean = (range_begin + range_end) / 2
                    param_value = np.random.normal(mean, abs(mean * sd_scale))
                    param_value = math.exp(param_value)
                    param_value = max(math.exp(range_begin), min(param_value, math.exp(range_end)))
                    grid_point[param_name] = param_value 

            grid_subset.append(grid_point)
        return grid_subset
    

    
    def select_fitness_prob(self, population:list[dict], fitness_scores:list, n: int) -> list[dict]:
        """
        Probabalistically select n dicts from list based on acompanying fitness scores. 
        Args:
        population:list[dict], list of dicts to select from
        fitness_scores:list, list of floats representing fitness scores
        n: int, number of dicts to choose, must be er equal to population length
        """
        selected_subset = []
        for _ in range(n):
            chosen_dict = rnd.choices(population, weights=fitness_scores, k=1)[0]
            selected_subset.append(chosen_dict)
            del fitness_scores[population.index(chosen_dict)]
            del population[population.index(chosen_dict)]
        return selected_subset

    def evaluate_on_testset(
        self,
        test_features: np.ndarray,
        emotion_test_labels: np.ndarray = None,
        intensity_test_labels: np.ndarray = None,
        title_suffix: str = ""
    ) -> None:
        """
        Evaluate the trained model on the test set.
        This method evaluates the model's performance on the test set
        and logs the results using ModelEvaluator. Both emotion and intensity
        labels are required for MultiheadEmotionCNN, while only one set of labels
        is required for AudioFeatureSVM or OneVsRestAudioFeatureSVM depending on the task.

        Args:
            test_features (np.ndarray): The features of the test set. Should be of shape
                (n_samples, n_features). For CNNs, this should be the spectrograms.
            emotion_test_labels (np.ndarray, optional): The corresponding emotion labels for the test set.
                Defaults to None.
            intensity_test_labels (np.ndarray, optional): The corresponding intensity labels for the test set.
                Defaults to None.
            title_suffix (str, optional): The suffix to be added to the evaluation title.
                Defaults to "".
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
