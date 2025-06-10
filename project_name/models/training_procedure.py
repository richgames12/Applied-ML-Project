from project_name.models.spectrogram_cnn import MultiheadEmotionCNN
from project_name.models.audio_feature_svm import AudioFeatureSVM
# Change above imports to star once this file is moved out of this dir
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from project_name.evaluation.model_evaluation import ModelEvaluator


from torch.utils.tensorboard import SummaryWriter

# quick hack
EMOTION_LABELS = {
    1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
    5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'
}


class TrainAndEval():
    def __init__(self, aug_features: np.ndarray, aug_labels: tuple, n_augmentations: int, model) -> None:
        # Do data augmentation on all data 
        self.aug_features = aug_features  # quick hack for label problem
        self.aug_labels = aug_labels[0] - 1, aug_labels[1] - 1  # The labels go from 1-9 and from 1-2, shift down by 1
        self.n_augmentations = n_augmentations 
        self.n_samples = int(aug_features.shape[0] / n_augmentations)
        self.seed = None
        self.model = model
        self.writer = SummaryWriter('logs/test')

        return None

    def train_model(self, cross_val: str):
        """
        Method that allows for training calls to the model. 

        Args:
        model pytorch model 
        cross_val: string, specifies type of cross validation to use
        features: np.ndarray, contains features to be trained and tested on
        labels: np.ndarray, contains labels for supervised training problems

        """

        # Check whether provided model is compatible with cross validation. 
        # if(not cross_val in self.compatiblity[self.model.model_type]):
        #    raise ValueError(
        #       "Cross validation type not copatible with model type."
        #    )

        spec_emotion_svm = OneVsRestClassifier(AudioFeatureSVM(
            regularization_parameter=10, seed=self.seed))

        if (cross_val == "holdout"):
            self._holdout()

    def _k_fold(model, features, k: int = 10):
        fol_obj = KFold(n_splits=k, shuffle=True)

        pass

    def _holdout(self):
        """Evaluate using holdout data split"""
        val_proportion = 0.2
        val_begin = int(self.n_samples * (1 - val_proportion))
        val_end = self.n_samples

        test_features, train_features = self._data_split(self.aug_features, val_begin, val_end)

        emote_test_labels, emote_train_labels = self._data_split(self.aug_labels[0], val_begin, val_end)
        intens_test_labels, intens_train_labels = self._data_split(self.aug_labels[1], val_begin, val_end)

        if (isinstance(self.model, MultiheadEmotionCNN)):
            self._train_CNN_split(train_features, test_features, emote_train_labels, emote_test_labels, intens_train_labels, intens_test_labels)
            self._write_confusion_matrix(test_features, emote_test_labels, intens_test_labels)

        elif (isinstance(self.model, AudioFeatureSVM)):
            self._train_SVM_split()

    def _data_split(self, data: np.ndarray, val_begin: int, val_end: int) -> tuple:
        val_begin *= self.n_augmentations
        val_end *= self.n_augmentations
        val_data = data[val_begin:val_end]
        val_data = val_data[::(self.n_augmentations)]
        train_data = np.concatenate((data[:val_begin], data[val_end:]))
        return val_data, train_data  

    def _train_CNN_split(self, train_features: np.ndarray, test_features: np.ndarray, 
                         emote_train_labels: np.ndarray, emote_test_labels: np.ndarray, intens_train_labels: np.ndarray, intens_test_labels: np.ndarray, n_epoch: int = 5) -> None:
        """Train cnn"""

        emote_test_labels = torch.from_numpy(emote_test_labels)
        intens_test_labels = torch.from_numpy(intens_test_labels)
        emote_train_labels = torch.from_numpy(emote_train_labels)
        intens_train_labels = torch.from_numpy(intens_train_labels)
        train_features = torch.tensor(train_features, dtype=torch.float32)
        test_features = torch.tensor(test_features, dtype=torch.float32)       

        train_dataset = TensorDataset(train_features, emote_train_labels, intens_train_labels)
        train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        test_dataset = TensorDataset(test_features, emote_test_labels, intens_test_labels)
        test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True)

        self.model.cross_val_fit(test_dataloader, train_dataloader, self.writer, n_epoch)

        return None

    def _train_SVM_split(self, train_features: np.ndarray, test_features: np.ndarray, test_labels: np.ndarray) -> None:

        spec_train_flat = train_features.reshape(train_features.shape[0], -1)
        spec_test_flat = test_features.reshape(test_features.shape[0], -1)
        # Reduce feature dimensionality to improve SVM efficiency
        pca = PCA(n_components=200, random_state=self.seed)
        spec_train_reduced = pca.fit_transform(spec_train_flat)
        spec_test_reduced = pca.transform(spec_test_flat)

        # Train spectrogram-based emotion SVM
        print("Training Spectrogram-based Emotion SVM.")

        self.modelmodel.fit(spec_train_reduced, train_labels)
        print("Spectrogram-based Emotion SVM trained.")

        return None

    def _mc_uncertainty(self, eval_data, n=100):
        # currently for a single sample
        self.model.train() # to enable dropout, because no grad or step this wont update weights
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

    def _write_confusion_matrix(self, eval_data, emote_ground_truth, intens_ground_truth):
        eval_data = torch.tensor(eval_data, dtype=torch.float32)   
        predictions = self.model.predict(eval_data)
        evaluator = ModelEvaluator(EMOTION_LABELS)
        evaluator.evaluate_from_predictions(emote_ground_truth + 1, (predictions[0].numpy() + 1))  # quick hack to solve class indexing

    def eval_metric(self, metric: str, ground_truth: np.ndarray, predictions: np.ndarray):
        """
        Calculate metric score:
        Args:
        metric: str, specifier of metric to calculate
        ground_truth: np.ndarray, true labels
        predictions: np.ndarrraym, 
        """
        pass  # for now
