from download_dataset import DATASET_DIR
from project_name.data.data_file_splitting import DataFileSplitter
from project_name.data.data_preprocessing import AudioPreprocessor
from project_name.data.data_augmentation import (
    RawAudioAugmenter, SpectrogramAugmenter
)
from project_name.features.audio_feature_extractor import AudioFeatureExtractor
from project_name.models.audio_feature_svm import AudioFeatureSVM
from project_name.models.spectrogram_cnn import MultiheadEmotionCNN
from project_name.models.training_procedure import TrainAndEval
from sklearn.multiclass import OneVsRestClassifier
from project_name.evaluation.model_evaluation import ModelEvaluator
from sklearn.decomposition import PCA

import numpy as np
import random


EMOTION_LABELS = {
    1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
    5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'
}
INTENSITY_LABELS = {0: 'normal', 1: 'strong'}

N_SPEC_AUGMENTATIONS = 3


if __name__ == "__main__":
    seed = None
    # Some classes also use a seed to control sklearn random processes
    np.random.seed(seed)
    random.seed(seed)

    # Toggle evaluation
    evaluate_mfcc = True
    evaluate_spec = True

    # ____________________________________________
    #                 Data Loading (MFCC)
    # ____________________________________________
    # Initialize the data file splitter
    splitter = DataFileSplitter(dataset_path=DATASET_DIR, test_size=0.1, eval_size=0.1, seed=seed)
    train_data, val_data, test_data = splitter.get_data_splits_copy()

    # ____________________________________________
    #         Spectrogram Preprocessing
    # ____________________________________________
    spectrogram_augmenter = SpectrogramAugmenter(
        freq_mask_prob=0, time_mask_prob=0, noise_std=0.5
    )

    # Initialize the spectrogram-based preprocessor
    spectrogram_preprocessor = AudioPreprocessor(
        spectrogram_augmenter=spectrogram_augmenter,
        use_spectrograms=True,
        n_augmentations=N_SPEC_AUGMENTATIONS
    )

    # Process spectrogram-based training and test data
    spec_train_data, spec_train_emotion_labels, spec_train_intensity_labels = \
        spectrogram_preprocessor.process_all(train_data)  # For quick testing

    print("Spectrogram training data processed.")

    # 🔧 Convert labels to correct type (np.int64 → torch.long later)
    spec_train_emotion_labels = spec_train_emotion_labels.astype(np.int64)
    spec_train_intensity_labels = spec_train_intensity_labels.astype(np.int64)

    # Flatten spectrograms for SVM input (not used here but kept for completeness)
    spec_train_flat = spec_train_data.reshape(spec_train_data.shape[0], -1)

    # ____________________________________________
    #              CNN eval test
    # ____________________________________________
    multi_task_cnn = MultiheadEmotionCNN()
    all_labels = (spec_train_emotion_labels, spec_train_intensity_labels)
    eval_obj = TrainAndEval(spec_train_data, all_labels, N_SPEC_AUGMENTATIONS, multi_task_cnn)
    eval_obj.train_and_eval_model("holdout")
