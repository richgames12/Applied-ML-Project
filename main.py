from download_dataset import DATASET_DIR
from project_name.data.data_file_splitting import DataFileSplitter
from project_name.data.data_preprocessing import AudioPreprocessor
from project_name.data.data_augmentation import SpectrogramAugmenter
from project_name.models.audio_feature_svm import AudioFeatureSVM
from project_name.models.one_vs_rest import OneVsRestAudioFeatureSVM
from project_name.models.spectrogram_cnn import MultiheadEmotionCNN
from project_name.models.training_procedure import TrainAndEval
import torch

import numpy as np
import random

N_SPEC_AUGMENTATIONS = 3

if __name__ == "__main__":
    print(torch.cuda.is_available())
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
    splitter = DataFileSplitter(dataset_path=DATASET_DIR, test_size=0.1, seed=seed)
    train_data, test_data = splitter.get_data_splits_copy()

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
        spectrogram_preprocessor.process_all(train_data)

    print("Spectrogram training data processed.")

    # Convert labels to correct type (np.int64 → torch.long later)
    # Otherwise, PyTorch will throw an error
    spec_train_emotion_labels = spec_train_emotion_labels.astype(np.int64)
    spec_train_intensity_labels = spec_train_intensity_labels.astype(np.int64)
    all_labels = (spec_train_emotion_labels, spec_train_intensity_labels)

    # _____________________________________________
    #   Paramater Grids for Hyperparameter Tuning
    # _____________________________________________
    # This grid should be made automatically but this is a simple example
    # More paramaters can be added but then it should also be adapted in the
    # filter function of the TrainAndEval class and also be implemented in the CNN class
    cnn_param_grid = [
        {
            "dropout_rate": 0.3,
            "learning_rate": 1e-3,
            "batch_size": 32,
            "n_epoch": 20,
        },
        {
            "dropout_rate": 0.5,
            "learning_rate": 5e-4,
            "batch_size": 64,
            "n_epoch": 30,
        },
    ]
    svm_param_grid = [
        {"kernel": "linear", "regularization_parameter": 1.0, "max_iter": 10000},
        {"kernel": "rbf", "regularization_parameter": 0.5, "gamma": 0.01, "max_iter": 10000},
    ]

    # ____________________________________________
    #    CNN training (emotion and intensity)
    # ____________________________________________

    multi_task_cnn = MultiheadEmotionCNN()
    eval_obj_cnn = TrainAndEval(spec_train_data, all_labels, N_SPEC_AUGMENTATIONS, multi_task_cnn)
    eval_obj_cnn.hyperparameter_tune(cnn_param_grid, "holdout", model_name="spectrogram_cnn")

    # ____________________________________________
    #           SVM training (emotion)
    # ____________________________________________

    # An example of using AudioFeatureSVM
    emotion_svm = AudioFeatureSVM()
    # Pass all labels for consistency, even if we only use emotion labels
    eval_obj_emotion_svm = TrainAndEval(spec_train_data, all_labels, N_SPEC_AUGMENTATIONS, emotion_svm, task="emotion")
    eval_obj_emotion_svm.hyperparameter_tune(svm_param_grid, cross_val="k_fold", model_name="emotion_svm")

    # An example of using OneVsRestAudioFeatureSVM
    emotion_svm_ovr = OneVsRestAudioFeatureSVM()
    # Pass all labels for consistency, even if we only use emotion labels
    eval_obj_emotion_svm_ovr = TrainAndEval(spec_train_data, all_labels, N_SPEC_AUGMENTATIONS, emotion_svm_ovr, task="emotion")
    eval_obj_emotion_svm_ovr.hyperparameter_tune(svm_param_grid, cross_val="holdout", model_name="emotion_svm_ovr")

    # ____________________________________________
    #          SVM training (intensity)
    # ____________________________________________

    # An example of using AudioFeatureSVM for intensity classification
    intensity_svm = AudioFeatureSVM()
    # Pass all labels for consistency, even if we only use emotion labels
    eval_obj_intensity_svm = TrainAndEval(spec_train_data, all_labels, N_SPEC_AUGMENTATIONS, intensity_svm, task="intensity")
    eval_obj_intensity_svm.hyperparameter_tune(svm_param_grid, cross_val="k_fold", model_name="intensity_svm")

    # ____________________________________________
    #         Final test set evaluation
    # ____________________________________________

    # Process test data
    spectrogram_preprocessor.spectrogram_augmenter = None  # Disable augmentation for test set
    spec_test_data, spec_test_emotion_labels, spec_test_intensity_labels = \
        spectrogram_preprocessor.process_all(test_data)

    # Adjust labels to start from 0
    # Also convert labels to correct type (np.int64 will be torch.long later)
    spec_test_emotion_labels = spec_test_emotion_labels.astype(np.int64) - 1
    spec_test_intensity_labels = spec_test_intensity_labels.astype(np.int64) - 1

    # CNN evaluation
    eval_obj_cnn.evaluate_on_testset(
        spec_test_data,
        spec_test_emotion_labels,
        spec_test_intensity_labels,
        "spectrogram_cnn"
    )

    # Emotion SVM evaluation
    eval_obj_emotion_svm.evaluate_on_testset(spec_test_data, emotion_test_labels=spec_test_emotion_labels, title_suffix="Emotion SVM")

    # One-vs-Rest Emotion SVM evaluation
    eval_obj_emotion_svm_ovr.evaluate_on_testset(spec_test_data, emotion_test_labels=spec_test_emotion_labels, title_suffix="One-vs-Rest Emotion SVM")

    # Intensity SVM evaluation
    eval_obj_intensity_svm.evaluate_on_testset(spec_test_data, intensity_test_labels=spec_test_intensity_labels, title_suffix="Intensity SVM")
