from download_dataset import DATASET_DIR
from project_name.data.data_file_splitting import DataFileSplitter
from project_name.data.data_preprocessing import AudioPreprocessor
from project_name.data.data_augmentation import SpectrogramAugmenter
from project_name.models.audio_feature_svm import AudioFeatureSVM
from project_name.models.spectrogram_cnn import MultiheadEmotionCNN
from project_name.models.training_procedure import TrainAndEval

import numpy as np
import random

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
        spectrogram_preprocessor.process_all(train_data)

    print("Spectrogram training data processed.")

    # Convert labels to correct type (np.int64 → torch.long later)
    # Otherwise, PyTorch will throw an error
    spec_train_emotion_labels = spec_train_emotion_labels.astype(np.int64)
    spec_train_intensity_labels = spec_train_intensity_labels.astype(np.int64)
    all_labels = (spec_train_emotion_labels, spec_train_intensity_labels)

    # ____________________________________________
    #        CNN training and evaluation
    # ____________________________________________
    #multi_task_cnn = MultiheadEmotionCNN()
    #eval_obj = TrainAndEval(spec_train_data, all_labels, N_SPEC_AUGMENTATIONS, multi_task_cnn)
    #eval_obj.train_and_eval_model("k_fold")

    # ____________________________________________
    #    SVM training and evalutation (emotion)
    # ____________________________________________
    emotion_svm = AudioFeatureSVM()
    # Pass all labels for consistency, even if we only use emotion labels
    eval_obj_emotion_svm = TrainAndEval(spec_train_data, all_labels, N_SPEC_AUGMENTATIONS, emotion_svm, task="emotion")
    eval_obj_emotion_svm.train_and_eval_model()

    # ____________________________________________
    #   SVM training and evalutation (intensity)
    # ____________________________________________
    intensity_svm = AudioFeatureSVM()
    # Pass all labels for consistency, even if we only use emotion labels
    eval_obj_intensity_svm = TrainAndEval(spec_train_data, all_labels, N_SPEC_AUGMENTATIONS, intensity_svm, task="intensity")
    eval_obj_intensity_svm.train_and_eval_model()

    # ____________________________________________
    #        Final test set evaluation (CNN)
    # ____________________________________________
    # Process test data for CNN
    #spectrogram_preprocessor.spectrogram_augmenter = None  # Disable augmentation for test set
    #spec_test_data, spec_test_emotion_labels, spec_test_intensity_labels = \
    #    spectrogram_preprocessor.process_all(test_data)

    # Adjust labels to start from 0
    # Also convert labels to correct type (np.int64 will be torch.long later)
    #spec_test_emotion_labels = spec_test_emotion_labels.astype(np.int64) - 1
    #spec_test_intensity_labels = spec_test_intensity_labels.astype(np.int64) - 1
    #print("Spectrogram test data processed.")

    # Evaluate the trained CNN on the true test set
    #eval_obj.evaluate_on_testset(
    #    spec_test_data,
    #    spec_test_emotion_labels,
    #    spec_test_intensity_labels
    #)

    # ____________________________________________
    #        Final test set evaluation (SVM)
    # ____________________________________________
    spectrogram_preprocessor.spectrogram_augmenter = None  # Disable augmentation for test set
    spec_test_data, spec_test_emotion_labels, spec_test_intensity_labels = spectrogram_preprocessor.process_all(test_data)

    # Emotion SVM evaluation
    spec_test_emotion_labels = spec_test_emotion_labels.astype(np.int64) - 1
    eval_obj_emotion_svm.evaluate_on_testset(spec_test_data, spec_test_emotion_labels)

    # Intensity SVM evaluation
    spec_test_intensity_labels = spec_test_intensity_labels.astype(np.int64) - 1
    eval_obj_intensity_svm.evaluate_on_testset(spec_test_data, intensity_test_labels=spec_test_intensity_labels)
