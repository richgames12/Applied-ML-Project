from download_dataset import DATASET_DIR
from project_name.data.data_file_splitting import DataFileSplitter
from project_name.data.data_preprocessing import AudioPreprocessor
from project_name.data.data_augmentation import (
    RawAudioAugmenter, SpectrogramAugmenter
)
from project_name.models.one_vs_rest import OneVsRestAudioFeatureSVM
from project_name.features.audio_feature_extractor import AudioFeatureExtractor
from project_name.models.audio_feature_svm import AudioFeatureSVM
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
    splitter = DataFileSplitter(dataset_path=DATASET_DIR, seed=seed)
    train_data, val_data, test_data = splitter.get_data_splits_copy()

    # ____________________________________________
    #              Data Preprocessing (MFCC)
    # ____________________________________________

    # Initialize the raw audio augmenter
    augmenter = RawAudioAugmenter(pitch_probability=0)

    # Initialize the audio preprocessor
    preprocessor = AudioPreprocessor(
        data_augmenter=augmenter, n_augmentations=2
    )

    # Process the training data
    train_processed, train_emotion_labels, train_intensity_labels = \
        preprocessor.process_all(train_data)
    print("Training data processed.")

    # Preprocess the test data
    preprocessor.data_augmenter = None  # Test data should not be augmented
    test_processed, test_emotion_labels, test_intensity_labels = \
        preprocessor.process_all(test_data)
    print("Test data processed.")

    # ____________________________________________
    #             Feature Extraction (MFCC)
    # ____________________________________________

    # Initialize the audio feature extractor
    feature_extractor = AudioFeatureExtractor(use_deltas=True, n_mfcc=20)

    # Extract features from the processed training data
    train_features = feature_extractor.extract_features_all(train_processed)
    print("Training features extracted.")

    # Extract features from the processed test data
    test_features = feature_extractor.extract_features_all(test_processed)
    print("Test features extracted.")

    # ____________________________________________
    #               Model Training (MFCC)
    # ____________________________________________

    # Shuffle the data before training the SVM
    indices = np.arange(len(train_features))
    np.random.shuffle(indices)
    train_features = train_features[indices]
    train_emotion_labels = train_emotion_labels[indices]
    train_intensity_labels = train_intensity_labels[indices]

    # Initialize and train the SVM for emotion recognition
    # Use a OneVsRest version to increase the models accuracy
    print("Training Emotion MFCC SVM.")
    mfcc_emotion_svm = OneVsRestAudioFeatureSVM(
        regularization_parameter=10, seed=seed
    )
    mfcc_emotion_svm.fit(train_features, train_emotion_labels)
    print("Emotion MFCC SVM trained.")

    mfcc_emotion_svm.save(model_name="emotion_svm")

    print("Training Intensity MFCC SVM.")
    # Only two classes so no OneVsRest
    mfcc_intensity_svm = AudioFeatureSVM(
        regularization_parameter=10, seed=seed
    )
    mfcc_intensity_svm.fit(train_features, train_intensity_labels)
    print("Intensity MFCC SVM trained.")

    mfcc_intensity_svm.save(model_name="intensity_svm.joblib")

    # ____________________________________________
    #         Spectrogram Preprocessing
    # ____________________________________________
    # Masks don't seem to work really well with PCA
    spectrogram_augmenter = SpectrogramAugmenter(
        freq_mask_prob=0, time_mask_prob=0, noise_std=0.5
    )

    # Initialize the spectrogram-based preprocessor
    spectrogram_preprocessor = AudioPreprocessor(
        spectrogram_augmenter=spectrogram_augmenter,
        use_spectrograms=True,
        n_augmentations=3
    )

    # Process spectrogram-based training and test data
    spec_train_data, spec_train_emotion_labels, spec_train_intensity_labels = \
        spectrogram_preprocessor.process_all(train_data)
    print("Spectrogram training data processed.")

    spec_test_data, spec_test_emotion_labels, spec_test_intensity_labels = \
        spectrogram_preprocessor.process_all(test_data)
    print("Spectrogram test data processed.")

    # Flatten spectrograms for SVM input
    spec_train_flat = spec_train_data.reshape(spec_train_data.shape[0], -1)
    spec_test_flat = spec_test_data.reshape(spec_test_data.shape[0], -1)

    # ____________________________________________
    #       Dimensionality Reduction (PCA)
    # ____________________________________________

    # Reduce feature dimensionality to improve SVM efficiency
    pca = PCA(n_components=200, random_state=seed)
    spec_train_reduced = pca.fit_transform(spec_train_flat)
    spec_test_reduced = pca.transform(spec_test_flat)

    # Train spectrogram-based emotion SVM
    print("Training Spectrogram-based Emotion SVM.")
    spec_emotion_svm = OneVsRestAudioFeatureSVM(
        regularization_parameter=10, seed=seed
    )
    spec_emotion_svm.fit(spec_train_reduced, spec_train_emotion_labels)
    print("Spectrogram-based Emotion SVM trained.")

    spec_emotion_svm.save(model_name="spectogram_emotion_svm")

    # Train spectrogram-based intensity SVM
    print("Training Spectrogram-based Intensity SVM.")
    spec_intensity_svm = AudioFeatureSVM(
        regularization_parameter=10, seed=seed)
    spec_intensity_svm.fit(spec_train_reduced, spec_train_intensity_labels)
    print("Spectrogram-based Intensity SVM trained.")

    spec_intensity_svm.save(model_name="spectogram_intensity_svm.joblib")

    # ____________________________________________
    #              Model Evaluation MFCC
    # ____________________________________________

    # Intitialize the evaluators
    emotion_evaluator = ModelEvaluator(class_labels=EMOTION_LABELS)
    intensity_evaluator = ModelEvaluator(class_labels=INTENSITY_LABELS)

    if evaluate_mfcc and test_features.shape[0] > 0:
        print("Evaluating MFCC Emotion Model")
        # Step 1 Predict labels and compute softmax-based probability estimates
        pred_emotion_labels = mfcc_emotion_svm.predict(test_features)
        probs_mfcc_emotion = mfcc_emotion_svm.predict_proba(test_features)

        # Step 2 Evaluate the predictions
        emotion_evaluator.evaluate_uncertainty_metrics(
            class_probabilities=probs_mfcc_emotion,
            title_suffix="MFCC emotion SVM"
        )
        emotion_evaluator.evaluate_from_predictions(
            labels_true=test_emotion_labels,
            labels_pred=pred_emotion_labels,
            title_suffix="Emotion Recognition SVM MFCC"
        )

        print("Evaluating Intensity MFCC Model")
        # Step 1 Predict labels and compute softmax-based probability estimates
        pred_intesity_labels = mfcc_intensity_svm.predict(test_features)
        probs_mfcc_intensity = mfcc_intensity_svm.predict_proba(test_features)

        # Step 2 Evaluate the predictions
        intensity_evaluator.evaluate_uncertainty_metrics(
            class_probabilities=probs_mfcc_intensity,
            title_suffix="MFCC intensity SVM"
        )
        intensity_evaluator.evaluate_from_predictions(
            labels_true=test_intensity_labels,
            labels_pred=pred_intesity_labels,
            title_suffix="Intensity Recognition SVM MFCC"
        )

    if evaluate_spec and spec_test_flat.shape[0] > 0:
        print("Evaluating Spectrogram Emotion Model")
        # Step 1 Predict labels and compute softmax-based probability estimates
        pred_spec_emotion = spec_emotion_svm.predict(spec_test_reduced)
        probs_spec_emotion = spec_emotion_svm.predict_proba(spec_test_reduced)

        # Step 2 Evaluate the predictions
        emotion_evaluator.evaluate_uncertainty_metrics(
            class_probabilities=probs_spec_emotion,
            title_suffix="Spectrogram emotion SVM"
        )
        emotion_evaluator.evaluate_from_predictions(
            labels_true=spec_test_emotion_labels,
            labels_pred=pred_spec_emotion,
            title_suffix="Spectrogram-based Emotion SVM"
        )

        print("Evaluating Spectrogram Intensity Model")
        # Step 1 Predict labels and compute softmax-based probability estimates
        pred_spec_intensity = spec_intensity_svm.predict(spec_test_reduced)
        probs_spec_intensity = spec_intensity_svm.predict_proba(
            spec_test_reduced
        )

        # Step 2 Evaluate the predictions
        intensity_evaluator.evaluate_uncertainty_metrics(
            class_probabilities=probs_spec_intensity,
            title_suffix="Spectrogram intensity SVM"
        )
        intensity_evaluator.evaluate_from_predictions(
            labels_true=spec_test_intensity_labels,
            labels_pred=pred_spec_intensity,
            title_suffix="Spectrogram-based Intensity SVM"
        )

    if not (evaluate_mfcc or evaluate_spec):
        print("Model evaluation skipped.")
