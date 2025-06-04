from download_dataset import DATASET_DIR
from project_name.data.data_file_splitting import DataFileSplitter
from project_name.data.data_preprocessing import AudioPreprocessor
from project_name.data.data_augmentation import RawAudioAugmenter
from project_name.features.audio_feature_extractor import AudioFeatureExtractor
from project_name.models.audio_feature_svm import AudioFeatureSVM
from sklearn.multiclass import OneVsRestClassifier
from project_name.evaluation.model_evaluation import ModelEvaluator
from sklearn.decomposition import PCA
from scipy.special import softmax
import numpy as np
from sklearn.multiclass import OneVsRestClassifier

def predict_softmax_proba(model: OneVsRestClassifier, X: np.ndarray) -> np.ndarray:
    """
    Computes softmax-based class probability estimates from a trained
        OneVsRestClassifier using its decision function scores.

    Args:
        model (OneVsRestClassifier): A trained OneVsRestClassifier model
            wrapping an SVM (with or without probability=True).
        X (np.ndarray): Test data input features (2D array of shape
            [n_samples, n_features]).

    Returns:
        np.ndarray: A 2D array of softmax-transformed decision scores,
            representing class probabilities per sample.
            Shape: [n_samples, n_classes].
            Example: np.array([[0.1, 0.7, 0.2], [0.4, 0.3, 0.3]])
    """
    decision_scores = model.decision_function(X)  # shape: (n_samples, n_classes)
    softmax_probs = softmax(decision_scores, axis=1)
    return softmax_probs

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
    base_emotion_svm = AudioFeatureSVM(
        probability=True, regularization_parameter=10, seed=seed
    )

    base_intensity_svm = AudioFeatureSVM(
        probability=False, regularization_parameter=10, seed=seed
    )

    print("Training Emotion SVM.")
    multiclass_emotion_svm = OneVsRestClassifier(base_emotion_svm)
    multiclass_emotion_svm.fit(train_features, train_emotion_labels)
    print("Emotion SVM trained.")

    print("Training Intensity SVM.")
    # Only two classes so no OneVsRest
    base_intensity_svm.fit(train_features, train_intensity_labels)
    print("Intensity SVM trained.")

    base_intensity_svm.save(model_name="intensity_svm.joblib")

    # ____________________________________________
    #         Spectrogram Feature Extraction
    # ____________________________________________

    # Initialize the spectrogram-based preprocessor
    spectrogram_preprocessor = AudioPreprocessor(
        data_augmenter=None, use_spectrograms=True, n_augmentations=0
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
    pca = PCA(n_components=200)
    spec_train_reduced = pca.fit_transform(spec_train_flat)
    spec_test_reduced = pca.transform(spec_test_flat)

    # Train spectrogram-based emotion SVM
    print("Training Spectrogram-based Emotion SVM.")
    spec_emotion_svm = OneVsRestClassifier(AudioFeatureSVM(
        probability=True, regularization_parameter=10, seed=seed))
    spec_emotion_svm.fit(spec_train_reduced, spec_train_emotion_labels)
    print("Spectrogram-based Emotion SVM trained.")

    # Train spectrogram-based intensity SVM
    print("Training Spectrogram-based Intensity SVM.")
    spec_intensity_svm = AudioFeatureSVM(
        probability=False, regularization_parameter=10, seed=seed
    )
    spec_intensity_svm.fit(spec_train_reduced, spec_train_intensity_labels)
    print("Spectrogram-based Intensity SVM trained.")

    # ____________________________________________
    #              Model Evaluation
    # ____________________________________________

    if evaluate_mfcc and test_features.shape[0] > 0:
        print("Evaluating Emotion Model")
        pred_emotion_labels = multiclass_emotion_svm.predict(test_features)

        # Initialize the emotion evaluator
        emotion_evaluator = ModelEvaluator(class_labels=EMOTION_LABELS)
        emotion_evaluator.evaluate_from_predictions(
            labels_true=test_emotion_labels,
            labels_pred=pred_emotion_labels,
            title_suffix="Emotion Recognition SVM"
        )

        print("Evaluating Intensity Model")
        pred_intesity_labels = base_intensity_svm.predict(test_features)

        # Initialize the intensity evaluator
        intensity_evaluator = ModelEvaluator(class_labels=INTENSITY_LABELS)
        intensity_evaluator.evaluate_from_predictions(
            labels_true=test_intensity_labels,
            labels_pred=pred_intesity_labels,
            title_suffix="Intensity Recognition SVM"
        )

    if evaluate_spec and spec_test_flat.shape[0] > 0:
        print("Evaluating Spectrogram Emotion Model")
        pred_spec_emotion = spec_emotion_svm.predict(spec_test_reduced)

        # Compute softmax-based probability estimates
        probs = predict_softmax_proba(spec_emotion_svm, spec_test_reduced)
        # Compute uncertainty metrics
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
        confidence = np.max(probs, axis=1)

        # Print average uncertainty stats (optional)
        print(f"Average entropy: {np.mean(entropy):.4f}")
        print(f"Average confidence: {np.mean(confidence):.4f}")

        # Initialize the spectrogram-based emotion evaluator
        spec_emotion_evaluator = ModelEvaluator(class_labels=EMOTION_LABELS)
        spec_emotion_evaluator.evaluate_from_predictions(
            labels_true=spec_test_emotion_labels,
            labels_pred=pred_spec_emotion,
            title_suffix="Spectrogram-based Emotion SVM"
        )

        print("Evaluating Spectrogram Intensity Model")
        pred_spec_intensity = spec_intensity_svm.predict(spec_test_reduced)

        # Initialize the spectrogram-based intensity evaluator
        spec_intensity_evaluator = ModelEvaluator(class_labels=INTENSITY_LABELS)
        spec_intensity_evaluator.evaluate_from_predictions(
            labels_true=spec_test_intensity_labels,
            labels_pred=pred_spec_intensity,
            title_suffix="Spectrogram-based Intensity SVM"
        )

    if not (evaluate_mfcc or evaluate_spec):
        print("Model evaluation skipped.")
