from download_dataset import DATASET_DIR
from project_name.data.data_file_splitting import DataFileSplitter
from project_name.data.data_preprocessing import AudioPreprocessor
from project_name.data.data_augmentation import RawAudioAugmenter
from project_name.features.audio_feature_extractor import AudioFeatureExtractor
from project_name.models.audio_feature_svm import AudioFeatureSVM
from sklearn.multiclass import OneVsRestClassifier
from project_name.evaluation.model_evaluation import ModelEvaluator

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

    # ____________________________________________
    #                 Data Loading
    # ____________________________________________
    # Initialize the data file splitter
    splitter = DataFileSplitter(dataset_path=DATASET_DIR, seed=seed)
    train_data, val_data, test_data = splitter.get_data_splits_copy()

    # ____________________________________________
    #              Data Preprocessing
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
    #             Feature Extraction
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
    #               Model Training
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

    # ____________________________________________
    #              Model Evaluation
    # ____________________________________________

    if test_features.shape[0] > 0:
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
    else:
        print("No test data available for evaluation.")
