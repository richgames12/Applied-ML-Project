from download_dataset import DATASET_DIR
from project_name.data.data_file_splitting import DataFileSplitter
from project_name.data.data_preprocessing import AudioPreprocessor
from project_name.data.data_augmentation import RawAudioAugmenter
from project_name.features.audio_feature_extractor import AudioFeatureExtractor
from project_name.models.audio_feature_svm import AudioFeatureSVM
from project_name.models.spectrogram_cnn import MultiheadEmotionCNN
from project_name.models.spectrogram_cnn import fit_model
from sklearn.multiclass import OneVsRestClassifier

import torch
import numpy as np
import random
from torch.utils.data import DataLoader, TensorDataset

if __name__ == "__main__":
    seed = None
    # Some classes also use a seed to control sklearn random processes
    np.random.seed(seed)
    random.seed(seed)

    # Initialize the data file splitter
    splitter = DataFileSplitter(dataset_path=DATASET_DIR, seed=seed)
    train_data, val_data, test_data = splitter.get_data_splits_copy()

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

    # Initialize the audio feature extractor
    feature_extractor = AudioFeatureExtractor(use_deltas=True, n_mfcc=20)

    # Extract features from the processed training data
    train_features = feature_extractor.extract_features_all(train_processed)
    print("Training features extracted.")



    #Create tensor dataset to be used with dataloader 
    dataset = TensorDataset(train_features, train_emotion_labels, train_intensity_labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    #Model train test
    CNN = MultiheadEmotionCNN()
    fit_model(CNN, dataloader, 10)


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
    multiclass_emotion_svm = OneVsRestClassifier(base_emotion_svm)
    multiclass_emotion_svm.fit(train_features, train_emotion_labels)
    print("Emotion SVM trained.")

    # Process the test data
    preprocessor.data_augmenter = None  # Test data should not be augmented
    test_processed, test_emotion_labels, test_intensity_labels = \
        preprocessor.process_all(test_data)
    print("Test data processed.")

    # Extract features from the processed test data
    test_features = feature_extractor.extract_features_all(test_processed)
    print("Test features extracted.")

    if test_features.shape[0] > 0:
        emotion_accuracy = multiclass_emotion_svm.score(
            test_features, test_emotion_labels
        )
        print(
            "Emotion recognition accuracy on the test set:",
            f"{emotion_accuracy:.4f}"
        )
    else:
        print("No test data available for evaluation.")


    #C