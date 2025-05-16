import librosa
import numpy as np
from project_name.data.data_augmentation import RawAudioAugmenter
from tqdm import tqdm


class AudioPreprocessor:
    def __init__(
            self,
            sampling_rate: int = 22050,
            target_length: int = 66150,
            data_augmenter: RawAudioAugmenter | None = None,
            use_spectrograms: bool = False
    ) -> None:
        """Initialize the audio preprocessor.

        Args:
            sampling_rate (int, optional): The sampling rate of the audio data.
                Defaults to 22050.
            target_length (int, optional): How long the audio data should be.
                Defaults to (3*22050 = 66150).
            data_augmenter (RawAudioAugmenter | None, optional): The augmenter
                that is to be used. If no augmenter is specified, there will be
                no augmentation. Defaults to None.
            use_spectrograms (bool, optional): Whether the preprocessor should
                use spectrograms or raw data. Defaults to False.
        """
        self.sampling_rate = sampling_rate
        self.target_length = target_length
        self.data_augmenter = data_augmenter
        self.use_spectrograms = use_spectrograms

    def process_all(
        self, file_path_label_pairs: list[tuple[str, tuple[int, int]]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load and preprocess audio data from (filepath, label) pairs.

        Args:
            file_path_label_pairs (List[Tuple[str, Tuple[int, int]]]): A list
                of tuples, where each tuple contains the file path to the audio
                data and its corresponding (emotion, intensity) label.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The processed data
            (raw or spectrogram), emotion labels, and intensity labels.
        """
        processed_data = []
        emotion_labels = []
        intensity_labels = []
        for fp, (emotion, intensity) in tqdm(file_path_label_pairs,
                                             desc="Preprocessing audio files"):
            data = self._process_single_file(fp)
            if data is not None:  # Only append succesfully loaded audio
                processed_data.append(data)
                emotion_labels.append(emotion)
                intensity_labels.append(intensity)
        return (np.array(processed_data),
                np.array(emotion_labels),
                np.array(intensity_labels))

    def _process_single_file(
            self, file_path: str
    ) -> tuple[np.ndarray, int, int] | None:
        """Load and preprocess the audio and extract the labels.

        Args:
            file_path (str): The path to where the audio file is stored.

        Returns:
            tuple[np.ndarray, int, int] | None: The preprocessed audio
                followed by the corresponding emotion and intensity labels.
        """
        raw_audio = self._load_audio(file_path)
        if raw_audio is None:
            return None
        # Process the raw data
        augmented_raw_audio = self._augment_data(raw_audio)
        length_standardized_audio = self._standardize_raw_length(
            augmented_raw_audio
        )
        return length_standardized_audio

    def _load_audio(self, file_path: str) -> np.ndarray | None:
        """
        Load the data of a single audio file.

        Args:
            file_path (str): The file path to the audio file.

        Returns:
            np.ndarray | None: The retrieved raw audio data.
        """
        try:
            audio, _ = librosa.load(file_path, sr=self.sampling_rate)
            return audio
        except Exception as e:
            print(f"Error loading audio file '{file_path}': {e}")
            return None

    def _augment_data(self, data: np.ndarray) -> np.ndarray:
        """Augment either the raw audio or spectogram.

        Args:
            data (np.ndarray): Either the raw audio when a raw audio augmenter
                is given or a spectrogram when a spectrogram augmenter is
                given. The right augmenter should be provided.

        Returns:
            np.ndarray: The augmented raw audio/ spectrogram.
        """
        if self.data_augmenter:
            return self.data_augmenter.augment_raw_file(data)
        return data

    def _standardize_raw_length(self, audio: np.ndarray) -> np.ndarray:
        """
        Pad or trim the raw audio data to make it the right lenght.

        Args:
            audio (np.ndarray): The raw audio for which the length needs to be
                standardized.

        Returns:
            np.ndarray: The raw audio with the standardized length.
        """
        current_length = len(audio)
        if current_length < self.target_length:
            # Audio is too short so pad where necessary
            padding = np.zeros(self.target_length - current_length)
            return np.concatenate((audio, padding))
        elif current_length > self.target_length:
            # Audio is too long so cut the end off.
            return audio[:self.target_length]
        # The audio is exactly the right length
        return audio
