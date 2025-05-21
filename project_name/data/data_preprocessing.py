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
            n_augmentations: int = 1,
            use_spectrograms: bool = False,
            n_mels: int = 128,
            n_fft: int = 2048,
            hop_length: int = 512
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
            n_augmentations (int, optional): The number of augmentation files
                that should be made for each data file. Defaults to 1.
            use_spectrograms (bool, optional): Whether the preprocessor should
                use spectrograms or raw data. Defaults to False.
            n_mels (int, optional): The number of frequency bins. Defaults to
                128.
            n_fft (int, optional): The length of the fft window. Defaults to
                2048.
            hop_length (int, optional): By how much the fft window is moved
                each step. Defaults to 512.

        """
        self.sampling_rate = sampling_rate
        self.target_length = target_length
        self.data_augmenter = data_augmenter
        self.n_augmentations = n_augmentations
        self.use_spectrograms = use_spectrograms
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

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
            data_list = self._process_single_file(fp)
            if data_list is not None:  # Only append succesfully loaded audio
                for data in data_list:
                    processed_data.append(data)
                    # If more files were added, more labels should be added.
                    emotion_labels.append(emotion)
                    intensity_labels.append(intensity)

        return (np.array(processed_data),
                np.array(emotion_labels),
                np.array(intensity_labels))

    def _process_single_file(
            self, file_path: str
    ) -> list[np.ndarray] | None:
        """Load and preprocess the audio and extract the labels.

        If self.n_augmentations > 1, there will be more augmented versions for
            each file.
        Args:
            file_path (str): The path to where the audio file is stored.

        Returns:
            list[np.ndarray] | None: The preprocessed audio.
        """
        raw_audio = self._load_audio(file_path)
        if raw_audio is None:
            return None
        # Process the raw data
        augmented = self._augment_data(raw_audio)

        processed = []
        for augmented_audio in augmented:
            standardized_audio = self._standardize_raw_length(augmented_audio)

            if self.use_spectrograms:
                # Process spectrograms
                spec = self._create_mel_spectrogram(standardized_audio)
                processed.append(spec)
            else:
                # Or just keep the waveform
                processed.append(standardized_audio)

        return processed

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

    def _create_mel_spectrogram(self, raw_audio: np.ndarray) -> np.ndarray:
        mel_spec = librosa.feature.melspectrogram(
            y=raw_audio,
            sr=self.sampling_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        # Put the values into a more manageble scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        # Add an empty dimension at the start as it is just 2D
        log_mel_spec[np.newaxis, :, :]  # (1, n_mels, time_frames)
        return log_mel_spec

    def _augment_data(self, data: np.ndarray) -> list[np.ndarray]:
        """Augment either the raw audio or spectogram.

        It creates "self.n_augmentations" augmented versions of the data.
        Args:
            data (np.ndarray): Either the raw audio when a raw audio augmenter
                is given or a spectrogram when a spectrogram augmenter is
                given. The right augmenter should be provided.

        Returns:
            list[np.ndarray]: The augmented raw audio/ spectrogram. More
                augmented versions are placed together in a list.
        """
        if self.data_augmenter:
            augmented_versions = [self.data_augmenter.augment_raw_file(data)
                                  for _ in range(self.n_augmentations)]
            return augmented_versions
        # Return without augmenting
        return [data]

    def _standardize_raw_length(self, audio: np.ndarray) -> np.ndarray:
        """
        Pad or trim the raw audio data to make it the right length.

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

    def _standardize_spectrogram_size(self, spectrogram):
        pass
