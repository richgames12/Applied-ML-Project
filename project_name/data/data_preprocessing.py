import librosa
import numpy as np
from project_name.data.data_augmentation import (
    RawAudioAugmenter,
    SpectrogramAugmenter
)
from tqdm import tqdm


class AudioPreprocessor:
    """A preprocessing class for raw audio and spectrograms."""
    def __init__(
            self,
            sampling_rate: int = 22050,
            target_length: int = 66150,
            data_augmenter: RawAudioAugmenter | None = None,
            spectrogram_augmenter: SpectrogramAugmenter | None = None,
            n_augmentations: int = 1,
            use_spectrograms: bool = False,
            n_mels: int = 128,
            n_fft: int = 2048,
            hop_length: int = 512
    ) -> None:
        """
        Initialize the audio preprocessor.

        Args:
            sampling_rate (int, optional): The sampling rate of the audio data.
                Defaults to 22050.
            target_length (int, optional): How long the audio data should be.
                Defaults to (3*22050 = 66150).
            data_augmenter (RawAudioAugmenter | None, optional): The augmenter
                that is to be used. If no augmenter is specified, there will be
                no augmentation. Defaults to None.
            spectrogram_augmenter (SpectrogramAugmenter | None, optional): The
                augmenter that is to be used for spectrograms. If no augmenter
                is specified, there will be no augmentation. Defaults to None.
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
        self.spectrogram_augmenter = spectrogram_augmenter
        self.n_augmentations = n_augmentations
        self.use_spectrograms = use_spectrograms
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

    def process_all(
        self,
        file_path_label_pairs: list[tuple[str, tuple[int, int]]] | list[str]
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """
        Load and preprocess audio data from (filepath, label) pairs.

        Args:
            file_path_label_pairs (list[Tuple[str, tuple[int, int]]] |
                list[str]): A list of tuples, where each tuple contains the
                file path to the audio data and its corresponding (emotion,
                intensity) label. If there are no labels, it should be just a
                list of file paths.

        Returns:
            Tuple[np.ndarray, np.ndarray | None, np.ndarray | None]: The
            processed data (raw or spectrogram), emotion labels, and intensity
            labels. If the input contains no emotion or intensity labels, they
            will be None
        """
        processed_data = []
        emotion_labels = []
        intensity_labels = []

        # Detect whether the input contains labels or not
        labels_present = (
            isinstance(file_path_label_pairs[0], tuple)
            and isinstance(file_path_label_pairs[0][1], tuple)
            and len(file_path_label_pairs[0][1]) == 2
        )

        for item in tqdm(
            file_path_label_pairs, desc="Preprocessing audio files"
        ):
            # Extract labels only when they are present
            if labels_present:
                file_path, (emotion, intensity) = item
            else:
                file_path = item

            data_list = self._process_single_file(file_path)
            if data_list is not None:  # Only append succesfully loaded audio
                for data in data_list:
                    processed_data.append(data)
                    # If more files were added, more labels should be added.
                    if labels_present:
                        # Only handle these if they are present
                        emotion_labels.append(emotion)
                        intensity_labels.append(intensity)

        data_array = np.array(processed_data)
        # Only return labels when they exist
        if labels_present:
            return (
                data_array,
                np.array(emotion_labels),
                np.array(intensity_labels)
            )
        else:
            return data_array, None, None

    def _process_single_file(
            self, file_path: str
    ) -> list[np.ndarray] | None:
        """
        Load and preprocess the audio and extract the labels.

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

        processed = []

        if self.use_spectrograms:
            # Standardize spectrogram length beforehand
            standardized_audio = self._standardize_raw_length(raw_audio)
            spectrogram = self._create_log_mel_spectrogram(standardized_audio)

            # Augment spectrogram instead of raw
            augmented_specs = self._augment_data(spectrogram)

            for spec in augmented_specs:
                normalized_spec = self._normalize_spectrogram(spec)
                processed.append(normalized_spec)

        else:
            # Augment raw waveform
            augmented_waveforms = self._augment_data(raw_audio)

            for waveform in augmented_waveforms:
                # Here the augmentation can change raw length
                standardized = self._standardize_raw_length(waveform)
                normalized = self._normalize_waveform(standardized)
                processed.append(normalized)

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

    def _create_log_mel_spectrogram(self, raw_audio: np.ndarray) -> np.ndarray:
        """
        Create a log mel spectrogram from raw audio.

        The loudness in each spectrogram is normalized by its own
            maximum power value.
        Args:
            raw_audio (np.ndarray): The audio for which the spectrogram needs
                to be made. Has shape: (len(audio)).

        Returns:
            np.ndarray: The resulting log mel spectrogram with an extra
                channel dimension of 1. Has shape: (1, n_mels, timeframes).
        """
        mel_spec = librosa.feature.melspectrogram(
            y=raw_audio,
            sr=self.sampling_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        # Put the values into a more manageble scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        # Add an empty dimension at the start as it is just 2D not RGB
        # It should have shape (1, n_mels, time_frames)
        log_mel_spec = log_mel_spec[np.newaxis, :, :]
        return log_mel_spec

    def _augment_data(self, data: np.ndarray) -> list[np.ndarray]:
        """
        Augment either the raw audio or spectogram.

        It creates "self.n_augmentations" augmented versions of the data.
        Args:
            data (np.ndarray): Either the raw audio when a raw audio augmenter
                is given or a spectrogram when a spectrogram augmenter is
                given. The right augmenter should be provided.

        Returns:
            list[np.ndarray]: The augmented raw audio/ spectrogram. More
                augmented versions are placed together in a list. The original
                version is always included before the augmented versions.
        """
        if self.use_spectrograms and self.spectrogram_augmenter:
            # Spectrograms are used and must be augmented
            augmented_versions = [self.spectrogram_augmenter(data)
                                  for _ in range(self.n_augmentations)]
            return [data] + augmented_versions  # Include original spectrogram

        elif not self.use_spectrograms and self.data_augmenter:
            # Raw audio is used and it must be augmented
            augmented_versions = [self.data_augmenter(data)
                                  for _ in range(self.n_augmentations)]
            return [data] + augmented_versions  # Include original raw waveform

        # No augmentations are applied
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

    def _normalize_waveform(self, audio: np.ndarray) -> np.ndarray:
        """
        Devide the entire waveform by the loudest sound.

        Args:
            audio (np.ndarray): The raw input waveform.

        Returns:
            np.ndarray: The normalized waveform.
        """
        max_val = np.max(np.abs(audio))
        return audio / max_val if max_val > 0 else audio

    def _normalize_spectrogram(self, spec: np.ndarray) -> np.ndarray:
        """
        Scale the spectrogram values to a range of (0-1).

        Args:
            spec (np.ndarray): The spectrogram that needs normalizing.

        Returns:
            np.ndarray: The normalized spectrogram.
        """
        min_val = np.min(spec)
        max_val = np.max(spec)
        if max_val - min_val == 0:
            return np.zeros_like(spec)  # Avoid division by zero
        return (spec - min_val) / (max_val - min_val)
