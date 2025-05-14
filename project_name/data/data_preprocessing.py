import librosa
import numpy as np
from data.data_augmentation import RawAudioAugmenter


class AudioPreprocessor:
    def __init__(
            self,
            sampling_rate: int = 22050,
            target_length: int = 66150,
            # TODO: When using a CNN we should also include the option to
            # select a different data augmenter for spectrograms
            data_augmenter: RawAudioAugmenter | None = None,
            use_spectrograms: bool = False
    ) -> None:
        self.sampling_rate = sampling_rate
        self.target_length = target_length
        self.data_augmenter = data_augmenter
        self.use_spectrograms = use_spectrograms

    def process_single_file(self, file_path):
        raw_audio = self._load_audio(file_path)

        if self.use_spectrograms:
            # Process the spectograms
            spectrogram = self._get_spectogram(raw_audio)
            if spectrogram is not None:
                augmented_spectogram = self._augment_spectogram(spectrogram)
                # TODO: Maybe standardize the length of the spectrogram
                return augmented_spectogram
            return None

        # Process the raw data
        augmented_raw_audio = self._augment_raw_audio(raw_audio)
        return self._standardize_length(augmented_raw_audio)

    def process_all_files(self, file_paths):
        pass

    def _load_audio(self, file_path: str) -> np.ndarray:
        """
        Load the data of a single audio file.

        Args:
            file_path (str): The file path to the audio file.

        Returns:
            np.ndarray: The retrieved raw audio data.
        """
        return librosa.load(file_path, sr=self.sampling_rate)[0]

    def _augment_raw_audio(self, audio):
        pass

    def _get_spectogram(self, audio: np.ndarray) -> np.ndarray:
        # TODO: Will have to be implemented later for CNN
        raise NotImplementedError("Get spectrograms is still in progress.")

    def _augment_spectogram(self, spectrogram: np.ndarray) -> np.ndarray:
        # TODO: Will have to be implemented later for CNN
        raise NotImplementedError("Augment spectrogram is still in progress.")

    def _standardize_length(self, audio: np.ndarray) -> np.ndarray:
        """Pad or trim the raw audio data to make it the right lenght.

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
