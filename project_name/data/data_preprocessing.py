import librosa
import numpy as np
from data.data_augmentation import RawAudioAugmenter


class AudioPreprocessor:
    def __init__(
            self,
            sampling_rate: int = 22050,
            target_length: int = 66150,
            # When using a CNN we should also include the option to select a
            # different data augmenter for spectrograms
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
        return self.standardize_length(augmented_raw_audio)

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

    def _get_spectogram(self, audio):
        raise NotImplementedError("Get spectrograms is still in progress.")

    def _augment_spectogram(self, spectogram):
        # Will have to be implemented later
        raise NotImplementedError("Augment spectrogram is still in progress.")

    def _standardize_length(self, audio):
        pass
