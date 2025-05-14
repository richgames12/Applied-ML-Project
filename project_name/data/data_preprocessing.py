import os
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
        emotion, intensity = self._extract_label_intensity(file_path)

        if self.use_spectrograms:
            # Process the spectograms
            spectrogram = self._get_spectogram(raw_audio)
            if spectrogram is not None:
                augmented_spectogram = self._augment_data(spectrogram)
                return self._standardize_spectrogram_length(
                    augmented_spectogram, emotion, intensity
                )
            return None, None, None

        # Process the raw data
        augmented_raw_audio = self._augment_data(raw_audio)
        length_standardized_audio = self._standardize_raw_length(
            augmented_raw_audio
        )
        return length_standardized_audio, emotion, intensity

    def process_all(self, file_paths: str) -> np.ndarray:
        processed_data = [self.process_single(fp) for fp in file_paths]
        # Filter out any None results if spectrogram processing failed
        return np.array([item for item in processed_data if item is not None])

    def _load_audio(self, file_path: str) -> np.ndarray:
        """
        Load the data of a single audio file.

        Args:
            file_path (str): The file path to the audio file.

        Returns:
            np.ndarray: The retrieved raw audio data.
        """
        return librosa.load(file_path, sr=self.sampling_rate)[0]

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
            return self.data_augmenter(data)
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

    def _extract_label_intensity(self, file_path: str) -> tuple[str, str]:
        """Extract the emotion label and the intensity from the file path.

        Expects the files to follow the following naming convention where,
        for example, in 03-01-01-01-01-01-01.wav the third number represents
        the label and the fourth one the intensity.
        (Expecting 8 different emotions and 2 different intensities)

        Args:
            file_path (str): The file path towards the audio file.
            should end with the following format: 03-01-01-01-01-01-01.wav.

        Returns:
            tuple[str, str]: The emotion label and the intensity.
        """
        filename = os.path.basename(file_path)
        parts = filename.split("-")[2]
        if len(parts) > 3:
            return parts[2], parts[3]
        raise ValueError(
            "Cannot extract emotion and intensity from the following ",
            f"filename: {filename}."
        )

    def _get_spectogram(self, audio: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Get spectrograms is still in progress.")

    def _augment_spectogram(self, spectrogram: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Augment spectrogram is still in progress.")

    def _standardize_spectrogram_length(
            self, spectrogram: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError("Standardize is still in progress.")
