from librosa.feature import mfcc
import numpy as np


class AudioFeatureExtractor:
    """
    Extract audio features from raw waveform data.
    Currently only works with mfccs but that can be extended.
    """
    def __init__(self, sampling_rate: int = 22050):
        """
        Initialize the audio feature extractor.

        Args:
            sampling_rate (int, optional): The sampling rate of the audio.
                Defaults to 22050.
        """
        self.sampling_rate = sampling_rate

    def _extract_features(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Extract the audio features from a single audio file.

        Args:
            audio_data (np.ndarray): The raw audio waveform.

        Returns:
            np.ndarray: The audio features of the audio file.
        """
        try:
            # Extract mel-frequency cepstral coefficients
            mfccs = mfcc(
                y=audio_data,
                sr=self.sampling_rate,
                n_mfcc=13,
                # Have the windows overlap to get smoother results
                n_fft=1024,
                hop_length=512
            )

            # More features can be added
            features = np.hstack([mfccs.mean(axis=1)])
            return features

        except Exception as e:
            print(f"Failed to extract features from the audio: {e}")
            return None
