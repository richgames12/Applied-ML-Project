from librosa.feature import mfcc, delta
import numpy as np


class AudioFeatureExtractor:
    """
    Extract audio features from raw waveform data.
    Currently only works with mfccs but that can be extended.
    """
    def __init__(
        self,
        sampling_rate: int = 22050,
        n_mfcc: int = 13,
        n_fft: int = 1024,
        hop_length: int = 512,
        use_deltas: bool = True
    ):
        """
        Initialize the audio feature extractor.

        Args:
            sampling_rate (int, optional): The sampling rate of the audio.
                Defaults to 22050.
            n_mfcc (int, optional): The number of MFCCs to extract.
                Defaults to 13.
            n_fft (int, optional): The size of each fft bin. Defaults to 1024.
            hop_length (int, optional): The amount by which the fft window is
                shifted. Defaults to 512.
            use_deltas (bool, optional): Also use the derivatives of the MFCCs.
                Defaults to True.
        """
        self.sampling_rate = sampling_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.use_deltas = use_deltas

    def extract_features(self, audio_data: np.ndarray) -> np.ndarray:
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
                n_mfcc=self.n_mfcc,
                # Have the windows overlap to get smoother results
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )

            if self.use_deltas:
                # The derivatives of the mfccs
                delta_mfccs = delta(mfccs)
                combined = np.vstack([mfccs, delta_mfccs])
            else:
                combined = mfcc

            return combined.mean(axis=1)

        except Exception as e:
            print(f"Failed to extract features from the audio: {e}")
            return None
