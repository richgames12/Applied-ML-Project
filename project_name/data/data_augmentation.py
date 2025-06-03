import numpy as np
import random
from audiomentations import (
    Compose,
    AddGaussianNoise,
    TimeStretch,
    PitchShift,
    Shift,
)


class RawAudioAugmenter:
    """
    Starts the augmenter with dataset and outputh path.
    Uses audiomentations to write the augmentation pipeline.
    """
    def __init__(
        self,
        sampling_rate: int = 22050,
        min_noise_amplitude: float = 0.001,
        max_noise_amplitude: float = 0.010,
        noise_probability: float = 0.3,
        min_speed: float = 0.9,
        max_speed: float = 1.1,
        speed_probability: float = 0.3,
        lower_pitch: float = -2,
        upper_pitch: float = 2,
        pitch_probability: float = 0.3,
        backward_shift: float = -0.1,
        forward_shift: float = 0.1,
        shift_probability: float = 0.3
    ) -> None:
        """Initializes the RawAudioAugmenter class with specific values.

        Args:
            sampling_rate (int, optional): The rate at which the data was
                sampled. Defaults to 22050.
            min_noise_amplitude (float, optional): The lower bound of the
                added noise amplitude. Defaults to 0.001.
            max_noise_amplitude (float, optional): The upper bound of the
                added noise amplitude. Defaults to 0.015.
            noise_probability (float, optional): The probability of applying
                the Gaussian noise. Defaults to 0.5.
            min_speed (float, optional): The lower bound of the tempo
                variations. Defaults to 0.8.
            max_speed (float, optional): The upper bound of the tempo
                variations. Defaults to 1.25.
            speed_probability (float, optional): The probability of applying
                the tempo variation. Defaults to 0.5.
            lower_pitch (float, optional): The lower bound of the pitch shift
                (in semitones). Defaults to -4.
            upper_pitch (float, optional): The upper bound of the pitch shift
                (in semitones). Defaults to 4.
            pitch_probability (float, optional): The probability of applying
                the pitch shift. Defaults to 0.5.
            backward_shift (float, optional): The furthest the audio can be
                shifted backwards (e.g. 0.5 = 50%). Defaults to -0.5.
            forward_shift (float, optional): The furthest the audio can be
                shifted forwards (e.g. 0.5 = 50%). Defaults to 0.5.
            shift_probability (float, optional): The probability of applying
                the shift. Defaults to 0.5.
        """
        self.sampling_rate = sampling_rate

        self.min_noise_amplitude = min_noise_amplitude
        self.max_noise_amplitude = max_noise_amplitude
        self.noise_probability = noise_probability

        self.min_speed = min_speed
        self.max_speed = max_speed
        self.speed_probability = speed_probability

        self.lower_pitch = lower_pitch
        self.upper_pitch = upper_pitch
        self.pitch_probability = pitch_probability

        self.backward_shift = backward_shift
        self.forward_shift = forward_shift
        self.shift_probability = shift_probability

    def __call__(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Augment the raw audio.

        AddGaussianNoise: Adds random noise with an amplitude between min and
            max noise amplitude.
        TimeStretch: Speeds up or slows down the audio between min and max
            speed.
        PitchShift: Alters pitch between lower and upper pitch.
        Shift: Moves audio forward/backward between min and max shift.

        Augmentation techniques can be disabled by setting the probability of
            that technique being applied to 0.

        Args:
            audio_data (np.ndarray): The raw audio data that is to be
                augmented.

        Returns:
            np.ndarray: The augmented audio data.
        """
        augment = Compose([
            AddGaussianNoise(self.min_noise_amplitude,
                             self.max_noise_amplitude,
                             p=self.noise_probability),
            TimeStretch(self.min_speed,
                        self.max_speed,
                        p=self.speed_probability),
            PitchShift(self.lower_pitch,
                       self.upper_pitch,
                       p=self.pitch_probability),
            Shift(self.backward_shift,
                  self.forward_shift,
                  p=self.shift_probability)
        ])
        return augment(samples=audio_data, sample_rate=self.sampling_rate)


class SpectrogramAugmenter:
    def __init__(
        self,
        freq_mask_percentage: float = 0.15,
        time_mask_percentage: float = 0.2,
        freq_mask_prob: float = 0.5,
        time_mask_prob: float = 0.5,
        noise_std: float = 0.01,
        noise_prob: float = 0.3,
        brightness_min: float = 0.9,
        brightness_max: float = 1.1,
        brightness_prob: float = 0.3
    ):
        """
        Augment spectrograms with a variety of effects.
        All percentages are fractions (e.g., 0.2 = 20%)

        Args:
            freq_mask_percentage: Max % of mel bins to mask
            time_mask_percentage: Max % of time steps to mask
            freq_mask_prob: Probability to apply frequency masking
            time_mask_prob: Probability to apply time masking
            noise_std: Standard deviation of Gaussian noise
            noise_prob: Probability to apply Gaussian noise
            brightness_min: Minimum brightness scaling factor
            brightness_max: Maximum brightness scaling factor
            brightness_prob: Probability to scale brightness
        """
        self.freq_mask_percentage = freq_mask_percentage
        self.time_mask_percentage = time_mask_percentage
        self.freq_mask_prob = freq_mask_prob
        self.time_mask_prob = time_mask_prob
        self.noise_std = noise_std
        self.noise_prob = noise_prob
        self.brightness_min = brightness_min
        self.brightness_max = brightness_max
        self.brightness_prob = brightness_prob

    def __call__(self, spectrogram: np.ndarray) -> np.ndarray:
        """
        Apply all augmentations to a single spectrogram.

        Args:
            spectrogram: Spectrogram of shape (1, n_mels, time_frames)

        Returns:
            np.ndarray: Augmented spectrogram
        """
        augmented = np.copy(spectrogram)
        _, n_mels, n_frames = augmented.shape

        # Frequency Masking
        if random.random() < self.freq_mask_prob:
            # Percentage based so find actual n_mels
            max_width = int(self.freq_mask_percentage * n_mels)
            width = random.randint(0, max_width)
            start = random.randint(0, max(0, n_mels - width))
            augmented[0, start:start + width, :] = 0

        # Time Masking
        if random.random() < self.time_mask_prob:
            # Percentage based so find actual n_frames
            max_width = int(self.time_mask_percentage * n_frames)
            width = random.randint(0, max_width)
            start = random.randint(0, max(0, n_frames - width))
            augmented[0, :, start:start + width] = 0

        # Gaussian Noise
        if random.random() < self.noise_prob:
            noise = np.random.normal(0, self.noise_std, augmented.shape)
            augmented += noise

        # Brightness Scaling
        if random.random() < self.brightness_prob:
            factor = random.uniform(self.brightness_min, self.brightness_max)
            augmented *= factor

        return augmented
