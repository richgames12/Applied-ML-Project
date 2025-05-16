import numpy as np
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

    def augment_raw_file(self, audio_data: np.ndarray) -> np.ndarray:
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
