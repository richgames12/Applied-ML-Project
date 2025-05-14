import os
import librosa
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift


class AudioAugmenter:
    """
    Starts the augmenter with dataset and outputh path
    Uses audiomentations to write the augmentation pipeline
    """
    def __init__(self, dataset_path, augmented_path="augmented_data"):
        self.dataset_path = dataset_path
        self.augmented_path = augmented_path

# Augmentation Descriptions:
# - AddGaussianNoise: Adds random noise between 0.001–0.015 amplitude to simulate background interference.
# - TimeStretch: Speeds up or slows down the audio by 0.8x to 1.25x, simulating tempo variations.
# - PitchShift: Alters pitch between -4 and +4 semitones to simulate vocal variation or emotional tone.
# - Shift: Moves audio forward/backward up to 50% of duration, simulating sync or alignment variation.


        self.augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
            Shift(min_shift=-0.5, max_shift=0.5, p=0.5)
        ])

    def augment_dataset(self):
        os.makedirs(self.augmented_path, exist_ok=True)

        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith('.wav'):
                    self._process_file(root, file)

        print(f"Augmented data saved in: {self.augmented_path}")

    def _process_file(self, root, file):
        file_path = os.path.join(root, file)
        audio, sr = librosa.load(file_path, sr=None)
        augmented_audio = self.augment(samples=audio, sample_rate=sr)

        relative_path = os.path.relpath(root, self.dataset_path)
        save_dir = os.path.join(self.augmented_path, relative_path)
        os.makedirs(save_dir, exist_ok=True)

        augmented_file_path = os.path.join(save_dir, f"aug_{file}")
        sf.write(augmented_file_path, augmented_audio, sr)


if __name__ == "__main__":
    dataset_path = r"C:\Users\josse\.cache\kagglehub\datasets\uwrfkaggler\ravdess-emotional-speech-audio\versions\1"
    augmenter = AudioAugmenter(dataset_path)
    augmenter.augment_dataset()
