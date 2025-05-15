import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class DataLoaderSplitter:
    """
    Extract .wav files from dataset_path together with their label and split
    them into train/validation/test groups.
    """
    def __init__(
            self,
            dataset_path: str,
            test_size: float = 0.10,
            eval_size: float = 0.10,
            seed: int = 1
    ) -> None:
        """
        Initialize the dataloadersplitter class.

        Args:
            dataset_path (str): Root path to dataset with .wav files.
            test_size (float, optional): Proportion of data to reserve for
                test set. Defaults to 0.10.
            eval_size (float, optional): Proportion of data to reserve for
                validation set. Defaults to 0.10.
            seed (int, optional): Random seed for reproducibility. Defaults
                to 1.
        """
        self.dataset_path = dataset_path
        self.test_size = test_size
        self.eval_size = eval_size
        self.seed = seed

        self.audio_paths = []  # list[str]
        self.labels = []  # list[tuple[int, int]]

        self.train_set = None  # list[tuple[str, tuple[int, int]]] | None
        self.val_set = None  # list[tuple[str, tuple[int, int]]] | None
        self.test_set = None  # list[tuple[str, tuple[int, int]]] | None

    def get_splits(self) -> tuple[
        list[tuple[str, tuple[int, int]]],
        list[tuple[str, tuple[int, int]]],
        list[tuple[str, tuple[int, int]]]
    ]:
        """Split the data into training/validation/testing sets.

        Returns:
            tuple[list[tuple[str, tuple[int, int]]],
                  list[tuple[str, tuple[int, int]]],
                  list[tuple[str, tuple[int, int]]]]:
                A tuple containing the training, validation, and test sets.

                Each set is a list of samples, where each sample is a tuple:
                (filepath, (emotion_label, intensity_label)).

                - filepath (str): Path to the .wav audio file.
                - emotion_label (int): Encoded emotion label.
                - intensity_label (int): Encoded intensity label.
        """
        if not self.train_set:
            self._split()
        return self.train_set, self.val_set, self.test_set

    def collect_files_labels(self) -> None:
        """Collect all files and corresponding labels in self.dataset_path."""
        for root, _, files in os.walk(self.dataset_path):
            for file in tqdm(files, desc="Collecting .wav files"):
                if file.endswith(".wav"):
                    full_path = os.path.join(root, file)
                    self.audio_paths.append(full_path)
                    self._extract_label_intensity(full_path)

    def _extract_label_intensity(self, file_path: str) -> None:
        """Extract the emotion label and the intensity from the file path.

        Expects the files to follow the following naming convention where,
        for example, in 03-01-01-01-01-01-01.wav the third number represents
        the label and the fourth one the intensity.
        (Expecting 8 different emotions and 2 different intensities).
        Updates self.labels directly.

        Args:
            file_path (str): The file path towards the audio file.
                Should end with the following format: 01-01-01-01-01-01-01.wav.
        """
        filename = os.path.basename(file_path)
        parts = filename.split("-")
        if len(parts) <= 3:
            raise ValueError(
                "Cannot extract emotion and intensity from the following ",
                f"filename: {filename}."
            )
        self.labels.append(parts[2], parts[3])

    def _split(self) -> None:
        self.collect_files()

        # First split into train and temp (val + test)
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            self.audio_paths, self.labels,
            test_size=self.test_size + self.eval_size,
            stratify=self.labels,
            random_state=self.seed
        )

        # Now split temp into val and test
        val_ratio = self.eval_size / (self.test_size + self.eval_size)
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels,
            test_size=1 - val_ratio,
            stratify=temp_labels,
            random_state=self.seed
        )

        self.train_set = list(zip(train_paths, train_labels))
        self.val_set = list(zip(val_paths, val_labels))
        self.test_set = list(zip(test_paths, test_labels))
