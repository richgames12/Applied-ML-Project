import os
from sklearn.model_selection import train_test_split
from copy import deepcopy


class DataFileSplitter:
    """
    Extract .wav files from dataset_path together with their label and split
    them into train and test groups.
    """
    def __init__(
            self,
            dataset_path: str,
            test_size: float = 0.10,
            seed: int | None = None
    ) -> None:
        """
        Initialize the dataloadersplitter class.

        Args:
            dataset_path (str): Root path to dataset with .wav files.
            test_size (float, optional): Proportion of data to reserve for
                test set. Defaults to 0.10.
            seed (int | None, optional): Random seed for reproducibility.
                Defaults to None.
        """
        self._dataset_path = dataset_path
        self.test_size = test_size
        self.seed = seed

        self._audio_paths = []  # list[str]
        self._labels = []  # list[tuple[int, int]]

        self._train_set = None  # list[tuple[str, tuple[int, int]]] | None
        self._test_set = None  # list[tuple[str, tuple[int, int]]] | None

        self._collect_files_labels()  # Collect files and labels at the start

    def get_data_all_copy(self) -> list[tuple[str, tuple[int, int]]]:
        """
        Return a deepcopy of all data without splitting.

        Returns:
            list[ tuple[str, tuple[int, int]] ]: The paths to the audio files
                and their emotion and intensity labels.
        """
        return deepcopy(list(zip(self._audio_paths, self._labels)))

    def get_data_splits_copy(self) -> tuple[
        list[tuple[str, tuple[int, int]]],
        list[tuple[str, tuple[int, int]]],
    ]:
        """
        Return a deepcopy of the data in training and testing sets.

        Returns:
            tuple[ list[tuple[str, tuple[int, int]]],
                  list[tuple[str, tuple[int, int]]] ]:
                A tuple containing the training and test sets.

                Each set is a list of samples, where each sample is a tuple:
                (filepath, (emotion_label, intensity_label)).

                - filepath (str): Path to the .wav audio file.
                - emotion_label (int): Encoded emotion label.
                - intensity_label (int): Encoded intensity label.
        """
        if not self._train_set:
            self._split()
        return (
            deepcopy(self._train_set),
            deepcopy(self._test_set)
        )

    def _collect_files_labels(self) -> None:
        """Collect all files and corresponding labels in self.dataset_path."""
        if self._audio_paths:  # Already collected
            return

        # The dataset should exist, otherwise no filepaths will be collected
        if not os.path.isdir(self._dataset_path):
            raise FileNotFoundError(
                f"Dataset path '{self._dataset_path}' does not exist or is"
                " not a directory.\nPlease make sure the dataset is downloaded"
                " and placed at this location."
            )

        for root, _, files in os.walk(self._dataset_path):
            for file in files:
                if file.endswith(".wav"):
                    try:
                        full_path = os.path.join(root, file)
                        self._audio_paths.append(full_path)
                        self._extract_label_intensity(full_path)
                    except Exception as e:
                        print(
                            f"Warning: Skipping file '{file}' due to error: ",
                            f"{e}."
                        )
        print(f"Collected {len(self._audio_paths)} unique files.")

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
        self._labels.append((int(parts[2]), int(parts[3])))

    def _split(self) -> None:
        """Split the data into training and testing sets."""
        # Split into train and test
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            self._audio_paths, self._labels,
            test_size=self.test_size,
            stratify=self._labels,
            random_state=self.seed
        )

        self._train_set = list(zip(train_paths, train_labels))
        self._test_set = list(zip(test_paths, test_labels))
