import os
from sklearn.model_selection import train_test_split


class trainTestSplit:

    def __init__(self, dataset_path, test_size=0.15, eval_size=0.15, seed=1) -> None:  # using seed ensures the same split for reproducibility
        self.dataset_path = dataset_path
        self.test_size = test_size
        self.eval_size = eval_size
        self.seed = seed
        self.audio_paths = []
        self.labels = []
        self.val_set = []
        self.test_set = []

    def get_splits(self):
        if not self.train_set:
            self._split()
        return self.train_set, self.val_set, self.test_set

    def _extract_labels(self, filename):     # example filename: 01-01-01-01-01-01.wav
        return int(filename._split("-")[2])  # boilerplate code, should be changed later

    def _collect_files(self):        # copied from svm in jesses branch
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith(".wav"):
                    full_path = os.path.join(root, file)
                    self.audio_paths.append(full_path)
                    self.labels.append(self._extract_labels)    

    def _split(self):
        self._collect_files()

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
