import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


class ModelEvaluator:
    def __init__(self, class_labels: dict[int, str]) -> None:
        """
        Initializes the ModelEvaluator with class labels for reporting.

        Args:
            class_labels (Dict[int, str]): A dictionary mapping numerical
                labels to class names. Example: {0: 'classA', 1: 'classB'}
        """
        self.class_labels = class_labels
        # Ensure labels are sorted by their integer keys for consistency
        self.sorted_class_names = [
            self.class_labels[i] for i in sorted(self.class_labels.keys())
        ]
        self.n_classes = len(self.class_labels)

    def _print_classification_report(
        self,
        labels_true: np.ndarray,
        labels_pred: np.ndarray,
        title_suffix: str = ""
    ) -> None:
        """
        Print the classification report which includes muliple metrics.

        Args:
            labels_true (np.ndarray): The correct labels.
            labels_pred (np.ndarray): The corresponding predicted labels.
            title_suffix (str, optional): Extra information to add to the
                title. For example "Emotion Recognition SVM". Defaults to "".
        """
        print(f"\n--- Classification Report for {title_suffix} ---")
        print(classification_report(
            labels_true, labels_pred, target_names=self.sorted_class_names
        ))

    def _plot_confusion_matrix(
        self,
        labels_true: np.ndarray,
        labels_pred: np.ndarray,
        title_suffix: str = ""
    ) -> None:
        """
        Plot the confusion matrix of the predicted labels.

        Args:
            labels_true (np.ndarray): The correct labels.
            labels_pred (np.ndarray): The corresponding predicted labels.
            title_suffix (str, optional): Extra information to add to the
                title. For example "Emotion Recognition SVM". Defaults to "".
        """
        cm = confusion_matrix(labels_true, labels_pred)
        plt.figure(figsize=(self.n_classes * 2, self.n_classes * 2))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=self.sorted_class_names,
                    yticklabels=self.sorted_class_names)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix {title_suffix}")
        plt.tight_layout()
        plt.show()

    def evaluate_from_predictions(
        self,
        labels_true: np.ndarray,
        labels_pred: np.ndarray,
        title_suffix: str = ""
    ) -> None:
        """
        Checks whether the supplied labels are correct and calls the print
            classification report and plot confusion matrix methods.

        Args:
            labels_true (np.ndarray): True class labels (1D array of integers).
                Example: np.array([0, 1, 0, 2])
            labels_pred (np.ndarray): Predicted class labels (1D array of
                integers). Example: np.array([0, 1, 2, 2])
            title_suffix (str): An optional suffix to add to plot titles.
        """
        # Ensure input labels are 1D integer arrays
        if labels_true.ndim != 1 or not np.issubdtype(
            labels_true.dtype, np.integer
        ):
            raise ValueError("labels_true must be a 1D array of integers.")
        if labels_pred.ndim != 1 or not np.issubdtype(
            labels_pred.dtype, np.integer
        ):
            raise ValueError("labels_pred must be a 1D array of integers.")

        # 1 Print Classification Report
        self._print_classification_report(
            labels_true, labels_pred, title_suffix
        )

        # 2 Plot Confusion Matrix
        self._plot_confusion_matrix(labels_true, labels_pred, title_suffix)
