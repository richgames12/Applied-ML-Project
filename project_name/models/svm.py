import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline

# === Constants ===
DATASET_PATH = r"C:\Users\josse\.cache\kagglehub\datasets\uwrfkaggler\ravdess-emotional-speech-audio\versions\1"
N_MFCC = 13
EMOTION_LABELS = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised"
}

# === Helper Functions ===
def extract_label(filename):
    # Filename format: '03-01-01-01-01-01-01.wav'
    return int(filename.split("-")[2])

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    return np.mean(mfcc.T, axis=0)  # average over time axis

# === Load and preprocess dataset ===
X, y = [], []

for root, _, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            try:
                features = extract_features(file_path)
                X.append(features)
                y.append(extract_label(file))
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

X = np.array(X)
y = np.array(y)

# === Binarize labels for ROC (One-vs-Rest) ===
classes = sorted(EMOTION_LABELS.keys())
y_bin = label_binarize(y, classes=classes)
n_classes = y_bin.shape[1]

# === Train/Test Split ===
X_train, X_test, y_train_bin, y_test_bin = train_test_split(
    X, y_bin, test_size=0.2, random_state=42, stratify=y
)

# === SVM pipeline with probability estimates ===
clf = OneVsRestClassifier(make_pipeline(StandardScaler(), SVC(kernel='rbf', C=10, gamma='scale', probability=True)))
clf.fit(X_train, y_train_bin)

# === Predict and Evaluate ===
y_score = clf.predict_proba(X_test)
y_pred = clf.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test_bin, axis=1)

print("Classification Report:\n")
print(classification_report(y_test_labels, y_pred_labels, target_names=[EMOTION_LABELS[i] for i in sorted(EMOTION_LABELS)]))

# === Confusion Matrix ===
cm = confusion_matrix(y_test_labels, y_pred_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[EMOTION_LABELS[i] for i in sorted(EMOTION_LABELS)],
            yticklabels=[EMOTION_LABELS[i] for i in sorted(EMOTION_LABELS)])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - SVM on RAVDESS")
plt.tight_layout()
plt.show()

# === ROC Curves ===
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i],
             label=f"{EMOTION_LABELS[classes[i]]} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - One-vs-Rest SVM on RAVDESS')
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
