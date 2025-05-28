import kagglehub
import os
import shutil


destination_dir = os.path.join(
    os.getcwd(),
    "project_name",
    "data",
    "ravdess-audio",
)
DATASET_DIR = os.path.join(destination_dir, '1', 'audio_speech_actors_01-24')

if __name__ == "__main__":
    if not os.path.exists(DATASET_DIR):
        # If the dataset has not been downloaded.
        path = kagglehub.dataset_download(
            "uwrfkaggler/ravdess-emotional-speech-audio"
        )

        # Move the dataset to a specific directory
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
        source_dir = path

        shutil.move(source_dir, destination_dir)
    else:
        print("Dataset has already been downloaded.")
    print("Path to dataset files:", DATASET_DIR)
