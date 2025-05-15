import kagglehub
import os
import shutil

path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")

# Move the dataset to a specific directory
destination_dir = os.path.join(os.getcwd(), "project_name", "data", "uwrfkaggler", "ravdess-emotional-speech-audio", "versions")
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)
source_dir = path

shutil.move(source_dir, destination_dir)
print("Path to dataset files:", path)