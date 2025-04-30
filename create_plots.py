import kagglehub
import matplotlib.pyplot as plt
import os
import numpy as np
import librosa.display

#plot the first audio file in the dataset
dataset_path = 'data/uwrfkaggler/ravdess-emotional-speech-audio/versions/1'
first_audio_file = os.path.join(dataset_path, 'Actor_01', '03-01-01-01-01-01-01.wav')
def plot_audio_file(file_path, sr=22050):
    # Load the audio file
    y, sr = librosa.load(file_path, sr=sr)

    # Create a time variable for plotting
    time = np.linspace(0, len(y) / sr, len(y))

    # Plot the waveform
    plt.figure(figsize=(12, 4))
    plt.plot(time, y)
    plt.title('Waveform of Audio File of Actor 01')
    plt.xlim(0, time[-1])
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    plot_audio_file(first_audio_file)