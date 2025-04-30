import kagglehub
import matplotlib.pyplot as plt
import os
import numpy as np
import librosa.display

#plot the first audio file in the dataset
dataset_path = 'data/uwrfkaggler/ravdess-emotional-speech-audio/versions/1'
first_audio_file = os.path.join(dataset_path, 'Actor_01', '03-01-02-02-01-01-01.wav')
def plot_audio_file(file_path, sr=44100):
    # Load the audio file
    y, sr = librosa.load(file_path, sr=sr)

    # Create a time variable for plotting
    time = np.linspace(0, len(y) / sr, len(y))
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

    # plot the waveform
    plt.figure(figsize=(10, 4))
    plt.plot(time, y)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Waveform')
    plt.show()

    # Plot the spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), y_axis='mel', x_axis='time', sr=sr, fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Time')
    plt.ylabel('Mel frequency')
    plt.title('Spectrogram')
    plt.show()

if __name__ == "__main__":
    plot_audio_file(first_audio_file)