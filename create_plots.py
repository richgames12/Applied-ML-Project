import kagglehub
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import numpy as np
import librosa.display

#plot the first audio file in the dataset
dataset_path = 'project_name/data/uwrfkaggler/ravdess-emotional-speech-audio/versions/1'
first_audio_file = os.path.join(dataset_path, 'Actor_01', '03-01-02-01-01-01-01.wav')

def plot_audio_file(file_path: str, sampling_rate: float=44100):
    # Load the audio file
    time_series, sampling_rate = librosa.load(file_path, sr=sampling_rate)

    # Create a time variable for plotting
    samples = np.linspace(0, len(time_series) / sampling_rate, len(time_series))
    spectrogram = librosa.feature.melspectrogram(y=time_series, sr=sampling_rate, n_mels=128, fmax=8000)

    # Reduce dimensionality with T-SNE
    tsne = TSNE(n_components=2, random_state=42)
    transformed_spectrogram = spectrogram.T
    reduced_spectrogram = tsne.fit_transform(transformed_spectrogram)

    # plot the waveform
    plt.figure(figsize=(10, 4))
    plt.plot(samples, time_series)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Waveform')
    plt.show()

    # Plot the spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        librosa.power_to_db(spectrogram, ref=np.max), y_axis='mel', x_axis='time', sr=sampling_rate, fmax=8000
    )
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Time')
    plt.ylabel('Mel frequency')
    plt.title('Spectrogram')
    plt.show()
    
    #plot the reduced spectrogram
    plt.figure(figsize=(10, 4))
    plt.scatter(reduced_spectrogram[:, 0], reduced_spectrogram[:, 1], c='blue', alpha=0.5)
    plt.title('T-SNE of Mel Spectrogram')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    plot_audio_file(first_audio_file)
