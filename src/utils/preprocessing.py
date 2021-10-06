import numpy as np

import pandas as pd

import librosa
import librosa.display
import librosa.feature as fe

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

import os

labels = ['acoustic_guitar', 'airplane', 'applause', 'bird', 'car', 'cat', 'child', 'church_bell', 'crowd',
          'dog_barking', 'engine', 'fireworks', 'footstep', 'glass_breaking', 'hammer', 'helicopter', 'knock',
          'laughter', 'mouse_click', 'ocean_surf', 'rustle', 'scream', 'speech', 'squeak', 'tone', 'violin',
          'water_tap', 'whistle']


def mfcc(x):
    return fe.mfcc(x, n_mfcc=20)


feature_extraction_functions = [
    mfcc,
    fe.rms,
    fe.chroma_stft,
    fe.spectral_centroid,
    fe.zero_crossing_rate,
    fe.spectral_contrast,
    librosa.onset.onset_strength,
    librosa.feature.spectral_rolloff
]


def training_set_features(path, genre_paths, to_csv=True):
    fs = 22050
    data, file_names, targets = list(), list(), list()

    with open('src/utils/files/utils/dataset_header.csv', 'r') as header_file:
        header = header_file.read().split(',')

    for genre in genre_paths:
        genre_path = path + genre
        for audio_file in os.listdir(genre_path):
            file_names.append(audio_file)
            targets.append(''.join([i for i in audio_file[:-4] if not i.isdigit()]))
            audio, sr = librosa.load(genre_path + '/' + audio_file, sr=fs)

            features = list()
            for function in feature_extraction_functions:
                feature = function(audio)
                if len(feature.shape) > 1 and feature.shape[0] > 0:
                    for values in feature:
                        features.append(np.mean(values))
                        features.append(np.var(values))
                        features.append(np.amin(values))
                        features.append(np.amax(values))
                else:
                    features.append(np.mean(feature))
            data.append(features)

    dataset = pd.DataFrame(data=np.array(data), columns=header)
    dataset.insert(loc=0, column='file_name', value=file_names)
    dataset.insert(loc=len(dataset.columns), column='label', value=targets)
    if to_csv:
        dataset.to_csv('src/features/training_set.csv')


def test_set_features(path, to_csv=True):
    fs = 22050
    data, file_names, targets = list(), list(), list()

    with open('src/utils/files/utils/dataset_header.csv', 'r') as header_file:
        header = header_file.read().split(',')

    for audio_file in os.listdir(path):
        file_names.append(audio_file)
        targets.append(''.join([i for i in audio_file[:-4] if not i.isdigit()]))
        audio, sr = librosa.load(path + '/' + audio_file, sr=fs)

        features = list()
        for function in feature_extraction_functions:
            feature = function(audio)
            if len(feature.shape) > 1 and feature.shape[0] > 0:
                for values in feature:
                    features.append(np.mean(values))
                    features.append(np.var(values))
                    features.append(np.amin(values))
                    features.append(np.amax(values))
            else:
                features.append(np.mean(feature))
        data.append(features)

    print(len(features))
    dataset = pd.DataFrame(data=np.array(data), columns=header)
    dataset.insert(loc=0, column='file_name', value=file_names)
    dataset.insert(loc=len(dataset.columns), column='label', value=targets)
    if to_csv:
        dataset.to_csv('src/features/test_set.csv')


def read_dataset(path: str, drop=True, process=False) -> pd.DataFrame:
    dataset = pd.read_csv(path)
    if drop:
        dataset.drop(dataset.columns[0], inplace=True, axis=1)

    X, y = dataset[dataset.columns[1:-1]].values, dataset['label'].values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    y = y.reshape(y.shape[0], 1)
    values = np.hstack((X, y))
    dataset = pd.DataFrame(data=values, columns=dataset.columns[1:])
    dataset = dataset.sample(frac=1).reset_index(drop=True)

    if process:
        scaler = StandardScaler()
        X, y = scaler.fit_transform(dataset[dataset.columns[1:-1]].values), \
               dataset['label'].values
        y = y.reshape(y.shape[0], 1)
        values = np.hstack((X, y))
        dataset = pd.DataFrame(data=values, columns=dataset.columns[1:])
        dataset = dataset.sample(frac=1).reset_index(drop=True)

    return dataset


def t_SNE(training_set, test_set):
    dataset = pd.concat([training_set, test_set], ignore_index=True)
    X, y = dataset[dataset.columns[:-1]].values, dataset['label'].values

    pca = PCA(n_components=30)
    X = pca.fit_transform(X)

    t_sne = TSNE(n_components=2)
    X = t_sne.fit_transform(X)

    y = y.reshape(y.shape[0], 1)
    values = np.hstack((X, y))

    clustering = pd.DataFrame(data=values, columns=['x', 'y', 'label'])
    clustering.to_csv('src/utils/files/clustering/t-SNE.csv')


def k_means(training_set, test_set):
    dataset = pd.concat([training_set, test_set], ignore_index=True)
    X, y = dataset[dataset.columns[:-1]].values, dataset['label'].values

    pca = PCA(n_components=2)
    X = pca.fit_transform(X)

    k_m = KMeans(n_clusters=28)
    k_m.fit(X)
    labels = k_m.labels_

    y = labels.reshape(labels.shape[0], 1)
    clustering = pd.DataFrame(data=np.hstack((X, y)), columns=['x', 'y', 'label'])
    clustering.to_csv('src/utils/files/clustering/k-means.csv')