import urllib.request as ur
import zipfile
import shutil


from utils.models.execution import neural_network, random_forest
from utils.models.definition import define_models
from utils.preprocessing import *
import datetime


if __name__ == '__main__':
    print('Downloading dataset')
    url = 'https://data.vision.ee.ethz.ch/cvl/ae_dataset/ae_dataset.zip'

    path_to_zip = 'datasets/raw/ae_dataset.zip'
    ur.urlretrieve(url, path_to_zip)

    with zipfile.ZipFile(path_to_zip, 'r') as zip_file:
        zip_file.extractall('datasets/raw/tmp')

    shutil.move('datasets/raw/tmp/AudioEventDataset/train', 'datasets/raw')
    shutil.move('datasets/raw/tmp/AudioEventDataset/test', 'datasets/raw')

    shutil.rmtree('datasets/raw/tmp')
    os.remove('datasets/raw/ae_dataset.zip')

    print('Extracting training set features')
    training_path = 'datasets/raw/train/'
    genre_paths = [x for x in os.listdir('datasets/raw/train_mini')]

    training_set_features(training_path, genre_paths, to_csv=True)

    print('Extracting test set features')
    test_path = 'datasets/raw/test/'
    test_set_features(test_path, to_csv=True)

    timestamp = datetime.datetime.now().strftime('%m-%d-%Y_%H:%M:%S')

    training_set = read_dataset('src/features/training_set.csv', drop=True, process=True)
    test_set = read_dataset('src/features/test_set.csv', drop=True, process=True)

    print('t-SNE execution')
    t_SNE(training_set, test_set)
    print('k-means execution')
    k_means(training_set, test_set)

    shape = len(training_set.columns) - 1

    print('Training random forests')
    random_forest(training_set, test_set, 5, timestamp, whole=True)

    print('Training neural networks')
    neural_network(training_set, test_set, 5, define_models(shape), timestamp, whole=True)

