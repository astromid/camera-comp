import numpy as np
import os
import inspect
import cv2
from glob import glob
from keras.utils import Sequence
from keras.callbacks import Callback
from tqdm import tqdm
from abc import abstractmethod
from sklearn.utils.class_weight import compute_sample_weight
from multiprocessing import Pool
from joblib import Parallel, delayed

LABELS = [
    'HTC-1-M7',
    'iPhone-4s',
    'iPhone-6',
    'LG-Nexus-5x',
    'Motorola-Droid-Maxx',
    'Motorola-Nexus-6',
    'Motorola-X',
    'Samsung-Galaxy-Note3',
    'Samsung-Galaxy-S4',
    'Sony-NEX-7'
]
N_CLASS = len(LABELS)
ROOT_DIR = '..'
TRAIN_DIR = os.path.join(ROOT_DIR, 'data', 'train')
TEST_DIR = os.path.join(ROOT_DIR, 'data', 'test')
ID2LABEL = {i: label for i, label in enumerate(LABELS)}
LABEL2ID = {label: i for i, label in ID2LABEL.items()}
CROP_SIDE = 512

# change built-in print with tqdm.write
old_print = print


def tqdm_print(*args, **kwargs):
    try:
        tqdm.write(*args, **kwargs)
    except:
        old_print(*args, **kwargs)


inspect.builtins.print = tqdm_print


class LoggerCallback(Callback):

    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs={}):
        metrics = self.params['metrics']
        metric_format = '{name}: {value:0.4f}'
        strings = [metric_format.format(
            name=metric,
            value=np.mean(logs[metric], axis=None)
        ) for metric in metrics if metric in logs]
        epoch_output = 'Epoch {value:05d}: '.format(value=(epoch + 1))
        output = epoch_output + ', '.join(strings)
        print(output)


class ImageStorage:

    def __init__(self):
        self.images = []
        self.labels = []
        self.files = None
        pass

    def load_train_images(self):
        files = [os.path.relpath(file, TRAIN_DIR) for file in
                 glob(os.path.join(TRAIN_DIR, '*', '*'))]
        '''
        for file in tqdm(files, desc='Loading train files'):
            label = os.path.dirname(file)
            filename = os.path.basename(file)
            image = cv2.imread(os.path.join(TRAIN_DIR, label, filename))
            self.images.append(image)
            self.labels.append(label)
        '''
        with Pool() as p:
            total = len(files)
            with tqdm(total=total) as pbar:
                for i, result in tqdm(enumerate(p.map(self._load_train_image, files))):
                    image, label = result
                    self.images.append(image)
                    self.labels.append(label)
                    pbar.update()

    def _load_train_image(self, file):
        label = os.path.dirname(file)
        filename = os.path.basename(file)
        image = cv2.imread(os.path.join(TRAIN_DIR, label, filename))
        # self.images.append(image)
        # self.labels.append(label)
        return image, label

    def load_test_images(self):
        self.files = [os.path.relpath(file, TEST_DIR) for file in
                      glob(os.path.join(TEST_DIR, '*'))]
        for file in tqdm(self.files, desc='Loading test files'):
            filename = os.path.basename(file)
            image = cv2.imread(os.path.join(TEST_DIR, filename))
            self.images.append(image)

    def shuffle_data(self):
        assert len(self.images) == len(self.labels)
        data = list(zip(self.images, self.labels))
        np.random.shuffle(data)
        self.images, self.labels = zip(*data)
        self.images = list(self.images)
        self.labels = list(self.labels)


class ImageSequence(Sequence):

    def __init__(self, data, params):
        self.batch_size = params['batch_size']
        self.augment = params['augment']
        self.data = data

    def __len__(self):
        return np.ceil(len(self.data.images) / self.batch_size).astype('int')

    @abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError

    @staticmethod
    def _crop_image(image, center=False):
        h, w, _ = image.shape
        if center is True:
            h_start = np.floor_divide(h - CROP_SIDE, 2)
            w_start = np.floor_divide(w - CROP_SIDE, 2)
        else:
            h_start = np.random.randint(0, h - CROP_SIDE)
            w_start = np.random.randint(0, w - CROP_SIDE)
        return image[h_start:h_start + CROP_SIDE, w_start:w_start + CROP_SIDE]

    def _augment_image(self, image):
        return image


class TrainSequence(ImageSequence):

    def __init__(self, data, params):
        super().__init__(data, params)
        # shuffle before start
        self.on_epoch_end()
        self.balance = params['balance']

    def __getitem__(self, idx):
        x = self.data.images[idx * self.batch_size:(idx + 1) * self.batch_size]
        y = self.data.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        label_ids = [LABEL2ID[label] for label in y]
        if self.augment == 0:
            images_batch = [self._crop_image(img) for img in x]
        else:
            images_batch = [self._augment_image(img) for img in x]
        labels_batch = []
        for id_ in label_ids:
            ohe = np.zeros(N_CLASS)
            ohe[id_] = 1
            labels_batch.append(ohe)
        images_batch = np.array(images_batch)
        labels_batch = np.array(labels_batch)
        if self.balance == 0:
            return images_batch, labels_batch
        else:
            weights = compute_sample_weight('balanced', label_ids)
            return images_batch, labels_batch, weights

    def on_epoch_end(self):
        self.data.shuffle_data()


class ValSequence(ImageSequence):

    def __init__(self, data, params):
        super().__init__(data, params)
        self.balance = params['balance']

    def __getitem__(self, idx):
        x = self.data.images[idx * self.batch_size:(idx + 1) * self.batch_size]
        y = self.data.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        label_ids = [LABEL2ID[label] for label in y]
        # no augmentation on validation time
        images_batch = [self._crop_image(img, center=True) for img in x]
        labels_batch = []
        for id_ in label_ids:
            ohe = np.zeros(N_CLASS)
            ohe[id_] = 1
            labels_batch.append(ohe)
        images_batch = np.array(images_batch)
        labels_batch = np.array(labels_batch)
        if self.balance == 0:
            return images_batch, labels_batch
        else:
            weights = compute_sample_weight('balanced', label_ids)
            return images_batch, labels_batch, weights


class TestSequence(ImageSequence):

    def __init__(self, data, params):
        super().__init__(data, params)

    def __getitem__(self, idx):
        x = self.data.images[idx * self.batch_size:(idx + 1) * self.batch_size]
        # for TTA
        if self.augment == 0:
            images_batch = x
        else:
            images_batch = [self._augment_image(img) for img in x]
        images_batch = np.array(images_batch)
        return images_batch
