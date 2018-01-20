import os
import inspect
import cv2
import numpy as np
from glob import glob
from keras.utils import Sequence
from keras.callbacks import Callback
from tqdm import tqdm
from abc import abstractmethod
from sklearn.utils.class_weight import compute_sample_weight
from multiprocessing import Pool
from skimage.exposure import adjust_gamma
from sklearn.model_selection import train_test_split

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

# change built-in print with tqdm_print
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
        self.val_images = []
        self.val_labels = []
        self.files = None

    def load_train_val_images(self, rate):
        train_files, val_files = self._list_train_val_files(rate)
        with Pool() as p:
            total = len(train_files)
            with tqdm(desc='Loading train files', total=total) as pbar:
                for results in p.imap_unordered(self._load_train_image, train_files, chunksize=2):
                    images, labels = results
                    self.images.append(images)
                    self.labels.append(labels)
                    pbar.update()
            total = len(val_files)
            with tqdm(desc='Loading validation files', total=total) as pbar:
                for results in p.imap_unordered(self._load_train_image, val_files, chunksize=2):
                    images, labels = results
                    self.val_images.append(images)
                    self.val_labels.append(labels)
                    pbar.update()

    def load_test_images(self):
        files = [os.path.relpath(file, TEST_DIR) for file in
                 glob(os.path.join(TEST_DIR, '*'))]
        with Pool() as p:
            total = len(files)
            with tqdm(desc='Loading test files', total=total) as pbar:
                for images in p.imap_unordered(self._load_test_image, files, chunksize=2):
                    self.images.append(images)
                    pbar.update()
        self.files = files

    def shuffle_train_data(self):
        assert len(self.images) == len(self.labels)
        data = list(zip(self.images, self.labels))
        np.random.shuffle(data)
        self.images, self.labels = zip(*data)
        self.images = list(self.images)
        self.labels = list(self.labels)

    @staticmethod
    def _load_train_image(file):
        label = os.path.dirname(file)
        filename = os.path.basename(file)
        image = cv2.imread(os.path.join(TRAIN_DIR, label, filename))
        return image.astype(np.uint8), label

    @staticmethod
    def _load_test_image(file):
        filename = os.path.basename(file)
        image = cv2.imread(os.path.join(TEST_DIR, filename))
        return image.astype(np.uint8)

    @staticmethod
    def _list_train_val_files(rate):
        files = [os.path.relpath(file, TRAIN_DIR) for file in
                 glob(os.path.join(TRAIN_DIR, '*', '*'))]
        labels = [os.path.dirname(file) for file in files]
        train_files, val_files = train_test_split(files, test_size=rate, stratify=labels)
        return train_files, val_files



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

    @staticmethod
    def _augment_image(image):
        # default augmentations (only 1 from 8)
        if np.random.rand() < 0.5:
            flag = np.random.choice(8)
            if flag == 0:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
                _, image = cv2.imencode('.jpg', image, encode_param)
                image = cv2.imdecode(image, 1)
            if flag == 1:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                _, image = cv2.imencode('.jpg', image, encode_param)
                image = cv2.imdecode(image, 1)
            if flag == 2:
                image = cv2.resize(image, dsize=0, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
            if flag == 3:
                image = cv2.resize(image, dsize=0, fx=0.8, fy=0.8, interpolation=cv2.INTER_CUBIC)
            if flag == 4:
                image = cv2.resize(image, dsize=0, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
            if flag == 5:
                image = cv2.resize(image, dsize=0, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            if flag == 6:
                image = adjust_gamma(image, 0.8)
            if flag == 7:
                image = adjust_gamma(image, 1.2)
        # additional augmentations
        if np.random.rand() < 0.5:
            n_rotate = np.random.choice([1, 2, 3])
            for _ in range(n_rotate):
                image = np.rot90(image)
        if np.random.rand() < 0.5:
            k_size = np.random.choice([2, 3, 4, 5])
            image = cv2.GaussianBlur(image, (k_size, k_size), 0)
        return image

    '''
    @staticmethod
    def _normalize_image(image, mean):
        _, _, ch = image.shape
        for i in range(ch):
            image[:, :, i] -= mean[i]
        return image
    '''


class TrainSequence(ImageSequence):

    def __init__(self, data, params):
        super().__init__(data, params)
        # calculate mean by channel across training dataset
        # self.mean = np.mean(self.data.images, axis=(0, 1, 2))
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
            images_batch = [self._crop_image(img) for img in images_batch]
        # images_batch = [self._normalize_image(img, self.mean) for img in images_batch]
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
        self.data.shuffle_train_data()


class ValSequence(ImageSequence):

    # def __init__(self, data, mean, params):
    def __init__(self, data, params):
        super().__init__(data, params)
        # self.mean = mean
        self.balance = params['balance']

    def __getitem__(self, idx):
        x = self.data.val_images[idx * self.batch_size:(idx + 1) * self.batch_size]
        y = self.data.val_labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        label_ids = [LABEL2ID[label] for label in y]
        # no augmentation on validation time
        images_batch = [self._crop_image(img, center=True) for img in x]
        # images_batch = [self._normalize_image(img, self.mean) for img in images_batch]
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

    # def __init__(self, data, mean, params):
    def __init__(self, data, params):
        super().__init__(data, params)
        # self.mean = mean

    def __getitem__(self, idx):
        x = self.data.images[idx * self.batch_size:(idx + 1) * self.batch_size]
        # for TTA
        if self.augment == 0:
            images_batch = x
        else:
            images_batch = [self._augment_image(img) for img in x]
        # images_batch = [self._normalize_image(img, self.mean) for img in images_batch]
        images_batch = np.array(images_batch)
        return images_batch
