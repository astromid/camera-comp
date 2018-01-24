import os
import inspect
import cv2
import numpy as np
import keras.backend as K
from glob import glob
from keras.utils import Sequence
from keras.callbacks import Callback, ReduceLROnPlateau
from tqdm import tqdm
from abc import abstractmethod
from sklearn.utils.class_weight import compute_sample_weight
from multiprocessing.pool import ThreadPool
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
# unalt <-> 0, manip <-> 1
AUG_WEIGHTS = {0: 7, 1: 3}

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
        metric_format = '{name}: {value:0.5f}'
        strings = [metric_format.format(
            name=metric,
            value=np.mean(logs[metric], axis=None)
        ) for metric in metrics if metric in logs]
        epoch_output = 'Epoch {value:05d}: '.format(value=(epoch + 1))
        output = epoch_output + ', '.join(strings)
        print(output)


class CycleReduceLROnPlateau(ReduceLROnPlateau):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_lr = K.get_value(self.model.optimizer.lr)
        self.min_lr_counter = 0

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        new_lr = K.get_value(self.model.optimizer.lr)
        if new_lr == self.min_lr:
            self.min_lr_counter += 1
        if self.min_lr_counter >= 1.5 * self.patience:
            K.set_value(self.model.optimizer.lr, self.start_lr)
            if self.verbose > 0:
                print('\nEpoch %05d: returning to starting learning rate %s.' % (epoch + 1, self.start_lr))
            self.cooldown_counter = self.cooldown
            self.wait = 0


class ImageStorage:

    def __init__(self):
        self.images = []
        self.labels = []
        self.val_images = []
        self.val_labels = []
        self.files = []

    def load_train_val_images(self, rate):
        train_files, val_files = self._list_train_val_files(rate)
        with ThreadPool() as p:
            total = len(train_files)
            with tqdm(desc='Loading train files', total=total) as pbar:
                for results in p.imap_unordered(self._load_train_image, train_files):
                    images, labels = results
                    self.images.append(images)
                    self.labels.append(labels)
                    pbar.update()
            total = len(val_files)
            with tqdm(desc='Loading validation files', total=total) as pbar:
                for results in p.imap_unordered(self._load_train_image, val_files):
                    images, labels = results
                    self.val_images.append(images)
                    self.val_labels.append(labels)
                    pbar.update()

    def load_test_images(self):
        files = [os.path.relpath(file, TEST_DIR) for file in
                 glob(os.path.join(TEST_DIR, '*'))]
        with ThreadPool() as p:
            total = len(files)
            with tqdm(desc='Loading test files', total=total) as pbar:
                for results in p.imap_unordered(self._load_test_image, files):
                    images, filenames = results
                    self.images.append(images)
                    self.files.append(filenames)
                    pbar.update()

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
        return image.astype(np.uint8), filename

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
        self.len_ = 0

    def __len__(self):
        return np.ceil(self.len_ / self.batch_size).astype('int')

    @abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError

    @staticmethod
    def _crop_image(args):
        image, side_len, center = args
        h, w, _ = image.shape
        if center is False:
            h_start = np.random.randint(0, h - side_len)
            w_start = np.random.randint(0, w - side_len)
        else:
            h_start = np.floor_divide(h - side_len, 2)
            w_start = np.floor_divide(w - side_len, 2)
        return image[h_start:h_start + side_len, w_start:w_start + side_len].copy()

    @staticmethod
    def _augment_image(args):
        image, center = args
        h, w, _ = image.shape
        status = 0
        # default augmentations (only 1 from 8)
        if np.random.rand() < 0.5:
            status = 1
            flag = np.random.choice(8)
            if flag == 0:
                aug_image = ImageSequence._crop_image((image, CROP_SIDE, center))
                enc_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
                _, aug_image = cv2.imencode('.jpg', aug_image, enc_param)
                aug_image = cv2.imdecode(aug_image, 1)
            elif flag == 1:
                aug_image = ImageSequence._crop_image((image, CROP_SIDE, center))
                enc_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                _, aug_image = cv2.imencode('.jpg', aug_image, enc_param)
                aug_image = cv2.imdecode(aug_image, 1)
            elif flag == 2:
                side_len = np.ceil(CROP_SIDE / 0.5).astype('int')
                if side_len < h and side_len < w:
                    aug_image = ImageSequence._crop_image((image, side_len, center))
                    aug_image = cv2.resize(aug_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
                else:
                    status = 0
                    aug_image = ImageSequence._crop_image((image, CROP_SIDE, center))
            elif flag == 3:
                side_len = np.ceil(CROP_SIDE / 0.8).astype('int')
                if side_len < h and side_len < w:
                    aug_image = ImageSequence._crop_image((image, side_len, center))
                    aug_image = cv2.resize(aug_image, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_CUBIC)
                else:
                    status = 0
                    aug_image = ImageSequence._crop_image((image, CROP_SIDE, center))
            elif flag == 4:
                side_len = np.ceil(CROP_SIDE / 1.5).astype('int')
                aug_image = ImageSequence._crop_image((image, side_len, center))
                aug_image = cv2.resize(aug_image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
                aug_image = ImageSequence._crop_image((aug_image, CROP_SIDE, center))
            elif flag == 5:
                side_len = np.ceil(CROP_SIDE / 2.0).astype('int')
                aug_image = ImageSequence._crop_image((image, side_len, center))
                aug_image = cv2.resize(aug_image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            elif flag == 6:
                aug_image = ImageSequence._crop_image((image, CROP_SIDE, center))
                aug_image = adjust_gamma(aug_image, 0.8)
            else:
                aug_image = ImageSequence._crop_image((image, CROP_SIDE, center))
                aug_image = adjust_gamma(aug_image, 1.2)
        else:
            aug_image = ImageSequence._crop_image((image, CROP_SIDE, center))
        # additional augmentations
        if np.random.rand() < 0.5:
            n_rotate = np.random.choice([1, 2, 3])
            for _ in range(n_rotate):
                aug_image = np.rot90(aug_image)
        if np.random.rand() < 0.5:
            k_size = np.random.choice([3, 5])
            aug_image = cv2.GaussianBlur(aug_image, (k_size, k_size), 0)
        try:
            assert aug_image.shape == (CROP_SIDE, CROP_SIDE, 3)
        except AssertionError:
            print('Assertion error in augment: ', aug_image.shape)
            raise AssertionError
        return aug_image, status


class TrainSequence(ImageSequence):

    def __init__(self, data, params):
        super().__init__(data, params)
        # shuffle before start
        self.on_epoch_end()
        self.weights = params['weights']
        self.len_ = len(self.data.images)

    def __getitem__(self, idx):
        x = self.data.images[idx * self.batch_size:(idx + 1) * self.batch_size]
        y = self.data.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        label_ids = [LABEL2ID[label] for label in y]
        images_batch = []
        images_status = []
        with ThreadPool() as p:
            args = list(zip(x, [False] * len(x)))
            if self.augment == 0:
                for images in p.imap(self._crop_image, args):
                    images_batch.append(images)
            else:
                for results in p.imap(self._augment_image, args):
                    images, status = results
                    images_batch.append(images)
                    images_status.append(status)
        labels_batch = []
        for id_ in label_ids:
            ohe = np.zeros(N_CLASS)
            ohe[id_] = 1
            labels_batch.append(ohe)
        images_batch = np.array(images_batch)
        labels_batch = np.array(labels_batch)
        if self.weights == 0:
            return images_batch, labels_batch
        else:
            weights = compute_sample_weight(AUG_WEIGHTS, images_status)
            return images_batch, labels_batch, weights

    def on_epoch_end(self):
        self.data.shuffle_train_data()


class ValSequence(ImageSequence):

    def __init__(self, data, params):
        super().__init__(data, params)
        self.weights = params['weights']
        self.len_ = len(self.data.val_images)

    def __getitem__(self, idx):
        x = self.data.val_images[idx * self.batch_size:(idx + 1) * self.batch_size]
        y = self.data.val_labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        label_ids = [LABEL2ID[label] for label in y]
        args = list(zip(x, [True] * len(x)))
        images_batch = []
        images_status = []
        with ThreadPool() as p:
            if self.augment == 0:
                for images in p.imap(self._crop_image, args):
                    images_batch.append(images)
            else:
                for results in p.imap(self._augment_image, args):
                    images, status = results
                    images_batch.append(images)
                    images_status.append(status)
        labels_batch = []
        for id_ in label_ids:
            ohe = np.zeros(N_CLASS)
            ohe[id_] = 1
            labels_batch.append(ohe)
        images_batch = np.array(images_batch)
        labels_batch = np.array(labels_batch)
        if self.weights == 0:
            return images_batch, labels_batch
        else:
            weights = compute_sample_weight(AUG_WEIGHTS, images_status)
            return images_batch, labels_batch, weights


class TestSequence(ImageSequence):

    def __init__(self, data, params):
        super().__init__(data, params)
        self.len_ = len(self.data.images)

    def __getitem__(self, idx):
        x = self.data.images[idx * self.batch_size:(idx + 1) * self.batch_size]
        # for TTA
        images_batch = []
        if self.augment == 0:
            images_batch = x
        else:
            with ThreadPool() as p:
                for images in p.imap(self._augment_image, x):
                    images_batch.append(images)
        images_batch = np.array(images_batch)
        return images_batch

    @staticmethod
    def _augment_image(image, center=False):
        # only additional augmentations for TTA
        aug_image = image.copy()
        if np.random.rand() < 0.5:
            n_rotate = np.random.choice([1, 2, 3])
            for _ in range(n_rotate):
                aug_image = np.rot90(aug_image)
        if np.random.rand() < 0.5:
            k_size = np.random.choice([3, 5])
            aug_image = cv2.GaussianBlur(aug_image, (k_size, k_size), 0)
        try:
            assert aug_image.shape == (CROP_SIDE, CROP_SIDE, 3)
        except AssertionError:
            print('Assertion error in augment: ', aug_image.shape)
            raise AssertionError
        return aug_image

