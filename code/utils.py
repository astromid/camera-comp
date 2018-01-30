import os
import inspect
import cv2
import numpy as np
import keras.backend as K
from glob import glob
from keras.utils import Sequence
from keras.callbacks import Callback, ReduceLROnPlateau
from tqdm import tqdm
from sklearn.utils.class_weight import compute_sample_weight
from multiprocessing.pool import ThreadPool
from skimage.exposure import adjust_gamma
from numba import jit
from keras.applications import *

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
    'Sony-NEX-7']
N_CLASS = len(LABELS)
ROOT_DIR = '..'
TRAIN_DIR = os.path.join(ROOT_DIR, 'data', 'train')
VAL_DIR = os.path.join(ROOT_DIR, 'data', 'val')
TEST_DIR = os.path.join(ROOT_DIR, 'data', 'test')
ID2LABEL = {i: label for i, label in enumerate(LABELS)}
LABEL2ID = {label: i for i, label in ID2LABEL.items()}
CROP_SIDE = 512
CLF2MODULE = {
    'densenet40': 'densenet',
    'densenet121': 'densenet',
    'densenet161': 'densenet',
    'densenet201': 'densenet',
    'resnet50': 'resnet50',
    'xception': 'xception'}
CLF2CLASS = {
    'densenet40': 'DenseNet40',
    'densenet121': 'DenseNet121',
    'densenet161': 'DenseNet161',
    'densenet201': 'DenseNet201',
    'resnet50': 'ResNet50',
    'xception': 'Xception'}
NONPRETRAINED_NETS = ['seresnet']

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

    def __init__(self, filepath, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_lr = None
        self.min_lr_counter = 0
        self.cycle_counter = 0
        self.filepath = filepath

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        self.start_lr = float(K.get_value(self.model.optimizer.lr)) / 5

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        lr = float(K.get_value(self.model.optimizer.lr))
        if np.isclose(lr, self.min_lr):
            self.min_lr_counter += 1
        if self.min_lr_counter >= 2 * self.patience:
            K.set_value(self.model.optimizer.lr, self.start_lr)
            self.min_lr /= 5
            self.patience += 1
            self.cooldown = 0
            self.min_lr_counter = 0
            self.cycle_counter += 1
            self.model.save(self.filepath + f'-cycle{self.cycle_counter}.h5')
            if self.verbose > 0:
                print('Epoch %05d: Cycle returning to learning rate %s.' % (epoch + 1, self.start_lr))
                print('Epoch %05d: Model snapshot successfully saved' % (epoch + 1))
            self.cooldown_counter = self.cooldown
            self.wait = 0


class ImageSequence(Sequence):

    def __init__(self, params):
        self.images = []
        self.labels = []
        self.len_ = 0
        self.center = None
        self.balance = None
        self.batch_size = params['batch_size']
        self.augmentation = params['augmentation']
        self.clf_name = params['clf_name']
        if self.clf_name in CLF2MODULE:
            print(f'Using preprocess function for {self.clf_name}')
            module_name = CLF2MODULE[self.clf_name]
            self._preprocess_batch = getattr(globals()[module_name], 'preprocess_input')
        elif self.clf_name in NONPRETRAINED_NETS:
            print('Non-pretrained model found, using identity preprocess function')
            self._preprocess_batch = lambda x: x
        else:
            print('Can\'t found suitable preprocess function')
            raise NameError
        self.p = ThreadPool()

    def __len__(self):
        return np.ceil(self.len_ / self.batch_size).astype('int')

    def __getitem__(self, idx):
        x = self.images[idx * self.batch_size:(idx + 1) * self.batch_size]
        y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        label_ids = [LABEL2ID[label] for label in y]
        images_batch = []
        manip_flags = []
        args = list(zip(x, [self.center] * len(x)))
        for result in self.p.imap(self._prepare_image, args):
            image, manip_flag = result
            images_batch.append(image)
            manip_flags.append(manip_flag)
        if self.augmentation:
            augmented_batch = []
            for image in self.p.imap(self._augment_image, images_batch):
                augmented_batch.append(image)
            images_batch = augmented_batch
        labels_batch = []
        for id_ in label_ids:
            ohe = np.zeros(N_CLASS)
            ohe[id_] = 1
            labels_batch.append(ohe)
        images_batch = self._preprocess_batch(np.array(images_batch).astype(np.float32))
        manip_flags = np.array(manip_flags)
        labels_batch = np.array(labels_batch)
        if self.balance:
            weights = compute_sample_weight('balanced', label_ids)
            return [images_batch, manip_flags], labels_batch, weights
        else:
            return [images_batch, manip_flags], labels_batch

    @staticmethod
    @jit
    def _crop_image(args):
        image, side_len, center = args
        h, w, _ = image.shape
        if h == side_len and w == side_len:
            return image.copy()
        assert h > side_len and w > side_len
        if center:
            h_start = np.floor_divide(h - side_len, 2)
            w_start = np.floor_divide(w - side_len, 2)
        else:
            h_start = np.random.randint(0, h - side_len)
            w_start = np.random.randint(0, w - side_len)
        return image[h_start:h_start + side_len, w_start:w_start + side_len].copy()

    @staticmethod
    @jit
    def _prepare_image(args):
        image, center = args
        if np.random.rand() < 0.3:
            manip_image = ImageSequence._crop_image((image, 2 * CROP_SIDE, center))
            manip_flag = np.random.choice([0, 0, 1, 1, 1, 1, 2, 2])
            if manip_flag == 0:
                rate = np.random.choice([70, 90])
                enc_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(rate)]
                _, encoded_image = cv2.imencode('.jpg', manip_image, enc_param)
                manip_image = cv2.imdecode(encoded_image, 1)
            elif manip_flag == 1:
                scale = np.random.choice([0.5, 0.8, 1.5, 2.0])
                manip_image = cv2.resize(manip_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            else:
                gamma = np.random.choice([0.8, 1.2])
                manip_image = adjust_gamma(manip_image, gamma)
            manip_image = ImageSequence._crop_image((manip_image, CROP_SIDE, center))
            manip_flag = 1.
        else:
            manip_image = ImageSequence._crop_image((image, CROP_SIDE, center))
            manip_flag = 0.
        return manip_image, manip_flag

    @staticmethod
    @jit
    def _augment_image(image):
        aug_image = image
        if np.random.rand() < 0.5:  # 0.66
            # axis_ = np.random.randint(0, 2)
            # aug_image = np.flip(aug_image, axis_)
            aug_image = np.rot90(aug_image, 1, (0, 1))
        if np.random.rand() < 0.5:  # 0.66
            # k_size = np.random.choice([3, 5])
            k_size = 3
            aug_image = cv2.GaussianBlur(aug_image, (k_size, k_size), 0)
        return aug_image


class TrainSequence(ImageSequence):

    def __init__(self, files, params):
        super().__init__(params)
        self.center = False
        self.balance = params['balance']
        self.load_images(files)
        # shuffle before start
        self.on_epoch_end()

    def on_epoch_end(self):
        data = list(zip(self.images, self.labels))
        np.random.shuffle(data)
        self.images, self.labels = zip(*data)
        self.images = list(self.images)
        self.labels = list(self.labels)

    def load_images(self, files):
        with tqdm(desc='Loading train files', total=len(files)) as pbar:
            for result in self.p.imap(self._load_image, files):
                image, label = result
                if image is not None:
                    self.images.append(image)
                    self.labels.append(label)
                pbar.update()
        self.len_ = len(self.images)
        print(f'Successfully loaded {self.len_} train images')

    @staticmethod
    def _load_image(file):
        label = os.path.dirname(file)
        filename = os.path.basename(file)
        image = cv2.imread(os.path.join(TRAIN_DIR, label, filename))
        # if image don't have 3rd channel or even not an image at all :)
        try:
            h, w, ch = image.shape
        except ValueError:
            return None, None
        except AttributeError:
            return None, None
        # discard some strange images with shape (_, _, 4)
        if h < 2 * CROP_SIDE or w < 2 * CROP_SIDE or ch != 3:
            return None, None
        else:
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image, label


class ValSequence(ImageSequence):

    def __init__(self, files, params):
        super().__init__(params)
        self.center = True
        self.balance = params['balance']
        self.val_len = params['val_length']
        self.load_images(files)
        # use subsample of entire validation set and shuffle it
        if self.val_len and self.val_len < self.len_:
            self.len_ = self.val_len
            self.on_epoch_end()
            print(f'Using random validation subset of length {self.len_}')
        else:
            print(f'Using entire validation set of length {self.len_}')

    def on_epoch_end(self):
        if self.val_len:
            data = list(zip(self.images, self.labels))
            np.random.shuffle(data)
            self.images, self.labels = zip(*data)
            self.images = list(self.images)
            self.labels = list(self.labels)

    def load_images(self, files):
        with tqdm(desc='Loading validation files', total=len(files)) as pbar:
            for result in self.p.imap(self._load_image, files):
                image, label = result
                if image is not None:
                    self.images.append(image)
                    self.labels.append(label)
                pbar.update()
        self.len_ = len(self.images)
        print(f'Successfully loaded {self.len_} validation images')

    @staticmethod
    def _load_image(file):
        label = os.path.dirname(file)
        filename = os.path.basename(file)
        image = cv2.imread(os.path.join(VAL_DIR, label, filename))
        # if image don't have 3rd channel or even not an image at all :)
        try:
            h, w, ch = image.shape
        except ValueError:
            return None, None
        except AttributeError:
            return None, None
        # discard some strange images with shape (_, _, 4)
        if h < 2 * CROP_SIDE or w < 2 * CROP_SIDE or ch != 3:
            return None, None
        else:
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return ImageSequence._crop_image((image, 2 * CROP_SIDE, True)), label


class TestSequence(ImageSequence):

    def __init__(self, params):
        super().__init__(params)
        self.files = []
        self.manip_flags = []
        self.load_test_images()

    def __getitem__(self, idx):
        x = self.images[idx * self.batch_size:(idx + 1) * self.batch_size]
        y = self.manip_flags[idx * self.batch_size:(idx + 1) * self.batch_size]
        # for TTA
        if self.augmentation:
            images_batch = []
            args = list(zip(x, [self.augmentation] * len(x)))
            for image in self.p.imap(self._augment_image, args):
                images_batch.append(image)
        else:
            images_batch = x
        images_batch = self._preprocess_batch(np.array(images_batch).astype(np.float32))
        manip_flags = np.array(y)
        return [images_batch, manip_flags]

    def load_test_images(self):
        files = sorted([os.path.relpath(file, TEST_DIR) for file in
                        glob(os.path.join(TEST_DIR, '*'))])
        self.len_ = len(files)
        with tqdm(desc='Loading test files', total=self.len_) as pbar:
            for result in self.p.imap(self._load_image, files):
                image, filename = result
                manip_flag = [1. if filename.find('manip') != -1 else 0.][0]
                self.images.append(image)
                self.files.append(filename)
                self.manip_flags.append(manip_flag)
                pbar.update()

    @staticmethod
    def _load_image(file):
        filename = os.path.basename(file)
        image = cv2.imread(os.path.join(TEST_DIR, filename))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, filename

    @staticmethod
    @jit
    def _augment_image(args):
        image, aug_flag = args
        if aug_flag == 1:
            # return np.flip(image, 0)
            return np.rot90(image, 1, (0, 1))
        # elif aug_flag == 2:
        #     return np.flip(image, 1)
        elif aug_flag == 2:  # 3
            return cv2.GaussianBlur(image, (3, 3), 0)
        # elif aug_flag == 4:
        #    return cv2.GaussianBlur(image, (5, 5), 0)
        elif aug_flag == 3:
            aug_image = np.rot90(image, 1, (0, 1))
            return cv2.GaussianBlur(aug_image, (3, 3), 0)
