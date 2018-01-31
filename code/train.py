import os
import sys
import argparse
import models
import utils
import signal
from glob import glob
from utils import TrainSequence, ValSequence
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.metrics import categorical_accuracy
from keras.models import load_model
from utils import LoggerCallback, CycleReduceLROnPlateau
from keras_tqdm import TQDMCallback
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


# try to handle gcp instance stopping
def handler(signal, frame):
    model.save(MODEL_PATH + '.h5')
    print('Got SIGTERM (maybe GCP instance is going to shutdown), model saved successfully')
    sys.exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', help='Name of the network in format {base_clf}-{number}')
    parser.add_argument('-e', '--epochs', type=int, help='Total # of epochs to train')
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('-l', '--load', type=int, default=0, help='Load model to continue training from a given epoch')
    parser.add_argument('-fe', '--f_epochs', type=int, default=0, help='# of epochs w/ frozen base model')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('-f', '--folds', type=int, default=1, help='# of folds for KFold / # tries for bagging')
    parser.add_argument('-bag', '--bagging', type=int, default=0, help='# of train samples for bagging')
    parser.add_argument('-cf', '--current_fold', type=int, default=1, help='Which fold is used for training')
    parser.add_argument('-s', '--seed', type=int, default=12017952)
    parser.add_argument('-x', '--extra', action='store_true', help='Enables extra train data')
    parser.add_argument('-vx', '--val_extra', action='store_true', help='Enables extra validation data')
    parser.add_argument('-bal', '--balance', action='store_true', help='Enables sample balancing')
    parser.add_argument('-aug', '--augmentation', action='store_true', help='Enables augmentation during training')
    parser.add_argument('-vl', '--val_length', type=int, default=0, help='Length of random validation subset')
    args = parser.parse_args()

    if args.extra and (args.folds < 3 or not args.bagging):
        print('No way to load entire extra dataset into RAM. Use folds or bagging modes')
        raise MemoryError

    MODEL_DIR = os.path.join(utils.ROOT_DIR, 'models', args.name)
    CLF_NAME = args.name.split('-')[0]
    TRAIN_CONFIG = {
        'batch_size': args.batch_size,
        'balance': args.balance,
        'augmentation': args.augmentation,
        'val_length': args.val_length,
        'clf_name': CLF_NAME}
    os.makedirs(MODEL_DIR, exist_ok=True)
    all_train_files = sorted([os.path.relpath(file, utils.TRAIN_DIR) for file in
                              glob(os.path.join(utils.TRAIN_DIR, '*', '*'))])
    if not args.extra:
        # all filenames in original dataset start with '('
        all_train_files = [file for file in all_train_files if os.path.basename(file).startswith('(')]
    extra_val_files = sorted([os.path.relpath(file, utils.VAL_DIR) for file in
                              glob(os.path.join(utils.VAL_DIR, '*', '*'))])
    if args.folds > 1 or args.bagging:
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        if args.bagging:
            # reversed in case of compability with SKF
            skf = StratifiedShuffleSplit(n_splits=args.folds, test_size=args.bagging, random_state=args.seed)
        labels = [os.path.dirname(file) for file in all_train_files]
        train_idxs = []
        val_idxs = []
        # val & train idx reversed - train on one fold, evaluate on (n-1)
        for val_idx, train_idx in skf.split(all_train_files, labels):
            train_idxs.append(train_idx)
            val_idxs.append(val_idx)
        train_idx = train_idxs[args.current_fold - 1]
        train_files = [all_train_files[idx] for idx in train_idx]
        if not args.val_extra:
            val_idx = val_idxs[args.current_fold - 1]
            val_files = [all_train_files[idx] for idx in val_idx]
            # need to redirect directory
            utils.VAL_DIR = utils.TRAIN_DIR
        else:
            val_files = extra_val_files
        model_name = f'fold{args.current_fold}'
        print(f'Current fold {args.current_fold}')
    else:
        train_files = all_train_files
        val_files = extra_val_files
        model_name = 'model'

    MODEL_PATH = os.path.join(MODEL_DIR, model_name)

    signal.signal(signal.SIGTERM, handler)

    train_seq = TrainSequence(train_files, TRAIN_CONFIG)
    val_seq = ValSequence(val_files, TRAIN_CONFIG)
    model_args = {
        'optimizer': Adam(lr=args.learning_rate),
        'loss': binary_crossentropy}
    # if args.balance == 0:
    monitor = 'val_categorical_accuracy'
    model_args['metrics'] = [categorical_accuracy]
    # else:
    #     monitor = 'val_weighted_categorical_accuracy'
    #     model_args['weighted_metrics'] = [categorical_accuracy]

    check_cb = ModelCheckpoint(
        filepath=MODEL_PATH + '-best.h5',
        monitor=monitor,
        verbose=1,
        save_best_only=True)
    cycle_cb = CycleReduceLROnPlateau(
        filepath=MODEL_PATH,
        monitor=monitor,
        factor=0.3,
        patience=6,
        verbose=1,
        epsilon=0.0001,
        min_lr=1e-8)
    tb_cb = TensorBoard(MODEL_DIR, batch_size=args.batch_size)
    log_cb = LoggerCallback()
    tqdm_cb = TQDMCallback(leave_inner=False)
    cb_f = [log_cb, tqdm_cb]
    cb_e = [check_cb, cycle_cb, tb_cb, log_cb, tqdm_cb]
    if args.load:
        model = load_model(MODEL_PATH + '.h5')
        print(f'Successfully loaded model, continue training from epoch {args.load}')
        model.fit_generator(
            generator=train_seq,
            steps_per_epoch=len(train_seq),
            epochs=args.epochs,
            verbose=0,
            callbacks=cb_e,
            validation_data=val_seq,
            validation_steps=len(val_seq),
            initial_epoch=args.load)
    else:
        if CLF_NAME in utils.CLF2MODULE:
            model = models.pretrained_model(CLF_NAME)
            model = models.train_pretrained_model(
                clf_name=CLF_NAME,
                model=model,
                train=train_seq,
                val=val_seq,
                model_args=model_args,
                f_epochs=args.f_epochs,
                epochs=args.epochs,
                cb_f=cb_f,
                cb_e=cb_e)
        elif CLF_NAME in utils.NONPRETRAINED_NETS:
            model = models.SeResNet3().model
            model.compile(**model_args)
            model.summary()
            model.fit_generator(
                generator=train_seq,
                steps_per_epoch=len(train_seq),
                epochs=args.epochs,
                verbose=0,
                callbacks=cb_e,
                validation_data=val_seq,
                validation_steps=len(val_seq))
        else:
            print('Can\'t found suitable model in models.py')
            raise NameError
    model.save(MODEL_PATH + '.h5')
    print('Model saved successfully')
