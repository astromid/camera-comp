import os
import argparse
import models
import utils
from glob import glob
from utils import TrainSequence, ValSequence
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.metrics import categorical_accuracy
from utils import LoggerCallback, CycleReduceLROnPlateau
from keras_tqdm import TQDMCallback

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch', type=int)
    # number of epochs with frozen pretrained part
    parser.add_argument('--f_epochs', type=int, default=1)
    parser.add_argument('--bal', type=int, default=0)
    parser.add_argument('--aug', type=int, default=0)
    args = parser.parse_args()

    MODEL_DIR = os.path.join(utils.ROOT_DIR, 'models', args.name)
    F_EPOCHS = args.f_epochs
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch
    TRAIN_PARAMS = {
        'batch_size': BATCH_SIZE,
        'balance': args.bal,
        'augment': args.aug
    }
    os.makedirs(MODEL_DIR, exist_ok=True)
    train_files = [os.path.relpath(file, utils.TRAIN_DIR) for file in
                   glob(os.path.join(utils.TRAIN_DIR, '*', '*'))]
    val_files = [os.path.relpath(file, utils.VAL_DIR) for file in
                 glob(os.path.join(utils.VAL_DIR, '*', '*'))]

    train_seq = TrainSequence(train_files, TRAIN_PARAMS)
    val_seq = ValSequence(val_files, TRAIN_PARAMS)

    model_args = {
        'optimizer': Adam(lr=1e-4),
        'loss': binary_crossentropy,
    }
    if args.bal == 0:
        monitor = 'val_categorical_accuracy'
        model_args['metrics'] = [categorical_accuracy]
    else:
        monitor = 'val_weighted_categorical_accuracy'
        model_args['weighted_metrics'] = [categorical_accuracy]

    check_cb = ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, 'model-best.h5'),
        monitor=monitor,
        verbose=1,
        save_best_only=True
    )
    cycle_cb = CycleReduceLROnPlateau(
        monitor=monitor,
        factor=0.25,
        patience=5,
        verbose=1,
        epsilon=0.0001,
        min_lr=1e-8
    )
    tb_cb = TensorBoard(MODEL_DIR, batch_size=BATCH_SIZE)
    log_cb = LoggerCallback()
    tqdm_cb = TQDMCallback(leave_inner=False)
    model = models.resnet50()
    if F_EPOCHS != 0:
        # train with frozen pretrained block
        model.compile(model_args)
        hist_f = model.fit_generator(
            generator=train_seq,
            steps_per_epoch=len(train_seq),
            epochs=F_EPOCHS,
            verbose=0,
            callbacks=[log_cb, tqdm_cb],
            validation_data=val_seq,
            validation_steps=len(val_seq)
        )
    if EPOCHS > F_EPOCHS:
        # defrost pretrained block
        for layer in model.layers:
            layer.trainable = True
        model.compile(model_args)
        hist = model.fit_generator(
            generator=train_seq,
            steps_per_epoch=len(train_seq),
            epochs=EPOCHS,
            verbose=0,
            callbacks=[check_cb, cycle_cb, tb_cb, log_cb, tqdm_cb],
            validation_data=val_seq,
            validation_steps=len(val_seq),
            initial_epoch=F_EPOCHS
        )
    model.save(os.path.join(MODEL_DIR, 'model.h5'))
    print('Model saved successfully')
