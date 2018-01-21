import os
import argparse
import models
import utils
from utils import ImageStorage, TrainSequence, ValSequence
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.optimizers import Adam, SGD
from keras.losses import binary_crossentropy, categorical_crossentropy
from keras.metrics import categorical_accuracy
from utils import LoggerCallback
from keras_tqdm import TQDMCallback

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name')
    # number of epochs with frozen resnet part
    parser.add_argument('--f_epochs', type=int, default=1)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--bal', type=int, default=0)
    parser.add_argument('--aug', type=int, default=0)
    args = parser.parse_args()

    MODEL_DIR = os.path.join(utils.ROOT_DIR, 'models', args.name)
    LOGS_PATH = os.path.join(MODEL_DIR, 'logs')
    F_EPOCHS = args.f_epochs
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch
    TRAIN_PARAMS = {
        'batch_size': BATCH_SIZE,
        'balance': args.bal,
        'augment': args.aug
    }
    os.makedirs(LOGS_PATH, exist_ok=True)
    data = ImageStorage()
    data.load_train_val_images(rate=0.2)

    train_seq = TrainSequence(data, TRAIN_PARAMS)
    val_seq = ValSequence(data, TRAIN_PARAMS)

    check_cb = ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, 'model-best.h5'),
        monitor='val_categorical_accuracy',
        verbose=1,
        save_best_only=True
    )
    reduce_cb = ReduceLROnPlateau(
        monitor='val_categorical_accuracy',
        factor=0.3,
        patience=5,
        verbose=1,
        epsilon=0.001,
        cooldown=3,
        min_lr=1e-6
    )
    tb_cb = TensorBoard(LOGS_PATH, batch_size=BATCH_SIZE)
    log_cb = LoggerCallback()
    tqdm_cb = TQDMCallback(leave_inner=False)
    model = models.resnet50()
    if F_EPOCHS != 0:
        # train with frozen resnet block
        model.compile(
            optimizer=Adam(),
            loss=binary_crossentropy,
            metrics=[categorical_accuracy]
            # weighted_metrics=[categorical_accuracy]
        )
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
        # defrost resnet block
        for layer in model.get_layer('resnet50').layers:
            layer.trainable = True
        model.compile(
            optimizer=Adam(),
            loss=binary_crossentropy,
            metrics=[categorical_accuracy]
            # weighted_metrics=[categorical_accuracy]
        )
        hist = model.fit_generator(
            generator=train_seq,
            steps_per_epoch=len(train_seq),
            epochs=EPOCHS,
            verbose=0,
            callbacks=[check_cb, reduce_cb, tb_cb, log_cb, tqdm_cb],
            validation_data=val_seq,
            validation_steps=len(val_seq),
            initial_epoch=F_EPOCHS
        )
    model.save(os.path.join(MODEL_DIR, 'model.h5'))
    print('Model saved successfully')

