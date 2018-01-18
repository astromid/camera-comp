import numpy as np
import pandas as pd
import os
import argparse
from keras.models import load_model
from utils import ImageStorage, TestSequence

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name')
    parser.add_argument('--best', type=int, default=0)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--aug', type=int, default=0)
    args = parser.parse_args()

    ROOT_DIR = '..'
    MODELS_DIR = os.path.join(ROOT_DIR, 'models')
    SUB_DIR = os.path.join(ROOT_DIR, 'subs')
    SUB_PROB_DIR = os.path.join(SUB_DIR, 'probs')
    N_AUG = args.aug

    if N_AUG == 0:
        sub_end = '.csv'
    else:
        sub_end = f'-tta-{N_AUG}.csv'

    if args.best == 0:
        MODEL_PATH = os.path.join(MODELS_DIR, args.name, 'model.h5')
        SUB_PATH = os.path.join(SUB_DIR, args.name + sub_end)
        SUB_PROB_PATH = os.path.join(SUB_PROB_DIR, args.name + sub_end)
    else:
        MODEL_PATH = os.path.join(MODELS_DIR, args.name, 'model-best.h5')
        SUB_PATH = os.path.join(SUB_DIR, args.name + '-best' + sub_end)
        SUB_PROB_PATH = os.path.join(SUB_PROB_DIR, args.name + '-best' + sub_end)

    BATCH_SIZE = args.batch
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
    ID2LABEL = {i: label for i, label in enumerate(LABELS)}
    TEST_PARAMS = {
        'batch_size': BATCH_SIZE,
        'augment': 0
    }
    data = ImageStorage().load_test_images()
    test_seq = TestSequence(data, TEST_PARAMS)
    model = load_model(MODEL_PATH)
    probs = model.predict_generator(
        generator=test_seq,
        steps=len(test_seq),
        verbose=1
    )
    if N_AUG != 0:
        test_seq.augment = 1
        for _ in range(N_AUG):
            aug_probs = model.predict_generator(
                generator=test_seq,
                steps=len(test_seq),
                verbose=1
            )
            probs += aug_probs
    ids = np.argmax(probs, axis=1)
    probs_max = np.max(probs, axis=1)
    labels = [ID2LABEL[id_] for id_ in ids]
    data = {
        'fname': test_seq.data.files,
        'label': labels,
        'prob': probs_max
    }
    sub = pd.DataFrame(data)
    sub.to_csv(SUB_PROB_PATH, index=False)
    print('Submission with probs created successfully')
    sub.drop(['prob'], axis=1, inplace=True)
    sub.to_csv(SUB_PATH, index=False)
    print('Submission file created successfully')
