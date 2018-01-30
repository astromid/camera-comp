import os
import argparse
import utils
import numpy as np
import pandas as pd
from keras.models import load_model
from utils import TestSequence
from glob import glob
from keras_contrib.layers import BatchRenormalization

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', help='Name of the network in format {base_clf}-{number}')
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('-f', '--folds', action='store_true', help='Average predictions from folded-train network')
    parser.add_argument('-best', action='store_true', help='Use models which were the best on validation set')
    parser.add_argument('-all', action='store_true', help='Use all snapshots in ensemble')
    parser.add_argument('-tta', action='store_true', help='Use TTA during evaluation')
    args = parser.parse_args()

    MODEL_DIR = os.path.join(utils.ROOT_DIR, 'models', args.name)
    SUB_DIR = os.path.join(utils.ROOT_DIR, 'subs')
    SUB_PROB_DIR = os.path.join(SUB_DIR, 'probs')

    if args.tta:
        sub_end = '-TTA.csv'
    else:
        sub_end = '.csv'

    TEST_CONFIG = {
        'batch_size': args.batch_size,
        'augmentation': False,
        'clf_name': args.name.split('-')[0]}
    test_seq = TestSequence(TEST_CONFIG)

    model_files = sorted(glob(os.path.join(MODEL_DIR, '*.h5')))
    if not args.all:
        if args.folds:
            model_files = [file for file in model_files if os.path.basename(file).startswith('fold')]
            sub_end = '-folded' + sub_end
        else:
            model_files = [file for file in model_files if os.path.basename(file).startswith('model')]
        if args.best:
            model_files = [file for file in model_files if os.path.basename(file).endswith('best.h5')]
            sub_end = '-best' + sub_end
        else:
            model_files = [file for file in model_files if not os.path.basename(file).endswith('best.h5')]
    else:
        sub_end = '-ALL' + sub_end

    SUB_PATH = os.path.join(SUB_DIR, args.name + sub_end)
    SUB_PROB_PATH = os.path.join(SUB_PROB_DIR, args.name + sub_end)

    print('Models in ensemble:')
    for file in model_files:
        print(file)
    global_probs = []
    for file in model_files:
        model = load_model(file)
        test_seq.augmentation = False
        probs = model.predict_generator(
            generator=test_seq,
            steps=len(test_seq),
            verbose=1)
        global_probs.append(probs)
        if args.tta:
            for aug_flag in range(1, 4):  # (1, 5)
                test_seq.augment = aug_flag
                probs = model.predict_generator(
                    generator=test_seq,
                    steps=len(test_seq),
                    verbose=1)
                global_probs.append(probs)
    probs = sum(global_probs) / len(global_probs)
    ids = np.argmax(probs, axis=1)
    probs_max = np.max(probs, axis=1)
    labels = [utils.ID2LABEL[id_] for id_ in ids]
    data = {
        'fname': test_seq.files,
        'camera': labels,
        'prob': probs_max}
    sub = pd.DataFrame(data)
    sub.to_csv(SUB_PROB_PATH, index=False)
    print('Submission with probs created successfully')
    sub.drop(['prob'], axis=1, inplace=True)
    sub.to_csv(SUB_PATH, index=False)
    print('Submission file created successfully')
