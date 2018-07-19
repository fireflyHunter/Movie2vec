from .utils import *

import os
import numpy as np
from sklearn.model_selection import train_test_split

"""
random seed to 123, to make sure generate same random split for every training
"""
np.random.seed(123)

the_dict = {0.0: 0, 0.5: 1, 1.0: 2, 1.5: 3, 2.0: 4, 2.5: 5, 3.0: 6, 3.5: 7, 4.0: 8, 4.5: 9, 5.0: 10}


def process_example_rating(max_sequence_length, items):
    temp = np.full(max_sequence_length, 11)
    for index, i in enumerate(items):
        temp[index] = the_dict.get(i)
    return temp


def process_example_length(max_sequence_length, max_item, items):
    temp = np.full(max_sequence_length, max_item)
    for index, i in enumerate(items):
        temp[index] = i
    return temp


def preprocess_load_training(file_path, embedding_size):
    if not os.path.exists(os.path.join(file_path, 'train_index_{}.npy'.format(embedding_size))):
        items_index = np.load(os.path.join(file_path, "10m-items.npy"))
        ratings = np.load(os.path.join(file_path, "10m-ratings.npy"))

        actual_length = [len(i) for i in items_index]
        max_item = np.max(items_index.max())
        max_sequence_length = np.max(actual_length)
        print(max_sequence_length)

        padded_index = [process_example_length(max_sequence_length, max_item, i) for i in items_index]
        padded_ratings = [process_example_rating(max_sequence_length, i) for i in ratings]

        print(len(padded_index), len(actual_length), len(padded_ratings))

        shuffle = np.random.permutation(len(padded_index))
        train_size, validate_size = int(0.8 * len(shuffle)), int(0.9 * len(shuffle))

        temp = np.array(padded_index)[shuffle]
        train_index, val_index, test_index = temp[:train_size], temp[train_size: validate_size], temp[validate_size:]

        temp = np.array(padded_ratings)[shuffle]
        train_ratings, val_ratings, test_ratings = temp[:train_size], temp[train_size: validate_size], temp[validate_size:]

        temp = np.array(actual_length)[shuffle]
        train_actual_size, val_actual_size, test_actual_size = temp[:train_size], temp[train_size: validate_size], temp[validate_size:]

        save_to_np_array(file_path, appendix=embedding_size,
                         train_index=train_index, train_actual_size=train_actual_size, train_ratings=train_ratings,
                         val_index=val_index, val_actual_size=val_actual_size, val_ratings=val_ratings,
                         test_index=test_index, test_actual_size=test_actual_size, test_ratings=test_ratings)
        return train_index, train_actual_size, train_ratings, val_index, val_actual_size, val_ratings
    else:
        train_index = np.load(os.path.join(file_path, 'train_index_{}.npy'.format(embedding_size)))
        train_actual_size = np.load(os.path.join(file_path, 'train_actual_size_{}.npy'.format(embedding_size)))
        train_ratings = np.load(os.path.join(file_path, 'train_ratings_{}.npy'.format(embedding_size)))
        val_index = np.load(os.path.join(file_path, 'val_index_{}.npy'.format(embedding_size)))
        val_actual_size = np.load(os.path.join(file_path, 'val_actual_size_{}.npy'.format(embedding_size)))
        val_ratings = np.load(os.path.join(file_path, 'val_ratings_{}.npy'.format(embedding_size)))

        return train_index, train_actual_size, train_ratings, val_index, val_actual_size, val_ratings
