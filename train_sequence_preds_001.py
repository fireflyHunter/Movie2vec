import os
import time
import argparse
import pandas as pd
import numpy as np

from sequence_preds_001.model import *
from sequence_preds_001.train import *
from sequence_preds_001.data_pipline import *
from sequence_preds_001.utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def path_check_crete(path):
    """
    check path exist, otherwise, create path
    :param path:
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    folder_name = './sequence_preds_001'

    parser = argparse.ArgumentParser(description='rs4fun')

    # data file and save directory
    parser.add_argument('-f', '--file_path', default='./data/')

    parser.add_argument('-s', '--save_path', default=folder_name + '/checkpoint_loss')
    parser.add_argument('-train', '--train_tensorboard_path', default=folder_name + '/train_tensorboard')
    parser.add_argument('-val', '--val_tensorboard_path', default=folder_name + '/val_tensorboard')

    # model param
    parser.add_argument('--rnn_size', default=256, type=int)
    parser.add_argument('--num_layers', default=2, type=int)

    # training param
    # parser.add_argument('--vector_length', default=100, type=int)
    parser.add_argument('--rating_scale', default=12, type=int)
    parser.add_argument('--batch_size', default=32, type=int)

    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--decay_rate', default=0.97, type=float)
    parser.add_argument('--grad_clip', default=5, type=int)
    parser.add_argument('--keep_prob', default=0.8, type=float)
    parser.add_argument('--epochs_start', default=1, type=int)
    parser.add_argument('--epochs_end', default=50, type=int)

    args = parser.parse_args()

    # asign params
    # file_path, csv_start, csv_end = args.file_path, args.csv_start, args.csv_end
    file_path = args.file_path
    save_path = args.save_path
    train_tensorboard_path = args.train_tensorboard_path
    val_tensorboard_path = args.val_tensorboard_path
    rnn_size, num_layers = args.rnn_size, args.num_layers
    rating_scale, batch_size = args.rating_scale, args.batch_size
    learning_rate, decay_rate, grad_clip, keep_prob = args.learning_rate, args.decay_rate, args.grad_clip, args.keep_prob
    epochs_start, epochs_end = args.epochs_start, args.epochs_end

    path_check_crete(save_path)
    path_check_crete(train_tensorboard_path)
    path_check_crete(val_tensorboard_path)

    # read data
    print('read data')

    embeddings = np.load(os.path.join(file_path, "vectors/ml-10m-vectors"))
    embedding_size = embeddings.shape[1]

    train_index, train_actual_size, train_ratings, val_index, val_actual_size, val_ratings = preprocess_load_training(
        file_path=file_path, embedding_size=embedding_size)

    model = Model(embeddings=embeddings, rating_scale=rating_scale, rnn_size=rnn_size, num_layers=num_layers,
                  keep_prob=keep_prob, grad_clip=grad_clip, learning_rate=learning_rate, decay_rate=decay_rate,
                  decay_steps=len(train_index))

    if not os.path.exists(os.path.join(save_path, 'rs4fun.json')):
        save_log_auto(folder_to_save=save_path, config_name='rs4fun', rating_scale=rating_scale, rnn_size=rnn_size,
                      num_layers=num_layers, keep_prob=keep_prob, grad_clip=grad_clip,
                      learning_rate=learning_rate, decay_rate=decay_rate,
                      decay_steps=len(train_index))

    train_model(model=model, epochs_start=epochs_start, epochs_end=epochs_end, save_path=save_path,
                train_tensorboard_path=train_tensorboard_path, val_tensorboard_path=val_tensorboard_path,
                batch_size=batch_size,
                train_embeddings=train_index, train_actual_length=train_actual_size, train_ratings=train_ratings,
                val_embeddings=val_index, val_actual_length=val_actual_size, val_ratings=val_ratings)










