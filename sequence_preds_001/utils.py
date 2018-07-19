
import os
import sys
import json
import numpy as np

from util.log import get_logger

log = get_logger()


class ShowProcess:
    i = 0
    max_steps = 0
    max_arrow = 50

    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.i = 0

    def show_process(self, i=None):
        if i is not None:
            self.i = i
        else:
            self.i += 1
        num_arrow = int(self.i * self.max_arrow / self.max_steps)
        num_line = self.max_arrow - num_arrow
        percent = self.i * 100.0 / self.max_steps
        process_bar = '[' + '>' * num_arrow + '-' * num_line + ']' + '%.2f' % percent + '%' + '\r'
        sys.stdout.write(process_bar)
        sys.stdout.flush()

    def close(self, words=''):
        # print(words)
        self.i = 0


def save_log_auto(folder_to_save, config_name, **params):
    save_json(folder_to_save=folder_to_save, config_name=config_name, results=params)


def save_log(folder_to_save, config_name,
             num_attention, num_characters, num_char, batch_size, rnn_size, num_layers,
             learning_rate, keep_prob, decay_rate, decay_steps):
    """
    save parameters into json log file
    """
    results = {
        'rnn param': {
            'number of attention': num_attention,
            'number of characters': num_characters,
            'number of characters per train': num_char,
            'batch size': batch_size,
            'rnn size': rnn_size,
            'num layers': num_layers
        },
        'training param': {
            'learning rate': learning_rate,
            'keep probability': keep_prob,
            'decay rate': decay_rate,
            'decay steps': decay_steps
        }
    }
    save_json(folder_to_save=folder_to_save, config_name=config_name, results=results)


def save_loss_json(folder_to_save, epochs, sequence_loss, nn_loss, nn_acc, tag='train'):
    config_name = tag + '_loss'

    results = {"epochs": epochs,
               "sequence loss": str(sequence_loss),
               "nn loss": str(nn_loss),
               "nn accuracy": str(nn_acc)}

    save_json(folder_to_save=folder_to_save, config_name=config_name, results=results)


def save_sequence_loss_json(folder_to_save, epochs, sequence_loss, tag='train'):
    config_name = tag + '_loss'

    results = {"epochs": epochs,
               "sequence loss": str(sequence_loss)}

    save_json(folder_to_save=folder_to_save, config_name=config_name, results=results)


def save_nn_loss_json(folder_to_save, epochs, nn_loss, nn_acc, tag='train'):
    config_name = tag + '_loss'

    results = {"epochs": epochs,
               "nn loss": str(nn_loss),
               "nn accuracy": str(nn_acc)}

    save_json(folder_to_save=folder_to_save, config_name=config_name, results=results)


def save_text_json(folder_to_save, epochs, batch, text, tag='train'):
    config_name = tag + '_text'

    results = {"epochs": epochs,
               "batch": batch,
               "generated text": text}

    save_json(folder_to_save=folder_to_save, config_name=config_name, results=results)


def save_json(folder_to_save, config_name, results):
    """
    tool part for appending info to json file
    """
    re = []

    # log.info('folder_to_save: {}', folder_to_save)
    if not os.path.exists('{}.json'.format(folder_to_save + '/' + str(config_name))):
        re.append(results)
        # os.makedirs(folder_to_save)
    else:
        with open('{}.json'.format(folder_to_save + '/' + str(config_name))) as data_input:
            re = json.load(data_input)
        re.append(results)

    with open('{}.json'.format(folder_to_save + '/' + str(config_name)), 'wt') as output:
        output.write(json.dumps(re, indent=4))


def save_checkpoint(folder_to_save, saver, sess, epochs):
    """
    save model per epochs
    """
    saver.save(sess, folder_to_save + '/' + 'e{}'.format(epochs))
    # return folder_to_save + '/' + 'e{}'.format(epochs)


def save_rnn_checkpoint(folder_to_save, saver, sess, epochs):
    """
    save model per epochs
    """
    saver.save(sess, folder_to_save + '/' + 'rnn_e{}'.format(epochs))


def save_nn_checkpoint(folder_to_save, saver, sess, epochs):
    """
    save model per epochs
    """
    saver.save(sess, folder_to_save + '/' + 'nn_e{}'.format(epochs))


def save_checkpoint_by_batch(folder_to_save, saver, sess, epochs, train_step):
    """
    save model per epochs
    """
    saver.save(sess, folder_to_save + '/' + 'e{}_train_step{}'.format(epochs, train_step))


def save_rnn_checkpoint_by_batch(folder_to_save, saver, sess, epochs, train_step):
    """
    save model per epochs
    """
    saver.save(sess, folder_to_save + '/' + 'rnn_e{}_train_step{}'.format(epochs, train_step))


def save_nn_checkpoint_by_batch(folder_to_save, saver, sess, epochs, train_step):
    """
    save model per epochs
    """
    saver.save(sess, folder_to_save + '/' + 'nn_e{}_train_step{}'.format(epochs, train_step))


def save_to_np_array(folder_tp_save, appendix=None, **array_to_save):
    if appendix is None:
        for name, value in array_to_save.items():
            np.save(os.path.join(folder_tp_save, name + '.npy'), value)
    else:
        for name, value in array_to_save.items():
            np.save(os.path.join(folder_tp_save, name + '_{}.npy'.format(appendix)), value)





