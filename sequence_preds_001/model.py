import os
import sys
import time
import string
import tensorflow as tf
from tensorflow.contrib.seq2seq import *
import numpy as np

from .utils import *

__description__ = "sequence rating prediction"


class Model:
    def __init__(self, embeddings, rating_scale, rnn_size=128, num_layers=2, keep_prob=0.8,
                 grad_clip=5, learning_rate=1e-3, decay_rate=0.9, decay_steps=100):
        self.embeddings = np.append(embeddings, [np.zeros(embeddings.shape[1])], axis=0)
        self.rating_scale = rating_scale
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.keep_prob = keep_prob
        self.grad_clip = grad_clip
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

        self._build_model()

    def _build_model(self):
        tf.reset_default_graph()
        self.graph = tf.Graph()
        with self.graph.as_default():

            """
            item embedding lookup
            """
            with tf.name_scope("char_lookup"):
                item_embeddings = tf.constant(self.embeddings, dtype=tf.float32)
                table = tf.convert_to_tensor(item_embeddings, name='item_embeddings_table')

            with tf.name_scope("input_data"):
                self.sequence_ids = tf.placeholder(tf.int32, [None, None], name='sequence_ids')
                self.sequence_targets = tf.placeholder(tf.int32, [None, None], name='sequence_targets')
                self.actual_length = tf.placeholder(tf.int32, [None], name='sequence_actual_length')

                sequence_inputs = tf.nn.embedding_lookup(table, self.sequence_ids, name='sequence_inputs')
                y = tf.slice(self.sequence_targets,
                             [0, 0], [tf.shape(self.sequence_ids)[0], tf.reduce_max(self.actual_length)])

            with tf.variable_scope("rnn_decoder"):
                self.cell = tf.contrib.rnn.MultiRNNCell(
                    [self.gru_cell(gru_size=self.rnn_size, keep_prob=self.keep_prob) for _ in range(self.num_layers)])

                # shape: [batch size, sequence length, rnn size], [batch size, rnn size]
                self.decoder_output, self.decoder_sate = tf.nn.dynamic_rnn(self.cell, sequence_inputs,
                                                                           sequence_length=self.actual_length,
                                                                           dtype=tf.float32)

                # shape: [batch size, max actual length, rating scale]
                self.decoder_output = tf.slice(
                    self.decoder_output, [0, 0, 0],
                    [tf.shape(self.sequence_ids)[0], tf.reduce_max(self.actual_length), self.rnn_size])
                scaled_decoder_output = tf.layers.batch_normalization(self.decoder_output, axis=2)

            with tf.variable_scope("output_layer"):
                # if dense activation function is None, then linear activation, that x * w +b
                # shape: [batch size * num char, vector length]
                softmax_w = tf.get_variable('weight', shape=[self.rnn_size, self.rating_scale],
                                            initializer=tf.contrib.layers.xavier_initializer())
                softmax_b = tf.get_variable('bias', initializer=tf.zeros(self.rating_scale))

                # shape: [batch size, max actual length, rating scale]
                output_logits = tf.map_fn(lambda rev: tf.matmul(rev, softmax_w) + softmax_b, scaled_decoder_output)

            with tf.variable_scope("sequence_predict"):
                # shape: [batch size, batch max sequence length, rating scale]
                self.predict_y_prob = tf.nn.softmax(logits=output_logits, name='predict_probability')

                # shape: [batch size, batch max sequence length]
                self.prediction = tf.argmax(input=self.predict_y_prob, axis=2, name='prediction')

                # shape: [batch size, max actual length]
                correct_prediction = tf.equal(tf.cast(self.prediction, tf.int32), y)
                self.predict_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            with tf.variable_scope("sequence_optimization", reuse=False):
                loss = sequence_loss(
                    logits=output_logits, targets=y, weights=tf.sequence_mask(self.actual_length, dtype=tf.float32))

                self.cost = tf.reduce_mean(loss)

                # adjust learning rate, make more smooth
                global_step = tf.Variable(0, trainable=False)
                self.sequence_learning_rate = tf.train.exponential_decay(learning_rate=self.learning_rate,
                                                                         global_step=global_step,
                                                                         decay_steps=self.decay_steps,
                                                                         decay_rate=self.decay_rate, staircase=True)

                # variables to optimize
                train_variables = tf.trainable_variables()

                # Optimizer for training, using gradient clipping to control exploding gradients
                grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, train_variables), self.grad_clip)

                optimizer = tf.train.AdamOptimizer(learning_rate=self.sequence_learning_rate)
                # optimizer = tf.train.RMSPropOptimizer(learning_rate=self.sequence_learning_rate, decay=0.95)

                # apply gradient descent
                self.gradients = optimizer.apply_gradients(grads_and_vars=zip(grads, train_variables),
                                                           global_step=global_step)

            # initial part
            with tf.name_scope("initialization"):
                self.global_init = tf.global_variables_initializer()
                self.local_init = tf.local_variables_initializer()
                self.table_init = tf.tables_initializer()

            with tf.name_scope("summary"):
                tf.summary.scalar('sequence_cost', self.cost)
                tf.summary.scalar('sequence_accuracy', self.predict_accuracy)
                tf.summary.scalar('learning_rate', self.sequence_learning_rate)
                self.merged_summary = tf.summary.merge_all()

    def lstm_cell(self, lstm_size, keep_prob):
        lstm = tf.contrib.rnn.LSTMCell(lstm_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        return drop

    def gru_cell(self, gru_size, keep_prob):
        gru = tf.contrib.rnn.GRUCell(gru_size)
        drop = tf.contrib.rnn.DropoutWrapper(gru, output_keep_prob=keep_prob)
        return drop















