
from .model import *
from .utils import *
from collections import OrderedDict


def train_setting(save_path):
    # define global step
    if os.path.exists(os.path.join(save_path, 'global_step.npy')):
        global_steps = np.load(os.path.join(save_path, 'global_step.npy'))
        train_global_steps = global_steps[0]
        val_global_steps = global_steps[1]
    else:
        train_global_steps = 0
        val_global_steps = 0

    # training configuration
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    return config, train_global_steps, val_global_steps


def call_model(sess, model, feed_dict, *call_opt_names):
    opt_to_all = [(i, getattr(model, i)) for i in call_opt_names]
    opt_to_all = OrderedDict(opt_to_all)

    # return a dictionary with same key order as "call_opt_names"
    return sess.run(opt_to_all, feed_dict=feed_dict)


def batch_train(sess, model, train_summary_writer, saver,
                num_train_batch, batch_size, train_global_steps, save_path, e, sp,
                train_embeddings, train_actual_length, train_ratings):

    shuffle_index = np.random.permutation(len(train_embeddings))
    train_input = train_embeddings[shuffle_index]
    train_actual = train_actual_length[shuffle_index]
    train_target = train_ratings[shuffle_index]

    # print(train_input.shape, train_actual.shape, train_target.shape)

    loss_list = []
    for batch_index in range(num_train_batch):

        batch_embeddings = train_input[batch_index * batch_size: (batch_index + 1) * batch_size]
        batch_actual = train_actual[batch_index * batch_size: (batch_index + 1) * batch_size]
        batch_ratings = train_target[batch_index * batch_size: (batch_index + 1) * batch_size]

        # print(batch_embeddings, batch_actual, batch_ratings)

        feed_dict = {model.sequence_ids: batch_embeddings,
                     model.sequence_targets: batch_ratings,
                     model.actual_length: batch_actual}

        batch_results = call_model(sess, model, feed_dict, 'gradients', 'cost', 'merged_summary')
        print(batch_results['cost'])
        loss_list.append(batch_results['cost'])
        train_summary_writer.add_summary(summary=batch_results['merged_summary'], global_step=train_global_steps)

        train_global_steps += 1

        if train_global_steps % 200 == 0:
            save_checkpoint_by_batch(folder_to_save=save_path, saver=saver, sess=sess,
                                         epochs=e, train_step=train_global_steps)

        sp.show_process()

        # break
    sp.close()

    return np.mean(loss_list), train_global_steps


def batch_val(sess, model, val_summary_writer, saver,
              num_val_batch, batch_size, val_global_steps, save_path, e, sp,
              val_embeddings, val_actual_length, val_ratings):
    shuffle_index = np.random.permutation(len(val_embeddings))
    val_input = val_embeddings[shuffle_index]
    val_actual = val_actual_length[shuffle_index]
    val_target = val_ratings[shuffle_index]

    loss_list = []
    for batch_index in range(num_val_batch):

        batch_embeddings = val_input[batch_index * batch_size: (batch_index + 1) * batch_size]
        batch_actual = val_actual[batch_index * batch_size: (batch_index + 1) * batch_size]
        batch_ratings = val_target[batch_index * batch_size: (batch_index + 1) * batch_size]

        feed_dict = {model.sequence_ids: batch_embeddings,
                     model.sequence_targets: batch_ratings,
                     model.actual_length: batch_actual}

        batch_results = call_model(sess, model, feed_dict, 'cost', 'merged_summary')
        loss_list.append(batch_results['cost'])
        val_summary_writer.add_summary(summary=batch_results['merged_summary'], global_step=val_global_steps)

        val_global_steps += 1

        if val_global_steps % 200 == 0:
            save_checkpoint_by_batch(folder_to_save=save_path, saver=saver, sess=sess,
                                     epochs=e, train_step=val_global_steps)

        sp.show_process()
        # break
    sp.close()

    return np.mean(loss_list), val_global_steps


def epoch_train(sess, model, train_global_steps, val_global_steps,
                epochs_start, epochs_end, batch_size, save_path, train_tensorboard_path, val_tensorboard_path,
                train_embeddings, train_actual_length, train_ratings, val_embeddings, val_actual_length, val_ratings):
    num_train_batch = len(train_embeddings) // batch_size
    num_val_batch = len(val_embeddings) // batch_size

    # model saver
    saver = tf.train.Saver(max_to_keep=None)

    # saving operation for save and restore the model
    sess.run([model.global_init, model.local_init, model.table_init])

    if epochs_start != 1:
        print('restore')
        saver.restore(sess, save_path + '/' + 'e{}'.format(epochs_start - 1))

    # tensorboard writer operation, for visualising on tensorboard
    train_summary_writer = tf.summary.FileWriter(train_tensorboard_path, sess.graph)
    # tensorboard writer operation, for visualising on tensorboard
    val_summary_writer = tf.summary.FileWriter(val_tensorboard_path, sess.graph)

    for e in range(epochs_start, epochs_end):
        print('')
        log.info('in epochs {}'.format(e))

        """
        TRAINING
        """
        # process bar
        sp = ShowProcess(num_train_batch)
        start_epochs = time.time()

        epochs_sequence_loss, train_global_steps = batch_train(
            sess=sess, model=model, train_summary_writer=train_summary_writer, saver=saver,
            num_train_batch=num_train_batch, batch_size=batch_size,
            train_global_steps=train_global_steps, save_path=save_path, e=e, sp=sp,
            train_embeddings=train_embeddings, train_actual_length=train_actual_length, train_ratings=train_ratings)
        print("epochs {}, TRAIN sequence loss {}, time cost {}".format(
            e, epochs_sequence_loss, time.time() - start_epochs))
        save_sequence_loss_json(folder_to_save=save_path, epochs=e, sequence_loss=epochs_sequence_loss, tag='train')

        """
        VALIDATE
        """
        # process bar
        sp = ShowProcess(num_val_batch)
        start_epochs = time.time()
        val_global_steps = train_global_steps

        epochs_sequence_loss, val_global_steps = batch_val(
            sess=sess, model=model, val_summary_writer=val_summary_writer, saver=saver,
            num_val_batch=num_val_batch, batch_size=batch_size,
            val_global_steps=val_global_steps, save_path=save_path, e=e, sp=sp,
            val_embeddings=val_embeddings, val_actual_length=val_actual_length, val_ratings=val_ratings)

        print("epochs {}, VAL sequence loss {}, time cost {}".format(
            e, epochs_sequence_loss, time.time() - start_epochs))
        save_sequence_loss_json(folder_to_save=save_path, epochs=e, sequence_loss=epochs_sequence_loss, tag='val')

        save_checkpoint(folder_to_save=save_path, saver=saver, sess=sess, epochs=e)
        global_steps = np.array([train_global_steps, val_global_steps])
        np.save(os.path.join(save_path, 'global_step.npy'), global_steps)


def train_model(model, epochs_start, epochs_end, save_path,
                train_tensorboard_path, val_tensorboard_path, batch_size,
                train_embeddings, train_actual_length, train_ratings,
                val_embeddings, val_actual_length, val_ratings):
    print('--------------learning--------------')

    config, train_global_steps, val_global_steps = train_setting(save_path)

    with tf.Session(graph=model.graph, config=config) as sess:
        epoch_train(sess, model, train_global_steps, val_global_steps,
                    epochs_start, epochs_end, batch_size, save_path, train_tensorboard_path, val_tensorboard_path,
                    train_embeddings, train_actual_length, train_ratings, val_embeddings, val_actual_length,
                    val_ratings)





