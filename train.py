import os
import time

import numpy as np
import tensorflow as tf

from brfss.data import load_train_data, build_batch, columns

data_dir = os.path.join(os.path.dirname(__file__), 'brfss', 'data')


class MultitaskDNN:
    def __init__(self, labels_ndims):
        self.labels_ndims = labels_ndims

    def model(self, x, dropout_rate):
        with tf.variable_scope('hidden1'):
            net = tf.layers.dense(x, 64, tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer(), name='dense')
            net = tf.layers.dropout(net, dropout_rate, name='dropout')

        with tf.variable_scope('hidden2'):
            net = tf.layers.dense(net, 32, tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer(), name='dense')
            net = tf.layers.dropout(net, dropout_rate, name='dropout')

        output_layers = []
        for i in range(self.labels_ndims):
            with tf.variable_scope('output%d' % i):
                output_layers.append(tf.layers.dense(net, 1, activation=None))

        return output_layers

    @staticmethod
    def loss(logits, labels):
        loss = 0.

        for (labels_val, logits_val) in zip(labels, logits):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_float(labels_val), logits=logits_val)
            loss += tf.reduce_mean(losses)

        return loss


def train(train_data_path, eval_data_path, log_dir, checkpoint_file, batch_size, learning_rate,
          decay_steps, decay_rate, dropout_rate, max_steps):
    tf.logging.set_verbosity(tf.logging.INFO)

    train_data = load_train_data(os.path.join(train_data_path))
    eval_data = load_train_data(os.path.join(eval_data_path))

    train_data_size = train_data.size
    print('Train Data Size: %d' % train_data_size)
    print('Eval Data Size: %d' % eval_data.size)

    graph = tf.Graph()

    with graph.as_default():
        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(learning_rate, global_step=global_step,
                                                   decay_steps=decay_steps,
                                                   decay_rate=decay_rate)

        model = MultitaskDNN(labels_ndims=2)

        def fill(val):
            return tf.fill([batch_size, ], val)

        features = {
            'WEIGHT2': fill(9999),
            'HEIGHT3': fill(9999),
            'SEX': fill(9),
            'EMPLOY1': fill(-1),
            'INCOME2': fill(-1),
            'MARITAL': fill(-1),
            'EDUCA': fill(-1),
            'CHILDREN': fill(-1),
            '_AGEG5YR': fill(-1)
        }
        input_layer = tf.feature_column.input_layer(features, columns)

        y_usenow3 = tf.placeholder(tf.int32, shape=[batch_size, 1], name='usenow3')
        y_ecignow = tf.placeholder(tf.int32, shape=[batch_size, 1], name='ecignow')

        label_logits = model.model(input_layer, dropout_rate)
        loss = MultitaskDNN.loss(label_logits, [y_usenow3, y_ecignow])

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('learning_rate', learning_rate)
        summary = tf.summary.merge_all()

        summary_saver_hook = tf.train.SummarySaverHook(save_steps=1000, output_dir=log_dir, summary_op=summary)
        profiler_hook = tf.train.ProfilerHook(save_steps=1000, output_dir=log_dir)
        stop_at_step_hook = tf.train.StopAtStepHook(num_steps=max_steps)
        logging_hook = tf.train.LoggingTensorHook({'loss': loss, 'learning_rate': learning_rate}, every_n_iter=1000)

    with graph.as_default():
        hooks = [summary_saver_hook, profiler_hook, stop_at_step_hook, logging_hook]

        with tf.train.SingularMonitoredSession(hooks=hooks) as session:
            print('Training')

            duration = 0.

            while not session.should_stop():
                start_time = time.time()

                x, y1, y2 = build_batch(train_data, batch_size)

                _, loss_val, summary_val, global_step_val, learning_rate_val = session.run([train_op, loss,
                                                                                            summary, global_step,
                                                                                            learning_rate], feed_dict={
                    features['WEIGHT2']: x['WEIGHT2'],
                    features['HEIGHT3']: x['HEIGHT3'],
                    features['SEX']: x['SEX'],
                    features['EMPLOY1']: x['EMPLOY1'],
                    features['INCOME2']: x['INCOME2'],
                    features['MARITAL']: x['MARITAL'],
                    features['EDUCA']: x['EDUCA'],
                    features['CHILDREN']: x['CHILDREN'],
                    features['_AGEG5YR']: x['_AGEG5YR'],
                    y_usenow3: np.expand_dims(y1.values, axis=1),
                    y_ecignow: np.expand_dims(y2.values, axis=1)
                })
                duration += time.time() - start_time

            print('Exiting')


if __name__ == '__main__':
    train(
        train_data_path=os.path.join(data_dir, 'LLCP2016_train.csv'),
        eval_data_path=os.path.join(data_dir, 'LLCP2016_train.csv'),
        log_dir=os.path.join('.', 'logs'),
        checkpoint_file=None,  # os.path.join('.', 'logs', 'model.ckpt'),
        batch_size=512,
        learning_rate=0.2,
        decay_steps=10000,
        decay_rate=0.90,
        dropout_rate=0.01,
        max_steps=35000
    )
