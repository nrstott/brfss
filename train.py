import os
import time

import numpy as np
import tensorflow as tf

from brfss.data import load_train_data, build_batch, columns

data_dir = os.path.join(os.path.dirname(__file__), 'brfss', 'data')


class MultitaskDNN:
    def __init__(self, input_layer, labels_ndims, dropout_rate=None):
        """Multitask DNN using the same hidden layers to train multiple logits layers.

        Args:
            input_layer (:obj:`tf.feature_columns.input_layer`) Input layer for the model.
            labels_ndims (int): Number of labels that this model will be trained on. One label per task.
            dropout_rate (float): Dropout rate of the dropout layers. If None, no dropout layers will be created.
        """
        self.labels_ndims = labels_ndims
        self.dropout_rate = dropout_rate

        with tf.variable_scope('hidden1'):
            net = tf.layers.dense(input_layer, 64, tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer(), name='dense')
            if dropout_rate is not None:
                net = tf.layers.dropout(net, dropout_rate, name='dropout')

        with tf.variable_scope('hidden2'):
            net = tf.layers.dense(net, 32, tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer(), name='dense')
            if dropout_rate is not None:
                net = tf.layers.dropout(net, dropout_rate, name='dropout')

        self.logits_layers = []
        for i in range(self.labels_ndims):
            with tf.variable_scope('output%d' % i):
                self.logits_layers.append(tf.layers.dense(net, 1, activation=None))

    def loss(self, labels):
        """
        Calculates loss for multitask DNN.

        Args:
              labels (list(int)): Labels. The number of layers must equal `self.labels_ndims`.

        Returns:
            Calculated loss. A `float`.

        """
        loss = 0.

        for (labels_val, logits_val) in zip(labels, self.logits_layers):
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
        model = MultitaskDNN(input_layer=input_layer, labels_ndims=2, dropout_rate=dropout_rate)

        y_usenow3 = tf.placeholder(tf.int32, shape=[batch_size, 1], name='usenow3')
        y_ecignow = tf.placeholder(tf.int32, shape=[batch_size, 1], name='ecignow')

        loss = model.loss([y_usenow3, y_ecignow])

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('learning_rate', learning_rate)
        summary = tf.summary.merge_all()

        checkpoint_saver_hook = tf.train.CheckpointSaverHook(checkpoint_dir=log_dir, save_steps=1000)
        summary_saver_hook = tf.train.SummarySaverHook(save_steps=1000, output_dir=log_dir, summary_op=summary)
        profiler_hook = tf.train.ProfilerHook(save_steps=1000, output_dir=log_dir)
        stop_at_step_hook = tf.train.StopAtStepHook(num_steps=max_steps)
        logging_hook = tf.train.LoggingTensorHook({'loss': loss, 'learning_rate': learning_rate}, every_n_iter=1000)

        hooks = [checkpoint_saver_hook, summary_saver_hook, profiler_hook, stop_at_step_hook, logging_hook]

    with graph.as_default():
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
