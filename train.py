import os
import time

import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE

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
            net = tf.layers.dense(input_layer, 64, tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer(),
                                  name='dense')
            if dropout_rate is not None:
                net = tf.layers.dropout(net, dropout_rate, name='dropout')

        with tf.variable_scope('hidden2'):
            net = tf.layers.dense(net, 32, tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer(), name='dense')
            if dropout_rate is not None:
                net = tf.layers.dropout(net, dropout_rate, name='dropout')

        self.logits_layers = []
        for i in range(self.labels_ndims):
            with tf.variable_scope('output%d' % i):
                dense = tf.layers.dense(net, 1, activation=None)
                self.logits_layers.append(dense)

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

    sm = SMOTE(kind='regular', k_neighbors=5)

    train_data_size = train_data.size
    print('Train Data Size: %d' % train_data_size)
    print('Eval Data Size: %d' % eval_data.size)
    print('Train Data USENOW3 1 %d' % train_data.loc[train_data.USENOW3 == 1].size)
    print('Train Data USENOW3 0 %d' % train_data.loc[train_data.USENOW3 == 0].size)

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

        predictions_usenow3 = tf.round(tf.sigmoid(model.logits_layers[0]))
        predictions_usenow3_str = tf.reduce_join(tf.as_string(tf.to_int32(predictions_usenow3)))

        predictions_ecignow = tf.round(tf.sigmoid(model.logits_layers[1]))
        predictions_ecignow_str = tf.reduce_join(tf.as_string(predictions_ecignow))

        labels_usenow3 = tf.reduce_join(tf.as_string(y_usenow3), axis=0)

        (usenow3_accuracy, usenow3_accuracy_update_op) = tf.metrics.accuracy(
            labels=labels_usenow3,
            predictions=predictions_usenow3_str)
        (ecignow_accuracy, ecignow_accuracy_update_op) = tf.metrics.accuracy(
            labels=tf.reduce_join(tf.as_string(y_ecignow), axis=0),
            predictions=predictions_ecignow_str)

        examples = (float(batch_size) * tf.to_float(global_step))

        def _add_summaries(labels, predictions, family):
            (true_positives, true_positives_update_op) = tf.metrics.true_positives(labels=labels,
                                                                                   predictions=predictions)

            (false_positives, false_positives_update_op) = tf.metrics.false_positives(labels=labels,
                                                                                      predictions=predictions)

            (true_negatives, true_negatives_update_op) = tf.metrics.true_negatives(labels=labels,
                                                                                   predictions=predictions)

            (false_negatives, false_negatives_update_op) = tf.metrics.false_negatives(labels=labels,
                                                                                      predictions=predictions)

            tf.summary.scalar('true_positives', tf.to_float(true_positives) / examples, family=family)
            tf.summary.scalar('true_negatives', tf.to_float(true_negatives) / examples, family=family)
            tf.summary.scalar('false_positives', tf.to_float(false_positives) / examples, family=family)
            tf.summary.scalar('false_negatives', tf.to_float(false_negatives) / examples, family=family)

            return [true_positives_update_op, false_positives_update_op,
                    true_negatives_update_op, false_negatives_update_op]

        summary_ops = []

        summary_ops += _add_summaries(y_usenow3, predictions_usenow3, family='usenow3')
        summary_ops += _add_summaries(y_ecignow, predictions_ecignow, family='ecignow')

        (usenow3_precisions, usenow3_precisions_update_op) = \
            tf.metrics.precision_at_thresholds(labels=y_usenow3,
                                               predictions=predictions_usenow3,
                                               thresholds=[0.1, 0.5, 0.75])

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('usenow3_accuracy', usenow3_accuracy, family='ecignow')
        tf.summary.scalar('ecignow_accuracy', ecignow_accuracy, family='ecignow')
        summary = tf.summary.merge_all()

        checkpoint_saver_hook = tf.train.CheckpointSaverHook(checkpoint_dir=log_dir, save_steps=1000)
        summary_saver_hook = tf.train.SummarySaverHook(save_steps=1000, output_dir=log_dir, summary_op=summary)
        profiler_hook = tf.train.ProfilerHook(save_steps=1000, output_dir=log_dir)
        stop_at_step_hook = tf.train.StopAtStepHook(num_steps=max_steps)
        logging_hook = tf.train.LoggingTensorHook({
            'loss': loss,
            'learning_rate': learning_rate,
            'usenow3_accuracy': usenow3_accuracy,
            'ecignow_accuracy': ecignow_accuracy}, every_n_iter=1000)

        hooks = [checkpoint_saver_hook, summary_saver_hook, profiler_hook, stop_at_step_hook, logging_hook]

    with graph.as_default():
        with tf.train.SingularMonitoredSession(hooks=hooks) as session:
            print('Training')

            duration = 0.

            while not session.should_stop():
                start_time = time.time()

                x, y1, y2 = build_batch(train_data, batch_size)

                _, loss_val, summary_val, global_step_val, learning_rate_val, _, _, _, _, _, _, _, _, _, _, _ = session.run(
                    [train_op, loss,
                     summary, global_step,
                     learning_rate, usenow3_accuracy_update_op,
                     ecignow_accuracy_update_op, usenow3_precisions_update_op,
                    ] + summary_ops, feed_dict={
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
        log_dir=os.path.join('.', 'logs2'),
        checkpoint_file=None,  # os.path.join('.', 'logs', 'model.ckpt'),
        batch_size=32,
        learning_rate=0.5,
        decay_steps=100000,
        decay_rate=0.96,
        dropout_rate=0.01,
        max_steps=10000000
    )
