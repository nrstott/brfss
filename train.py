import os
import time

import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from tensorflow.python.estimator.model_fn import ModeKeys

from brfss.data import load_train_data, build_batch, columns

data_dir = os.path.join(os.path.dirname(__file__), 'brfss', 'data')


class MultitaskDNN:
    def __init__(self, input_layer, labels_ndims, dropout_rate=0):
        """Multitask DNN using the same hidden layers to train multiple logits layers.

        Args:
            input_layer (:obj:`tf.feature_columns.input_layer`) Input layer for the model.
            labels_ndims (int): Number of labels that this model will be trained on. One label per task.
            dropout_rate (float): Dropout rate of the dropout layers.
        """
        self.labels_ndims = labels_ndims
        self.dropout_rate = dropout_rate

        with tf.variable_scope('hidden1'):
            net = tf.layers.dense(input_layer, 64, tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer(),
                                  name='dense')
            tf.summary.histogram('hiddenlayer1/activation', net)
            net = tf.layers.dropout(net, dropout_rate, name='dropout')

        with tf.variable_scope('hidden2'):
            net = tf.layers.dense(net, 32, tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer(), name='dense')
            tf.summary.histogram('hiddenlayer2/activation', net)
            net = tf.layers.dropout(net, dropout_rate, name='dropout')

        self.logits_layers = []
        for i in range(self.labels_ndims):
            with tf.variable_scope('logits%d' % i) as logits_scope:
                dense = tf.layers.dense(net, 1, activation=None, kernel_initializer=tf.glorot_uniform_initializer(),
                                        name=logits_scope)
                tf.summary.histogram('logits%d/activation' % i, dense)
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


def _prediction_variables(logits):
    logistic = tf.squeeze(tf.sigmoid(logits, name='logistic'))
    two_class_logits = tf.concat((tf.zeros_like(logits), logits), axis=-1)
    probabilities = tf.nn.softmax(two_class_logits, axis=-1)
    class_ids = tf.to_int32(tf.greater_equal(logistic, 0.1, name='class_ids'))

    return logistic, probabilities, class_ids


def _add_summaries(labels, logistic, family, n_examples):
    labels = tf.squeeze(labels)
    (true_positives, true_positives_update_op) = tf.metrics.true_positives(labels=labels,
                                                                           predictions=logistic)

    (false_positives, false_positives_update_op) = tf.metrics.false_positives(labels=labels,
                                                                              predictions=logistic)

    (true_negatives, true_negatives_update_op) = tf.metrics.true_negatives(labels=labels,
                                                                           predictions=logistic)

    (false_negatives, false_negatives_update_op) = tf.metrics.false_negatives(labels=labels,
                                                                              predictions=logistic)

    true_positives_percent = tf.divide(tf.to_float(true_positives), tf.to_float(n_examples),
                                       name='true_positives_percent')
    true_negatives_percent = tf.divide(tf.to_float(true_negatives), tf.to_float(n_examples),
                                       name='true_negatives_percent')
    false_positives_percent = tf.divide(tf.to_float(false_positives), tf.to_float(n_examples),
                                        name='false_positives_percent')
    false_negatives_percent = tf.divide(tf.to_float(false_negatives), tf.to_float(n_examples),
                                        name='false_negatives_percent')

    true_positives_percent = tf.Print(true_negatives_percent,
                                      [labels, logistic, n_examples, true_positives, true_negatives, false_positives, false_negatives,
                                       true_positives_percent, true_negatives_percent, false_negatives_percent,
                                       false_negatives_percent])

    tf.summary.scalar('true_positives', true_positives_percent, family=family)
    tf.summary.scalar('true_negatives', true_negatives_percent, family=family)
    tf.summary.scalar('false_positives', false_positives_percent, family=family)
    tf.summary.scalar('false_negatives', false_negatives_percent, family=family)

    return [true_positives_update_op, false_positives_update_op,
            true_negatives_update_op, false_negatives_update_op]


def build_graph(checkpoint_dir, log_dir, batch_size, max_steps, dropout_rate=0,
                mode=tf.estimator.ModeKeys.TRAIN,
                learning_rate=None,
                decay_rate=None,
                decay_steps=None):
    g = tf.Graph()
    with g.as_default():
        y_usenow3 = tf.placeholder(tf.int32, shape=[batch_size, 1], name='usenow3')
        y_ecignow = tf.placeholder(tf.int32, shape=[batch_size, 1], name='ecignow')

        global_step = tf.train.get_or_create_global_step()

        if mode == ModeKeys.TRAIN:
            if learning_rate is None:
                raise Exception('learning_rate must be set in train mode')

            if (decay_steps is None and decay_rate is not None) or (decay_rate is None and decay_steps is not None):
                raise Exception('decay_steps and decay_rate must both be set if one is set')

            if decay_steps is not None and decay_rate is not None:
                learning_rate = tf.train.exponential_decay(learning_rate, global_step=global_step,
                                                           decay_steps=decay_steps,
                                                           decay_rate=decay_rate)

            examples = (float(batch_size) * tf.to_float(global_step))
        else:
            examples = float(batch_size)

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

        loss = model.loss([y_usenow3, y_ecignow])

        with tf.variable_scope('predictions_usenow3'):
            logistic_usenow3, probabilities_usenow3, class_ids_usenow3 = _prediction_variables(
                model.logits_layers[0])

        with tf.variable_scope('predictions_ecignow'):
            logistic_ecignow, probabilities_ecignow, class_ids_ecignow = _prediction_variables(
                model.logits_layers[1])

        predictions_usenow3_str = tf.reduce_join(tf.as_string(class_ids_usenow3))
        predictions_ecignow_str = tf.reduce_join(tf.as_string(class_ids_ecignow))

        labels_usenow3 = tf.reduce_join(tf.as_string(y_usenow3), axis=0)

        (usenow3_accuracy, usenow3_accuracy_update_op) = tf.metrics.accuracy(
            labels=labels_usenow3,
            predictions=predictions_usenow3_str)
        (ecignow_accuracy, ecignow_accuracy_update_op) = tf.metrics.accuracy(
            labels=tf.reduce_join(tf.as_string(y_ecignow), axis=0),
            predictions=predictions_ecignow_str)

        summary_ops = []

        summary_ops += _add_summaries(y_usenow3, class_ids_usenow3, family='usenow3', n_examples=examples)
        summary_ops += _add_summaries(y_ecignow, class_ids_ecignow, family='ecignow', n_examples=examples)

        (usenow3_precisions, usenow3_precisions_update_op) = \
            tf.metrics.precision_at_thresholds(labels=y_usenow3,
                                               predictions=logistic_usenow3,
                                               thresholds=[0.1, 0.5, 0.75])
        tf.summary.scalar('precision_at_0.1', usenow3_precisions[0], family='usenow3')
        tf.summary.scalar('precision_at_0.5', usenow3_precisions[1], family='usenow3')
        tf.summary.scalar('precision_at_0.75', usenow3_precisions[2], family='usenow3')
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('usenow3_accuracy', usenow3_accuracy, family='usenow3')
        tf.summary.scalar('ecignow_accuracy', ecignow_accuracy, family='ecignow')
        tf.summary.histogram('probabilities', probabilities_usenow3, family='usenow3')
        tf.summary.histogram('probabilities', probabilities_ecignow, family='ecignow')

        if mode == ModeKeys.TRAIN:
            tf.summary.scalar('learning_rate', learning_rate)

        summary = tf.summary.merge_all()
        saver = tf.train.Saver()

        if mode == ModeKeys.TRAIN:
            save_increment = 1000
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=global_step)
            checkpoint_saver_hook = tf.train.CheckpointSaverHook(checkpoint_dir=checkpoint_dir,
                                                                 save_steps=save_increment, saver=saver)
            summary_saver_hook = tf.train.SummarySaverHook(save_steps=save_increment, output_dir=log_dir,
                                                           summary_op=summary)
            profiler_hook = tf.train.ProfilerHook(save_steps=save_increment, output_dir=log_dir)
            stop_at_step_hook = tf.train.StopAtStepHook(num_steps=max_steps)
            logging_hook = tf.train.LoggingTensorHook({
                'loss': loss,
                # 'learning_rate': learning_rate if learning_rate is not None else 0,
                'usenow3_accuracy': usenow3_accuracy,
                'ecignow_accuracy': ecignow_accuracy,
                'usenow3_precision_at_thresholds': usenow3_precisions}, every_n_iter=save_increment)

            hooks = [checkpoint_saver_hook, summary_saver_hook, profiler_hook, stop_at_step_hook, logging_hook]

            ops = [global_step, train_op, loss,
                   usenow3_accuracy_update_op,
                   ecignow_accuracy_update_op,
                   usenow3_precisions_update_op] + summary_ops
        else:
            summary_saver_hook = tf.train.SummarySaverHook(save_steps=1, output_dir=log_dir,
                                                           summary_op=summary)
            hooks = [summary_saver_hook]
            ops = [summary, usenow3_accuracy, ecignow_accuracy, usenow3_accuracy_update_op,
                   ecignow_accuracy_update_op] + summary_ops

        return g, saver, (features, y_usenow3, y_ecignow), hooks, ops


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

    (graph, saver, (features, y_usenow3, y_ecignow), hooks, ops) = build_graph(mode=ModeKeys.TRAIN,
                                                                               checkpoint_dir=log_dir,
                                                                               log_dir=os.path.join(log_dir, 'train'),
                                                                               batch_size=batch_size,
                                                                               max_steps=max_steps,
                                                                               dropout_rate=dropout_rate,
                                                                               learning_rate=learning_rate,
                                                                               decay_rate=decay_rate,
                                                                               decay_steps=decay_steps)

    with graph.as_default():
        latest_checkpoint = tf.train.latest_checkpoint(log_dir)

        with tf.train.SingularMonitoredSession(checkpoint_filename_with_path=latest_checkpoint,
                                               hooks=hooks) as session:
            print('Training')

            while not session.should_stop():
                x, y1, y2 = build_batch(train_data, batch_size)

                session.run(
                    ops, feed_dict={
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

    tf.reset_default_graph()

    (eval_graph, saver, (features, y_usenow3, y_ecignow), hooks, ops) = build_graph(mode=ModeKeys.EVAL,
                                                                                    checkpoint_dir=log_dir,
                                                                                    log_dir=os.path.join(log_dir,
                                                                                                         'eval'),
                                                                                    batch_size=2048,
                                                                                    max_steps=1)

    with eval_graph.as_default():
        latest_checkpoint = tf.train.latest_checkpoint(log_dir)

        with tf.train.SingularMonitoredSession(checkpoint_filename_with_path=latest_checkpoint, hooks=hooks) as session:
            print('Evaluating')
            x, y1, y2 = build_batch(eval_data, 2048)

            results = session.run(
                ops, feed_dict={
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
            print(results[1:])
            x, y1, y2 = build_batch(eval_data, 2048)

            results = session.run(
                ops, feed_dict={
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
            print(results[1:])


if __name__ == '__main__':
    train(
        train_data_path=os.path.join(data_dir, 'LLCP2016_train.csv'),
        eval_data_path=os.path.join(data_dir, 'LLCP2016_train.csv'),
        log_dir=os.path.join('.', 'logs'),
        checkpoint_file=None,  # os.path.join('.', 'logs', 'model.ckpt'),
        batch_size=32,
        learning_rate=0.5,
        decay_steps=100000,
        decay_rate=0.96,
        dropout_rate=0.01,
        max_steps=5000
    )
