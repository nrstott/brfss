import numpy as np
import tensorflow as tf


def _add_metric(metric_fn, name, labels, predictions, family=None, eval_metrics=None):
    v = metric_fn(labels=labels, predictions=predictions, name=name)
    tf.summary.scalar(name, v[1], family=family)
    if eval_metrics is not None:
        if family is not None:
            eval_metrics['%s/%s/%s/%s' % (family, 'dnn', family, name)] = v
        else:
            eval_metrics[name] = v


def model_fn(features, labels, mode, params):
    feature_columns = params['feature_columns']
    hidden_units = params['hidden_units']
    learning_rate = getattr(params, 'learning_rate', 0.1)
    activation = getattr(params, 'activation', tf.nn.relu)
    dropout = getattr(params, 'dropout', 0.)

    net = tf.feature_column.input_layer(features, feature_columns)

    with tf.variable_scope('dnn') as dnn_scope:
        for idx, units in enumerate(hidden_units):
            with tf.variable_scope('hidden%d' % idx, values=(units, dropout)) as scope:
                net = tf.layers.dense(net, units, activation=activation,
                                      kernel_initializer=tf.glorot_uniform_initializer(),
                                      name=scope)

                tf.summary.histogram('hidden%d/activation' % idx, net)

                net = tf.layers.dropout(net, rate=dropout)

        logits_layers = []
        for key, value in labels.items():
            with tf.variable_scope('logits_%s' % key) as scope:
                logits = tf.layers.dense(net, 1, activation=None, kernel_initializer=tf.glorot_uniform_initializer(),
                                         name=scope)
                tf.summary.histogram('logits_%s/activation' % key, logits)
                logits_layers.append(logits)

        predicted_classes_by_layer = []
        class_ids_by_layer = []
        probabilities_by_layer = []

        for logits in logits_layers:
            predicted_classes = tf.round(tf.sigmoid(logits))
            class_ids = [predicted_classes[:, tf.newaxis]]
            probabilities = [tf.nn.softmax(logits)]

            predicted_classes_by_layer.append(predicted_classes)
            class_ids_by_layer.append(class_ids)
            probabilities_by_layer.append(probabilities)

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'class_ids': class_ids_by_layer,
                'probabilities': probabilities_by_layer,
                'logits': tf.stack(logits_layers)
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        metrics = {}

        for (labels_key, labels_val), predicted_classes_val in zip(labels.items(), predicted_classes_by_layer):
            _add_metric(tf.metrics.true_positives, 'true_positives', labels_val, predicted_classes_val,
                        family=labels_key)

            _add_metric(tf.metrics.true_negatives, 'true_negatives', labels_val, predicted_classes_val,
                        family=labels_key)

            _add_metric(tf.metrics.false_positives, 'false_positives', labels_val, predicted_classes_val,
                        family=labels_key)

            _add_metric(tf.metrics.false_negatives, 'false_negatives', labels_val, predicted_classes_val,
                        family=labels_key)

            _add_metric(tf.metrics.auc, 'auc', labels_val, predicted_classes_val, family=labels_key,
                        eval_metrics=metrics)

            _add_metric(tf.metrics.accuracy, 'accuracy', labels_val, predicted_classes_val,
                        family=labels_key, eval_metrics=metrics)

            _add_metric(tf.metrics.recall, 'recall', labels_val, predicted_classes_val, family=labels_key,
                        eval_metrics=metrics)

            _add_metric(tf.metrics.precision, 'precision', labels_val, predicted_classes_val, family=labels_key,
                        eval_metrics=metrics)

        loss = 0.

        for labels_val, logits_val in zip(labels.values(), logits_layers):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_float(labels_val[:, tf.newaxis]),
                                                             logits=logits_val)

            def _weight(label):
                label = tf.squeeze(label)
                return tf.cond(tf.equal(label, 1), lambda: np.int64(10), lambda: np.int64(1))

            weights = tf.map_fn(_weight, labels_val[:, tf.newaxis], name='weights')[:, tf.newaxis]

            weighted_loss = tf.losses.compute_weighted_loss(losses,
                                                            weights=weights,
                                                            reduction=tf.losses.Reduction.MEAN)

            loss += weighted_loss

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
