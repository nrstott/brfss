import os

import tensorflow as tf
from brfss.multitaskdnn import model_fn
from brfss.data import load_train_data, columns

data_dir = os.path.join('.', 'brfss', 'data')


def train(train_filepath, eval_filepath, model_dir, batch_size, learning_rate):
    def features_dict(df):
        return {
            'WEIGHT2': df.WEIGHT2.values,
            'HEIGHT3': df.HEIGHT3.values,
            'SEX': df.SEX.values,
            'EMPLOY1': df.EMPLOY1.values,
            'INCOME2': df.INCOME2.values,
            'MARITAL': df.MARITAL.values,
            'EDUCA': df.EDUCA.values,
            'CHILDREN': df.EDUCA.values,
            '_AGEG5YR': df._AGEG5YR.values
        }

    def labels_dict(df):
        return {
            'USENOW3': df.USENOW3.values,
            'ECIGNOW': df.ECIGNOW.values
        }

    train_df = load_train_data(train_filepath)

    train_df_size = len(train_df.index)
    usenow3_size = len(train_df.loc[train_df.USENOW3 == 1].index)
    ecignow_size = len(train_df.loc[train_df.ECIGNOW == 1].index)
    print('train_df_size: %f' % train_df_size)
    print('usenow3_size: %f, ecignow_size: %f' % (usenow3_size, ecignow_size))

    scale_usenow3 = 100 / (usenow3_size / train_df_size * 100)
    scale_ecignow = 100 / (ecignow_size / train_df_size * 100)

    print('usenow3 weight: %f, ecignow weight: %f' % (scale_usenow3, scale_ecignow))

    def weights(df):
        weight_usenow3 = df.USENOW3.apply(lambda x: 1. if x == 0 else scale_usenow3)
        weight_ecignow = df.ECIGNOW.apply(lambda x: 1. if x == 0 else scale_ecignow)
        return (weight_usenow3 + weight_ecignow) / 2.

    train_data = features_dict(train_df)
    train_data['weight'] = weights(train_df)

    train_labels = labels_dict(train_df)

    del train_df

    train_input_fn = tf.estimator.inputs.numpy_input_fn(x=train_data, y=train_labels, batch_size=batch_size,
                                                        num_epochs=None, shuffle=True, queue_capacity=10000)

    model = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, params={'hidden_units': [150, 50],
                                                                                   'feature_columns': columns,
                                                                                   'learning_rate': learning_rate,
                                                                                   'weight_column': 'weight'})
    model.train(train_input_fn, steps=100000)

    eval_df = load_train_data(eval_filepath)
    eval_data = features_dict(eval_df)
    eval_labels = labels_dict(eval_df)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x=eval_data, y=eval_labels, batch_size=batch_size, shuffle=False)

    model.evaluate(eval_input_fn)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    train(train_filepath=os.path.join(data_dir, 'LLCP2016_train.csv'),
          eval_filepath=os.path.join(data_dir, 'LLCP2016_eval.csv'),
          model_dir='./model',
          batch_size=32,
          learning_rate=0.1)
