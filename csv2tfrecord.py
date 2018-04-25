import os
import math

import pandas as pd
import tensorflow as tf

from brfss.data import _train_headers as headers
from brfss.estimator import feature_columns, ecignow_feature_column, get_feature_column_key

data_dir = os.path.join('.', 'brfss', 'data')
print('data directory: %s' % data_dir)

keys = [get_feature_column_key(x) for x in (feature_columns + [ecignow_feature_column])]
print('keys: %s' % ','.join(keys))

df = pd.read_csv(os.path.join(data_dir, 'LLCP2016_eval.csv'), sep=',', header=None, names=headers)
df = df.loc[df['ECIGNOW'].notna(), keys]

df.SXORIENT = df.SXORIENT.fillna(-1).astype(int)
df.EMPLOY1 = df.EMPLOY1.fillna(-1).astype(int)
df.INCOME2 = df.INCOME2.fillna(-1).astype(int)
df.INTERNET = df.INTERNET.fillna(-1).astype(int)
df.MARITAL = df.MARITAL.fillna(-1).astype(int)
df.EDUCA = df.EDUCA.fillna(-1).astype(int)
df.VETERAN3 = df.VETERAN3.fillna(-1).astype(int)
df._AGEG5YR = df._AGEG5YR.fillna(-1).astype(int)
df.ECIGNOW = df.ECIGNOW.astype(str)

print(df.head(2))

with tf.python_io.TFRecordWriter(os.path.join(data_dir, 'LLCP2016_eval.tfrecords')) as writer:
    print('writing records')
    for row in df.values:
        feature={
            'WEIGHT2': tf.train.Feature(float_list=tf.train.FloatList(value=[row[0]])),
            'HEIGHT3': tf.train.Feature(float_list=tf.train.FloatList(value=[row[1]])),
            'SEX': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(row[2])])),
            'SXORIENT': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(row[3])])),
            'EMPLOY1': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(row[4])])),
            'INCOME2': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(row[5])])),
            'INTERNET': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(row[6])])),
            'MARITAL': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(row[7])])),
            'EDUCA': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(row[8])])),
            'VETERAN3': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(row[9])])),
            '_AGEG5YR': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(row[10])])),
            'CHILDREN':tf.train.Feature(float_list=tf.train.FloatList(value=[row[11]])),
            'ECIGNOW':tf.train.Feature(bytes_list=tf.train.BytesList(value=[row[12]]))
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())