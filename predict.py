import os

import tensorflow as tf

from brfss.data import TrainData
from brfss.estimator import read_csv, dnn, feature_columns


data_dir = os.path.join(os.path.dirname(__file__), 'brfss', 'data')

def from_dataset(ds):
    return lambda: ds.make_one_shot_iterator().get_next()

if __name__ == '__main__':
	predict_data_path = os.path.join(data_dir, 'LLCP2016_holdout.csv')

	predict_data = TrainData(train_path=predict_data_path).filter_by_features(feature_columns).batch(64)

	label_vocabulary = list(map(lambda x: str(x), list(read_csv('ECIGNOW').value)))

	features, estimator = dnn(label_vocabulary, model_dir='./model')

	predict_result = estimator.predict(from_dataset(predict_data))

	for x in predict_result:
		print(x)