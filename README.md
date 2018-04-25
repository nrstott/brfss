# Predictive Analyatics on the Behavior Risk Factory Surveillance System (BRFSS)

Multitask DNN in Tensorflow for predictive analytics on the CMS Behavioral Risk Factor Surveillance System (BRFSS).

## Data

The [LLCP_2016.zip](brfss/data/LLCP_2016.zip) file in the [brfss/data](brfss/data) directory contains the 2016 BRFSS from 
CMS shuffled and separated into three files:

* LLCP2016_holdout.csv: 20% of the data held out.
* LLCP2016_train.csv: 80% of the non-holdout data for training the multitask deep neural network.
* LLCP2016_eval.csv: 20% of the non-holdout data for evaluating the multitask deep neural network during training.
