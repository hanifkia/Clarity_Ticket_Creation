# Clarity Ticket Creation

# Prerequisites

* This project should work on any OS. Although, the OS used in development process is Ubuntu.
* Python >= 3.6.5
* CUDA version: 10.2

# Getting Started
## Installation

* Clone this repo

```bash
git clone url/[your user]/ticket_creation
cd ticket_creation
```
It is suggested to create a virtual environment. Having created the env, the easiest way to install the dependencies is to run the follwoing command:

```bash
pip install -r requirements.txt 
```

## Train the model

to train the model, past the data into data directory, then run the following command:

```bash
python3 train.py -ud generate
```
once you running the train.py script, train and test data will store in the data directory, So they can be reloaded next time.
the training history saved in logs directory. to live view the loss and accuracy diagrams during the epochs you can use tensorboard by running the following command:

```bash
tensorboard --logdir=./logs
```

The arguments can be passed using command line are as follow:

```
-d --data_path : path to labeled alarm data
-tp --temp_path : path to save sequential processed data format
-fn --features_name : name of features
-ln --labels_name : name of labels
-dm --data_mode : data model type
-sr --save_rate : size of each saved batch
-ud --used_data : load saved data or generate new sequential data from csv
-st --histTime' : past window time duration
-dt --holdTime : future window time duration. catch time
-nf --num_features : list of numerical features
-cf --cat_features : list of categorical features
-ss --train_size : train data split size
-tl --train_label : select label to train
-sp --save_path' : path to save models and checkpoints
-ld --log_dir path to save tensorboard logs
-ae --ae : train new autoencoder or load pre-trained one
-cnn --cnn : train new CNN clasifier or load a trained one
-is --inStyle_func : function to convert xData style to qualify model input style
```

## Test the model
to evaluate the model run the test.py script as follow:

```bash
python3 test.py
```
The arguments can be passed using command line are as follow:

```
-d --data_mode : data model type
-m --model_path : path to saved model
```