import argparse
import joblib
import numpy as np

from clarity.preprocessing import Preprocessing, FeatureEncoder
from clarity.data_loader import DataModel, read_sequences
from clarity.models import AutoEncoder, CnnClassifier
from clarity.models.utils import *
from clarity.models import utils

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import warnings
warnings.filterwarnings("ignore")


def str2list(item):
    l = item.strip().split(',')
    return [f.strip() for f in l]


arg = argparse.ArgumentParser()

######################################################################
################## data loading and processing arguments #############
######################################################################

arg.add_argument('-d', '--data_path', type=str,
                 default='./data/Clarity_cleanData_sample.csv',
                 help='path to labeled alarm data')

arg.add_argument('-tp', '--temp_path', type=str,
                 default='./data/temporary/',
                 help='path to save sequential processed data format')

arg.add_argument('-fn', '--features_name', type=str2list,
                 default='duration, timeType, mask, count, cell',
                 help='name of features')

arg.add_argument('-ln', '--labels_name', type=str2list,
                 default='l_cellDown, l_siteDown',
                 help='name of labels')

arg.add_argument('-dm', '--data_mode', type=str,
                 default='histAndSubseq',
                 choices = ['histAndSubseq'],
                 help='data model type')

arg.add_argument('-sr', '--save_rate', type=int,
                 default=1000,
                 help='size of each saved batch')

arg.add_argument('-ud', '--used_data', type=str,
                 default='load',
                 choices = ['load', 'generate'],
                 help='load saved data or generate new sequential data from csv')

arg.add_argument('-st', '--histTime', type=float,
                 default=24*60,
                 help='past window time duration')

arg.add_argument('-dt', '--holdTime', type=float,
                 default=10,
                 help='future window time duration. catch time')

arg.add_argument('-nf', '--num_features', type=str2list,
                 default='duration',
                 help='list of numerical features')

arg.add_argument('-cf', '--cat_features', type=str2list,
                 default='count, cell, timeType, mask',
                 help='list of categorical features')

arg.add_argument('-ss', '--train_size', type=float,
                 default=0.8,
                 help='train data split size')

arg.add_argument('-tl', '--train_label', type=str,
                 default='l_cellDown',
                 choices = ['l_cellDown', 'l_siteDown'],
                 help='select label to train')

######################################################################
#################### machine learning models arguments ###############
######################################################################


arg.add_argument('-sp', '--save_path', type=str,
                 default='./saved_models/',
                 help='path to save models and checkpoints')

arg.add_argument('-ld', '--log_dir', type=str,
                 default='./logs',
                 help='path to save tensorboard logs')

arg.add_argument('-ae', '--ae', type=str,
                 default='train',
                 choices = ['train', 'load'],
                 help='train new autoencoder or load pre-trained one')

arg.add_argument('-cnn', '--cnn', type=str,
                 default='train',
                 choices = ['train', 'load'],
                 help='train new CNN clasifier or load a trained one')

arg.add_argument('-is', '--inStyle_func', type=str,
                 default='arr2dict',
                 choices = ['arr2dict'],
                 help='input style')


args = arg.parse_args()



if __name__ == "__main__":
    if args.used_data == 'generate':
        dm = DataModel(data_path = args.data_path,
                      sep = ',',
                      save_path = args.temp_path,
                      features_name = args.features_name,
                      labels_name = args.labels_name,
                      histTime = args.histTime,
                      holdTime = args.holdTime
                      )
        dm.fit(args.data_mode, save_rate=args.save_rate)
    
    xData, yData, _ = read_sequences(args.temp_path)
    
        
    pre = Preprocessing(
                        data_mode = args.data_mode,
                        num_features = args.num_features,
                        cat_features = args.cat_features,
                        features_name = args.features_name,
                        labels_name = args.labels_name,
                        xData = xData,
                        yData = yData,
                        train_size = args.train_size,
                        shuffle = True,
                        stratify = args.train_label,
                        padding_max_len = None
                       )
    pre.fit()
    xTrain, xTest, yTrain, yTest, info = pre.get_xy('./data/{}/'.format(args.data_mode))

    fe = FeatureEncoder(
                    num_features = args.num_features,
                    cat_features = args.cat_features,
                    features_name = args.features_name,
                )

    manualDict = {'timeType' : {'on' : 1, 'off' : 2}}
    xTrain , xTest = fe.fit(xTrain, xTest, manualDict)
    
    joblib.dump({'numerical' : fe.numEncoder,
                 'categorical' : fe.catEncoder,
                 'other' : manualDict} , args.save_path + 'scalerDict.joblib')
    

    ae = AutoEncoder(mode = 'train',
                     features_name = args.features_name,
                     cat_features = args.cat_features,
                     padding_max_len = info['maxLen'],
                     cat_unique_values = fe.cat_unique_values,
                     hidden_nLayers = 1,
                     hidden_dims = [100],
                     latent_dims = 80,
                     drop_out = 0.2,
                     kernel_regularizer = None,
                     activity_regularizer = None,
                     leatning_rate = 0.001,
                     batch_size = 32,
                     epochs = 80,
                     save_path = args.save_path,
                     log_dir = args.log_dir,
                    )
    
    if args.ae == 'train':
        ae.train(xTrain, xTest)
        ae.save_model()
    else:
        ae.load_model()
        
    clf = CnnClassifier(mode = 'train',
                        pre_train = ae.get_encoder(),
                        inStyle_func = args.inStyle_func,
                        features_name = args.features_name,
                        input_shape = None,
                        nClasses = np.unique(yTrain[:,args.labels_name.index(args.train_label)]).shape[0],
                        hidden_nLayers = 6,
                        filter_dims = [100, 100, 100, 100, 100, 100],
                        kernel_size = 4,
                        activation = 'relu',
                        conv_padding = 'same',
                        maxPooling = True,
                        pool_size = 2,
                        pooling_strides = 2,
                        pooling_padding = 'same',
                        drop_out = 0.2,
                        leatning_rate = 0.001,
                        batch_size = 32,
                        epochs = 22,
                        save_path = args.save_path,
                        log_dir = args.log_dir,                    
                       )
    
    
    if args.cnn == 'train':
        clf.train(xTrain,
                  yTrain[:,args.labels_name.index(args.train_label)],
                  xTest,
                  yTest[:,args.labels_name.index(args.train_label)])
        
        clf.save_model()
    else:
        clf.load_model()
    
    if args.inStyle_func != None:
        xTest = getattr(utils, args.inStyle_func)(xTest, args.features_name)
        
    y_pred = clf.model.predict(xTest)
    clf_performance(yTest[:,args.labels_name.index(args.train_label)],
                    y_pred)
                    
      
    

    
