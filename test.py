import joblib
import argparse
import numpy as np
from clarity.preprocessing import FeatureEncoder
from clarity.models import AutoEncoder, CnnClassifier
from clarity.models.utils import *

arg = argparse.ArgumentParser()

arg.add_argument('-d', '--data_mode', type=str,
                 default='histAndSubseq',
                 choices = ['histAndSubseq'],
                 help='data model type')

arg.add_argument('-m', '--model_path', type=str,
                 default='./saved_models/',
                 help='path to saved model')

args = arg.parse_args()

test_path = './data/{}/test.joblib'.format(args.data_mode)
info_path = './data/{}/info.joblib'.format(args.data_mode)


if __name__ == "__main__":
    
    # load data
    testData = joblib.load(test_path)
    info = joblib.load(info_path)
    xTest, yTest = testData['x'], testData['y']
    
    # load scaler and model
    scaler = joblib.load(args.model_path + 'scalerDict.joblib')
    clf = CnnClassifier(save_path = args.model_path, mode='inf')
    clf.load_model()
    
    # labeling
    fe = FeatureEncoder(cat_features = info['cat_features'],
                        num_features = info['num_features'],
                        features_name = info['features_name'])
    xTest = fe.manualEncoder(xTest, scaler['other'])
    xTest = fe.catTransform(xTest, scaler['categorical'])
    xTest = fe.numTransform(xTest, scaler['numerical'])
    
    # predict
    testDict = arr2dict(xTest, info['features_name'])
    y_pred = clf.predict(testDict)
    
    #prediction performance
    clf_performance(yTest[:,0], y_pred)





