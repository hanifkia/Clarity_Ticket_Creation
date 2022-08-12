import numpy as np
from math import ceil
import joblib

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.keras.preprocessing.sequence import pad_sequences

import warnings
warnings.filterwarnings("ignore")        
        

class Preprocessing():
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        
    # def __repr__(self):
    #     pass
    

    def __train_test_split(self):
        stratify = None if self.kwargs['stratify'] == None else self.kwargs['yData'][:,self.kwargs['labels_name'].index(self.kwargs['stratify'])]
        return train_test_split(self.kwargs['xData'], self.kwargs['yData'], 
                                                                            train_size=self.kwargs['train_size'],
                                                                            shuffle=self.kwargs['shuffle'],
                                                                            stratify = stratify
                                                                           )
        
    
    def __getPaddingMaxLen(self, arr):
        self.maxLen = np.vectorize(lambda x: x.shape[0])(arr)
        self.maxLen = ceil((np.mean((self.maxLen.mean(), self.maxLen.std())) + self.maxLen.max()) / 2) * 2

    
    def padding(self, data, maxlen = None):
        maxlen = self.maxLen if maxlen == None else maxlen
        self.kwargs['maxLen'] = maxlen
        return pad_sequences(data, maxlen = maxlen,
                             dtype='object',
                             padding = 'pre',
                             value = 0
                            )
    def __check_save_path(self, path):
        if os.path.isdir(path):
            os.system('rm -rf {}*.joblib'.format(path))
        else:
            os.mkdir(path)

    def __saveData(self, kw, trainData, testData):
        path = './data/{}/'.format(kw['data_mode'])
        self.__check_save_path(path)
        del kw['xData']
        del kw['yData']
        joblib.dump(trainData ,path + 'train.joblib')
        joblib.dump(testData ,path + 'test.joblib')
        joblib.dump(kw, path + 'info.joblib')
        
    def get_xy(self, path):
        train = joblib.load(path + 'train.joblib')
        test = joblib.load(path + 'test.joblib')
        info = joblib.load(path + 'info.joblib')
        return train['x'], test['x'], train['y'], test['y'], info

    
    def fit(self):
        xTrain , xTest, yTrain, yTest = self.__train_test_split()
        if self.kwargs['padding_max_len'] == None:
            self.__getPaddingMaxLen(xTrain)
        else:
            self.maxLen = self.kwargs['padding_max_len']

        xTrain = self.padding(xTrain)
        xTest = self.padding(xTest)                
        self.features_name = self.kwargs['features_name']
        self.labels_name = self.kwargs['labels_name']
        self.__saveData(self.kwargs,
                 {'x' : xTrain, 'y' : yTrain},
                 {'x' : xTest, 'y' : yTest}
                )


class FeatureEncoder():
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        
    # def __repr__(self):
    #     pass
    
        
    def __uValues__(self, xTrain):
        self.cat_unique_values = {}
        for col in self.kwargs['cat_features']:
            tmp = [0]
            tmp.extend(list(np.unique(xTrain[:,:,self.kwargs['features_name'].index(col)].reshape(-1,1))))
            self.cat_unique_values[col] = list(map(int, list(set(tmp))))
        
    def __numericalEncode(self, xTrain):
        self.numEncoder = {}
        for col in self.kwargs['num_features']:
            s = MinMaxScaler()
            s.fit(xTrain[:,:,self.kwargs['features_name'].index(col)])
            self.numEncoder[col] = s
            
    def numTransform(self, data, encoderDict):
        for col in self.kwargs['num_features']:
            data[:,:,self.kwargs['features_name'].index(col)] = encoderDict[col].transform(data[:,:,self.kwargs['features_name'].index(col)])
            # data[col] = data[col].apply(lambda x: encoderDict[col].transform(np.array(x).reshape(-1,1))[:,0])
        return data
      
    
    def __categoricalEncode(self, xTrain):
        self.catEncoder = {}
        self.cat_unique_values = {}
        for col in self.kwargs['cat_features']:
            s = LabelEncoder()
            s.fit(np.unique(xTrain[:,:,self.kwargs['features_name'].index(col)]))
            # self.cat_unique_values[col] = np.unique(xTrain[:,:,self.kwargs['features_name'].index(col)].reshape(-1,1))
            self.catEncoder[col] = s
            
    def catTransform(self, data, encoderDict):
        for col in self.kwargs['cat_features']:
            tmp = encoderDict[col].transform(np.reshape(data[:,:,self.kwargs['features_name'].index(col)], (-1,1)))
            data[:,:,self.kwargs['features_name'].index(col)] = np.reshape(tmp, (data[:,:,self.kwargs['features_name'].index(col)].shape[0], data[:,:,self.kwargs['features_name'].index(col)].shape[1]))  
            # data[col] = data[col].apply(lambda x: encoderDict[col].transform(np.array(x).reshape(-1,1)))
        return data
    
    def manualEncoder(self, data, manualDict):
        for item in manualDict.keys():
            mDict = manualDict[item]
            func = np.vectorize(lambda x: mDict[x] if x in mDict.keys() else x)
            return func(data)
    
    def fit(self, xTrain, xTest, manualDict):
        xTrain = self.manualEncoder(xTrain, manualDict)
        xTest = self.manualEncoder(xTest, manualDict)
        self.__numericalEncode(xTrain)
        self.__categoricalEncode(xTrain)
        xTrain = self.numTransform(xTrain, self.numEncoder)
        xTest = self.numTransform(xTest, self.numEncoder)
        xTrain = self.catTransform(xTrain, self.catEncoder)
        xTest = self.catTransform(xTest, self.catEncoder)
        self.__uValues__(xTrain)
        return xTrain, xTest

        
        
        
        