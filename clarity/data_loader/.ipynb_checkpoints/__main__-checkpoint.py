import pandas as pd
from datetime import timedelta
from tqdm import tqdm
import numpy as np
from math import ceil
import os
import re
import joblib

import warnings
warnings.filterwarnings("ignore")      


class DataModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.__check_save_path()
        self.__read()
        self.__toDateTime__()
        
    def __check_save_path(self):
        if os.path.isdir(self.kwargs['save_path']):
            os.system('rm -rf {}*.joblib'.format(self.kwargs['save_path']))
        else:
            os.mkdir(self.kwargs['save_path'])       
            
    
    def __read(self):
        chunk = pd.read_csv(self.kwargs['data_path'], sep = self.kwargs['sep'], chunksize=10000)
        self.df = pd.concat(chunk, ignore_index=True)

    def __toDateTime__(self):
        for col in self.df.columns:
            if 'Time' in col or '_time' in col:
                self.df[col] = pd.to_datetime(self.df[col])
    
    def histAndSubseq(self, data, alam):
        hist = data.groupby('siteCode').get_group(alam['siteCode'])
        hist = hist.loc[(hist['reportedTime'] < alam['reportedTime']) & (hist['reportedTime'] > alam['reportedTime'] - timedelta(minutes = self.kwargs['histTime']))]
        hist['duration'] = hist['reportedTime'].apply(lambda x : (x - alam['reportedTime']).total_seconds())

        subseq = data.loc[(alam['reportedTime'] <= data['reportedTime']) & (data['reportedTime'] < alam['reportedTime']+timedelta(minutes=self.kwargs['holdTime']))]
        subseq['duration'] = subseq['reportedTime'].apply(lambda x : (x - alam['reportedTime']).total_seconds())    
        
        frame = pd.concat([hist, subseq])
        frame['mask'] = frame['siteCode'].apply(lambda x: 1 if x==alam['siteCode'] else 2)
        
        return frame[self.kwargs['features_name']].values
    
    def fit(self, dmType, save_rate = None):    
        func = getattr(self, dmType)
        save_rate = self.df.shape if save_rate == None else save_rate
        xData = []
        yData = []
        c=0
        for i in tqdm(range(len(self.df)), desc = 'generate sequences'):
            alam = self.df.iloc[i].to_dict()
            xData.append(func(self.df, alam))
            yData.append(self.df[self.kwargs['labels_name']].iloc[i].values)
            if (i+1)%save_rate == 0 and i!= 0 or (i+1) == len(self.df):
                joblib.dump({'X' : np.array(xData),
                             'Y' : np.array(yData),
                             'features_name' : self.kwargs['features_name'],
                             'labels_name' : self.kwargs['labels_name']
                            }, self.kwargs['save_path'] + 'batch_{}_{}.joblib'.format(c, i))
                c = i
                xData = []
                yData = []
        
        
        
def read_sequences(save_path):
    f = ["{}".format(filename) for _, _, filenames in os.walk(save_path) for filename in filenames if re.match(r'^.*\.(?:joblib)$', filename)]
    xData = []
    yData = []
    trap = []
    for i in tqdm(range(len(f)), desc = 'load sequences'):
        tmp = joblib.load(save_path + f[i])
        if len(tmp['X'].shape) == 1:

            xData.append(tmp['X'])
            yData.append(tmp['Y'])
        else:
            trap.append(i)
    return np.concatenate(xData), np.concatenate(yData), trap

