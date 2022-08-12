import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score


def arr2dict(mat, feature_list):
    d = {}
    try:
        for i, col in enumerate(feature_list):
            d[col] = mat[:,:, i]
        return d
    except:
        return None
    

def clf_performance(yTest, y_pred):
    
    print('Fi Score: ', f1_score(yTest, np.argmax(y_pred,1)))
    print(classification_report(yTest, np.argmax(y_pred,1)))