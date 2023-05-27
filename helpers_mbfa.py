import tomli
import numpy as np
import pandas as pd
from pyts.image import GramianAngularField

WINDOW = 60

def get_toml_data(path):
    """
    return data like a dict (key value pair)
    :param path: file path
    :type path: str
    :return: tomi infos
    """

    with open(path, mode='rb') as f:
        return tomli.load(f)
    
def rescale_and_gaf_15(data, column):
    min_ = min(data['close_d'].values[-1] - data['adr'].values[-1], min(data.low.values))
    max_ = max(data['close_d'].values[-1] + data['adr'].values[-1], max(data.high.values))
    
    scaled_serie = np.array((2*data[column].values - max_ - min_) / (max_ - min_)).reshape(1, -1)

    gaf = GramianAngularField(WINDOW, None, 'd')
    
    img = gaf.fit_transform(scaled_serie)

    return img

def scale_gaf_d(data, column):
    min_ = data['low'].min()
    max_ = data['high'].max()
    
    scaled_serie = np.array((2*data[column].values - max_ - min_) / (max_ - min_)).reshape(1, -1)
    gaf = GramianAngularField(WINDOW, None, 'd')
    
    img = gaf.fit_transform(scaled_serie)
    
    return img

def no_scale_gaf(data, column):
    
    serie = np.array(data[column].values).reshape(1,-1)
    gaf = GramianAngularField(WINDOW, (-1, 1), 'd')
    
    img = gaf.fit_transform(serie)

    return img