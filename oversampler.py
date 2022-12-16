"""This module is used to do oversampling of the to improve performance of the model"""

import pandas as pd
import numpy as np
import math
import random

def oversampler(x:np.ndarray, y:np.ndarray)->tuple[np.ndarray, np.ndarray]:
    """
    This function do the oversampling for training data
    """

    df = pd.DataFrame(list(zip(x,y)), columns= ['coulomb matrix', 'coupling'])
    #df2 = pd.cut(df['coupling'], bins=np.arange(-1,110,10))
    #print(df2)

    minimal_coupling = math.floor(min(y)*10)/10
    maximum_coupling = math.ceil(max(y)*10)/10
    coupling_range = maximum_coupling - minimal_coupling

    #make into 10 classes
    df['coupling_interval'] = pd.cut(df['coupling'], bins=np.arange(minimal_coupling,maximum_coupling,coupling_range/10), include_lowest=True)
    target_counts = df['coupling_interval'].value_counts().max()

    dicts_to_append = []

    for i in df['coupling_interval'].dropna().unique():
        
        current_class_count = len(df[df['coupling_interval']==i])
        
        count_to_add = target_counts - current_class_count

        list_of_dict = df[df['coupling_interval']==i].to_dict('records')
        
        chosen_list_of_dict = random.choices(list_of_dict, k = count_to_add)
        
        dicts_to_append += chosen_list_of_dict

    modified_df = df.append(dicts_to_append, ignore_index=True, sort=False)
    
    modified_x = modified_df['coulomb matrix'].values.tolist()
    modified_y = modified_df['coupling'].values.tolist()

    return np.array(modified_x), np.array(modified_y)
