"""
Created on March 2, 2012 

@author Paul Torres

This file contains the data preprocessing pipeline for recreation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


def preprocessing(df0):
    df = df0.copy()
    
    df['area code'] = df['area code'].astype(str, errors = 'ignore')

    df['voice mail plan'] = np.where( df['voice mail plan']== 'yes', '1', '0')
    df['international plan'] = np.where( df['international plan']== 'yes', '1', '0')


    df['churn'] = df['churn'].astype(str, errors = 'ignore')
    df['churn'] = np.where( df['churn']== 'True', '1', '0')

    df = df.drop(['phone number','state'], axis=1)
    
    df['total_minutes'] = (df['total day minutes'] + df['total eve minutes'] + df['total intl minutes'] + df['total night minutes'])
    
    df['day_perc'] = (df['total day minutes'] / df['total_minutes'])
    df['night_perc'] = (df['total night minutes'] / df['total_minutes'])
    df['eve_perc'] = (df['total eve minutes'] / df['total_minutes'])
    df['intl_perc'] = (df['total intl minutes'] / df['total_minutes'])


    df['total_calls'] = (df['total day calls'] + df['total eve calls'] + df['total night calls'] + df['total intl calls'])
    """
    The following conditionals were dervied when I created the percentage features above. Anything over the 75 percentile
    is classified as this category
    """
    df['night_owl'] = np.where( (df['night_perc'] > np.percentile(df['night_perc'],75) ) ,'1','0')
    df['day_only'] = np.where( (df['day_perc'] > np.percentile(df['day_perc'],75) ) ,'1','0')
    df['traveler'] = np.where( (df['intl_perc'] > np.percentile(df['intl_perc'],75) ) ,'1','0')
    df['eve_only'] = np.where( (df['eve_perc'] > np.percentile(df['eve_perc'],75) ) ,'1','0')
    
    corr = df.corr()
    
    
    upper_tri = corr.where(np.triu(np.ones(corr.shape),k=1).astype(np.bool))
    
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.8)] 
    
    features = [x for x in df.columns.values if x not in to_drop]
    
    df1 = df[features]
    
    return df1


def process_variables(X_train, X_test,cat_lst,con_lst):
    
    ohe = OneHotEncoder(handle_unknown='ignore')
    X_train_enc_arr = ohe.fit_transform(X_train[cat_lst]).toarray()
    X_train_enc = pd.DataFrame(data = X_train_enc_arr,
                        columns = ohe.get_feature_names(cat_lst))
    
    X_test_enc_arr = ohe.transform(X_test[cat_lst]).toarray()
    X_test_enc = pd.DataFrame(data = X_test_enc_arr,
                        columns = ohe.get_feature_names(cat_lst))
    
    scaler = StandardScaler()
    X_train_cont_arr = scaler.fit_transform(X_train[con_lst])
    X_train_scal = pd.DataFrame(data = X_train_cont_arr,
                         columns = con_lst)

    X_test_cont_arr = scaler.transform(X_test[con_lst])
    X_test_scal = pd.DataFrame(data = X_test_cont_arr,
                         columns = con_lst)
    
    X_train_final = X_train_scal[con_lst].merge(X_train_enc,left_index=True, right_index=True)
    X_test_final = X_test_scal[con_lst].merge(X_test_enc,left_index=True, right_index=True)
    
    
    
    
    return X_train_final, X_test_final


def scale_con_var(X_train,X_test,lst):
    scaler = StandardScaler()
    X_train_cont_arr = scaler.fit_transform(X_train[lst])
    X_train_scal = pd.DataFrame(data = X_train_cont_arr,
                         columns = lst)

    X_test_cont_arr = scaler.transform(X_test[lst])
    X_test_scal = pd.DataFrame(data = X_test_cont_arr,
                         columns = lst)
    
    return X_train_scal, X_test_scal