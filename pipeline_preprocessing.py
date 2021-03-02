"""
Created on March 2, 2012 

@author Paul Torres

This file contains the data preprocessing pipeline for recreation.
"""

import pandas as pd
import numpy as np


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

