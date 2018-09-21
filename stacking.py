# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 08:53:17 2018

@author: lenovo
"""
from utils import stacking1,stacking2
import pandas as pd
if __name__ == '__main__':
    """程序入口"""
    train=pd.read_csv('train.csv')
    test=pd.read_csv('test.csv')
    label=train['label']
    
    stacking1(train.drop(['label'],axis=1),label,test)
    stacking2(train.drop(['label'],axis=1),label,test)