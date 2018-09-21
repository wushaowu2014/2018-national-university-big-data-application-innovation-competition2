# -*- coding: utf-8 -*-
"""
Created on Sun May 20 17:24:03 2018

@author: shaowu
"""
import numpy as np  
import pandas as pd  
import itertools
import matplotlib.pyplot as plt
import os
os.environ['KERAS_BACKEND']='tensorflow'#'theano'
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dense,Dropout,Convolution1D,Flatten,Conv1D, BatchNormalization,PReLU
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam,SGD
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from scipy import sparse
from sklearn.utils import shuffle
from sklearn.cross_validation import StratifiedKFold
from collections import Counter
from keras.models import load_model


from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix
import time
seed=7
np.random.seed(seed)
def label_to_new(label):
    """决赛与预赛标签映射"""
    if label==1:
        return 9
    elif label==2:
        return 4
    elif label==3:
        return 8
    elif label==4:
        return 2
    elif label==5:
        return 7
    elif label==7:
        return 5
    elif label==8:
        return 3
    elif label==9:
        return 1
    else:
        return label
def process(data):
    """把子牌映射到0-51"""
    m=[]
    for line in data.values:
        if line[0]=='C':
            if line[1]=='J':
                m.append(10)
            elif line[1]=='Q':
                m.append(11)
            elif line[1]=='K':
                m.append(12)
            else:
                m.append(int(line[1])-1)
        if line[0]=='H':
            if line[1]=='J':
                m.append(23)
            elif line[1]=='Q':
                m.append(24)
            elif line[1]=='K':
                m.append(25)    
            else:
                m.append(int(line[1])+12)
        if line[0]=='D':
            if line[1]=='J':
                m.append(36)  
            elif line[1]=='Q':
                m.append(37)
            elif line[1]=='K':
                m.append(38)
            else:
                m.append(int(line[1])+25)          
        if line[0]=='S':
            if line[1]=='J':
                m.append(49)     
            elif line[1]=='Q':
                m.append(50)  
            elif line[1]=='K':
                m.append(51)
            else:
                m.append(int(line[1])+38)   
    return m
def get_newdata(data):
    newdata=pd.DataFrame()
    newdata['feat1']=process(data[['ziduan1','num1']])
    newdata['feat2']=process(data[['ziduan2','num2']])
    newdata['feat3']=process(data[['ziduan3','num3']])
    newdata['feat4']=process(data[['ziduan4','num4']])
    newdata['feat5']=process(data[['ziduan5','num5']])
    if 'label' in data.columns:
        newdata['label']=data['label']
    else:
        pass
    return newdata
def gather_data1(data):
    """全排列"""
    newdata=pd.DataFrame()
    col_list=list(itertools.permutations(list(['feat1','feat2','feat3','feat4','feat5']),5))
    for i in col_list:
        col=[]
        for j in i:
            col=col+[j]
        if 'label' in list(data.columns): #data有标签情况
            m=data[col+['label']]
        else:                             #data无标签情况
            m=data[col]
        m.columns=data.columns
        newdata=pd.concat([newdata,m])
    return newdata
def feat_onehot1(data,cols):
    """
    data: DataFrame类型数据
    cols: list类型数据 data中字段名列表
    """
    m=to_categorical(data[cols[0]],52)
    for col in cols[1:]:
        m=np.hstack((m,to_categorical(data[col],52)))
    return m

def str_to_int(data):
    if data=='C':
        return 0
    elif data=='D':
        return 1
    elif data=='H':
        return 2
    elif data=='S':
        return 3
    elif data=='J':
        return 10
    elif data=='Q':
        return 11
    elif data=='K':
        return 12
    else:
        return int(data)-1
    
def lab_one_coder(df_train,df_test):
    """one-hot编码"""
    enc=OneHotEncoder()
    lb=LabelEncoder()
    feats=df_train.columns
    for k,feat in enumerate(feats):
        tmp=lb.fit_transform((list(df_train[feat])+list(df_test[feat])))
        enc.fit(tmp.reshape(-1,1))
        x_train=enc.transform(lb.transform(df_train[feat]).reshape(-1, 1))
      #  x_val=enc.transform(lb.transform(df_val[feat]).reshape(-1, 1))
        x_test=enc.transform(lb.transform(df_test[feat]).reshape(-1, 1))
        if k==0:
            X_train,X_test=x_train,x_test
        else:
            X_train,X_test=sparse.hstack((X_train, x_train)),sparse.hstack((X_test, x_test))
    return X_train,X_test
def feat_onehot(data,cols):
    """
    data: DataFrame类型数据
    cols: list类型数据 data中字段名列表
    """
    m=to_categorical(data[cols[0]],4)
    flag=0
    for col in cols[1:]:
        if flag%2<1:
            m=np.hstack((m,to_categorical(data[col],13)))
            flag=flag+1
        else:
            m=np.hstack((m,to_categorical(data[col],4)))
            flag=flag+1
    return m

def gather_data(data):
    """全排列"""
    newdata=pd.DataFrame()
    col_dict={'ziduan1':'num1','ziduan2':'num2','ziduan3':'num3','ziduan4':'num4','ziduan5':'num5'}
    col_list=list(itertools.permutations(list(col_dict.keys()),5))
    for i in col_list:
        col=[]
        for j in i:
            col=col+[j,col_dict.get(j)]
        if 'label' in list(data.columns): #data有标签情况
            m=data[col+['label']]
        else:                             #data无标签情况
            m=data[col]
        m.columns=data.columns
        newdata=pd.concat([newdata,m])
    return newdata

def model1(x,y):
    model = Sequential()  
    model.add(Dense(1000, input_shape=(x.shape[1],),activation='relu')) 
    model.add(Dense(30,activation='tanh'))  
    model.add(BatchNormalization())
    model.add(Dense(10, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(10, activation='linear'))
    model.add(BatchNormalization())
    model.add(Dense(10, activation='softmax')) #输出文本属于10种类别中每个类别的概率  
    
# 优化器我这里用了adadelta，也可以使用其他方法  
    model.compile(loss='categorical_crossentropy',  
              optimizer=Adam(lr=0.001, epsilon=1e-07, decay=0.0),
             # 'rmsprop',
             # "adam",
              metrics=['accuracy'])
    model.summary()
    model.fit(x, y, epochs=15, batch_size=360, #validation_data=(val,to_categorical(val_y))
    )
    return model
def model2(x,y):
    model = Sequential()  
    model.add(Dense(1000, input_shape=(x.shape[1],),activation='relu')) 
    model.add(Dense(30,activation='tanh'))  
    model.add(BatchNormalization())
    model.add(Dense(10, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(10, activation='linear'))
    model.add(BatchNormalization())
    model.add(Dense(10, activation='softmax')) #输出文本属于10种类别中每个类别的概率  
    
# 优化器我这里用了adadelta，也可以使用其他方法  
    model.compile(loss='categorical_crossentropy',  
              optimizer=#Adam(lr=0.001, epsilon=1e-07, decay=0.0),
              'rmsprop',
             # "adam",
              metrics=['accuracy'])
    model.summary()
    model.fit(x, y, epochs=20, batch_size=256, #validation_data=(val,to_categorical(val_y))
    )
    return model
def pre1(x,z,test,late_nn_model):
    if np.argsort(x)[-1]==0:
        if np.max(x)>0.999:
            return np.argsort(x)[-1]
        else:
            return predict_pre_sample(test[z.index(x):z.index(x)+1],late_nn_model)
    elif np.argsort(x)[-1]==9:
        if np.max(x)>0.999:
            return np.argsort(x)[-1]
        else:
            return predict_pre_sample(test[z.index(x):z.index(x)+1],late_nn_model)
    elif np.argsort(x)[-1]==5:
        if np.max(x)>0.999:
            return np.argsort(x)[-1]
        else:
            return predict_pre_sample(test[z.index(x):z.index(x)+1],late_nn_model)
    elif np.argsort(x)[-1]==3:
            return predict_pre_sample(test[z.index(x):z.index(x)+1],late_nn_model)
    
    elif np.argsort(x)[-1]==2:
        if np.max(x)>0.999:
            return np.argsort(x)[-1]
        else:
            return predict_pre_sample(test[z.index(x):z.index(x)+1],late_nn_model)
    
    elif np.argsort(x)[-1]==1:
            return predict_pre_sample(test[z.index(x):z.index(x)+1],late_nn_model)
    else:
        return np.argsort(x)[-1]
def stacking1(train_x,train_y,test):
    """做5折和10折stacking特征"""
    if not os.path.exists('stack_model1'): #stack模型路径
        os.makedirs('stack_model1')
    else:
        print('stack_model1目录已存在！\n')
    if not os.path.exists('stack_model1/5_folds_stack_model0.h5'):
        #for n_folds in [5,10]:
        n_folds=5
        print(str(n_folds)+'_folds_stacking')
        stack_train = np.zeros((len(train_y), 10))
        stack_test = np.zeros((test.shape[0], 10))
        score_va = 0
        train_y=train_y.astype(int)
        for i, (tr, va) in enumerate(StratifiedKFold(train_y, n_folds=n_folds, random_state=2018)):
            print('stack:%d/%d' % ((i + 1), n_folds))
            model=model1(train_x.ix[tr],to_categorical(train_y[tr]))
            model.save('stack_model1/'+str(n_folds)+'_folds_stack_model'+str(i)+'.h5') ##保存模型，下次可以直接调用
            score_va = model.predict(train_x.ix[va])
            score_te = model.predict(test)
            stack_train[va] += score_va
            stack_test += score_te
        stack_test /= n_folds
        df_stack_train = pd.DataFrame()
        df_stack_test = pd.DataFrame()
        for i in range(stack_test.shape[1]):
            df_stack_train['nnmodel1_'+str(n_folds)+'_classfiy_{}'.format(i)] = stack_train[:, i]
            df_stack_test['nnmodel1_'+str(n_folds)+'_classfiy_{}'.format(i)] = stack_test[:, i]
        df_stack_train.to_csv('nnmodel1_'+str(n_folds)+'folds_train_feat.csv', index=None, encoding='utf8')
        df_stack_test.to_csv('nnmodel1_'+str(n_folds)+'folds_test_feat.csv', index=None, encoding='utf8')
        print(str(n_folds)+'_folds_stacking特征已保存\n')
        return stack_test
            
            
def stacking2(train_x,train_y,test):
    """做5折和10折stacking特征"""
    if not os.path.exists('stack_model2'): #stack模型路径
        os.makedirs('stack_model2')
    else:
        print('stack_model2目录已存在！\n')
    if not os.path.exists('stack_model2/5_folds_stack_model0.h5'):
        #for n_folds in [5,10]:
        n_folds=5
        print(str(n_folds)+'_folds_stacking')
        stack_train = np.zeros((len(train_y), 10))
        stack_test = np.zeros((test.shape[0], 10))
        score_va = 0
        train_y=train_y.astype(int)
        for i, (tr, va) in enumerate(StratifiedKFold(train_y, n_folds=n_folds, random_state=2018)):
            print('stack:%d/%d' % ((i + 1), n_folds))
            model=model2(train_x.ix[tr],to_categorical(train_y[tr]))
            model.save('stack_model2/'+str(n_folds)+'_folds_stack_model'+str(i)+'.h5') ##保存模型，下次可以直接调用
            score_va = model.predict(train_x.ix[va])
            score_te = model.predict(test)
            stack_train[va] += score_va
            stack_test += score_te
        stack_test /= n_folds
        df_stack_train = pd.DataFrame()
        df_stack_test = pd.DataFrame()
        for i in range(stack_test.shape[1]):
            df_stack_train['nnmodel2_'+str(n_folds)+'_classfiy_{}'.format(i)] = stack_train[:, i]
            df_stack_test['nnmodel2_'+str(n_folds)+'_classfiy_{}'.format(i)] = stack_test[:, i]
        df_stack_train.to_csv('nnmodel2_'+str(n_folds)+'folds_train_feat.csv', index=None, encoding='utf8')
        df_stack_test.to_csv('nnmodel2_'+str(n_folds)+'folds_test_feat.csv', index=None, encoding='utf8')
        print(str(n_folds)+'_folds_stacking特征已保存\n')
        return stack_test
            

def processing(train_path,test_path):
    """数据的预处理工作"""
    start_time=time.time()
    
    #读入数据：
    train = pd.read_csv(train_path,header=None) #决赛训练数据路径
    training=pd.read_csv('training.csv',header=None) #预赛训练数据
    test = pd.read_csv(test_path,header=None) #测试数据路
    print('读入数据所需时间：',time.time()-start_time)
    
    #添加字段名，方便后面处理
    train.columns=['ziduan1','num1','ziduan2','num2','ziduan3','num3','ziduan4','num4','ziduan5','num5','label']
    test.columns=['ziduan1','num1','ziduan2','num2','ziduan3','num3','ziduan4','num4','ziduan5','num5']
    training.columns=list(train.columns)
    training['label']=training['label'].apply(lambda x:label_to_new(x))
    
    #决赛数据和预赛训练数据结合 ：
    train=pd.concat([train,training]).reset_index(drop=True)
    #字符转换成数字：
    for col in test.columns:
        train[col]=train[col].apply(str_to_int)
        test[col]=test[col].apply(str_to_int)
    
    ###进行全排列
    train=gather_data(train)

    ##打乱顺序,并删除重复样本
    train=shuffle(train)
    train=train.drop_duplicates().reset_index(drop=True)
    
    #保存标签，以备往后方便使用
    #label=train['label']
    train[['label']].to_csv('label.csv',index=None)
    #label=pd.read_csv('label.csv')
    
   # train['sum']=train['num1']+train['num2']+train['num3']+train['num4']+train['num5']
   # test['sum']=test['num1']+test['num2']+test['num3']+test['num4']+test['num5']
   # train['sum']=train.drop(['ziduan{}'.format(i) for i in range(1,6)]+['label'],axis=1).sum(axis=1)
   # test['sum']=test.drop(['ziduan{}'.format(i) for i in range(1,6)],axis=1).sum(axis=1)
    ###保存训练、测试数据：
    train.to_csv('train.csv',index=None)
    test.to_csv('test.csv',index=None)

    
    
    """
    zz=train[train.label==5]
    z=training[training.label==8]
    zzz=pd.concat([z.drop(['label'],axis=1),zz.drop(['label'],axis=1)]).reset_index(drop=True)
    zzzz=zzz.drop_duplicates().reset_index(drop=True)
    """
def predict_pre_sample(test,model):
    """
    逐条样本预测
    test:  一条待测试样本
    model: 已训练好的模型
    """
    sample=gather_data(test).reset_index(drop=True) #对该条样本进行全排列，返回120条样本
    result=model.predict(sample) #预测，返回120个结果
    #m=sorted(Counter(np.argmax(result,axis=1)).items(), key=lambda d: d[1])[-1][0] #取120个结果中，类别数最多的类别，作为最终类别
    return np.argmax(result[np.argmax(result.max(axis=1))]) #取120个结果中，概率最大的类别，作为最终类别
def by_cossim(x,z,test,train):
    """计算相似余弦度"""
    from numpy import linalg as la
    XY=(train.drop(['label'],axis=1)*np.tile(test.iloc[z.index(x)],(len(train),1))).sum(axis=1)
    XX=la.norm(train.drop(['label'],axis=1),axis=1)
    YY=la.norm(np.tile(test.iloc[z.index(x)],(len(train),1)),axis=1)
    COSSIM=(0.5+0.5*(XY/(XX*YY))).tolist()
    return train.loc[COSSIM.index(np.max(COSSIM)),'label']
    