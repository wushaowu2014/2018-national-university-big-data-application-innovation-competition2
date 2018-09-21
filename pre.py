# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 00:00:53 2018

@author: wushaowu
"""
from nn_model import nn_model1,nn_model2
from utils import predict_pre_sample,model2,gather_data,feat_onehot,get_newdata
from keras.models import load_model
import numpy as np  
import pandas as pd  
import os
os.environ['KERAS_BACKEND']='tensorflow'#'theano'
from keras.utils.np_utils import to_categorical
from collections import Counter
seed=7
np.random.seed(seed)
from keras.layers import Dense,Dropout,Convolution1D,Flatten,Conv1D, BatchNormalization,PReLU
from keras.layers import Input, Embedding,concatenate,add,average,multiply,maximum
from keras.models import Model
from keras.optimizers import Adam
def nn_model3(train_x,train_y):
    """建立第二个五层的神经网络"""
    inputs=Input(shape=(train_x.shape[1],))
    x1 = Dense(40, activation='relu')(inputs)
    x1=BatchNormalization()(x1)
    x2 = Dense(40, activation='tanh')(inputs)
    x2=BatchNormalization()(x2)

    x=maximum([x1,x2])
    x = Dense(20, activation='sigmoid')(x)

    predictions = Dense(10, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=predictions)
    model.summary()
    model.compile(optimizer=#Adam(lr=0.001, epsilon=1e-09, decay=0.0),
                  'rmsprop',
                #  "adam",
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    model.fit(train_x, train_y,epochs=5, batch_size=500, validation_split=0.0)
    return model


def pre1(x,z,test,late_nn_model):
    if np.argsort(x)[-1]==0:
        if np.max(x)>0.90:
            return np.argsort(x)[-1]
        else:
            return predict_pre_sample(test[z.index(x):z.index(x)+1],late_nn_model)
    elif np.argsort(x)[-1]==9:
        if np.max(x)>0.9999999:
            return np.argsort(x)[-1]
        else:
            return predict_pre_sample(test[z.index(x):z.index(x)+1],late_nn_model)
    elif np.argsort(x)[-1]==5:
        if np.max(x)>0.9999999:
            return np.argsort(x)[-1]
        else:
            return predict_pre_sample(test[z.index(x):z.index(x)+1],late_nn_model)
    elif np.argsort(x)[-1]==3:
            return predict_pre_sample(test[z.index(x):z.index(x)+1],late_nn_model)
    
    elif np.argsort(x)[-1]==2:
        if np.max(x)>0.9999999:
            return np.argsort(x)[-1]
        else:
            return predict_pre_sample(test[z.index(x):z.index(x)+1],late_nn_model)
    
    elif np.argsort(x)[-1]==1:
            return predict_pre_sample(test[z.index(x):z.index(x)+1],late_nn_model)
    else:
        return np.argsort(x)[-1]

if __name__ == '__main__':
    """程序入口"""
    
    #读入特征数据
    train=pd.read_csv('train.csv')
    test=pd.read_csv('test.csv')
    label=train['label']
    
    nnmodel1_5folds_train_feat= pd.read_csv('nnmodel1_5folds_train_feat.csv')
    nnmodel1_5folds_test_feat= pd.read_csv('nnmodel1_5folds_test_feat.csv')
    
    nnmodel2_5folds_train_feat= pd.read_csv('nnmodel2_5folds_train_feat.csv')
    nnmodel2_5folds_test_feat= pd.read_csv('nnmodel2_5folds_test_feat.csv')

    #特征融合
    new_tr=pd.concat([nnmodel1_5folds_train_feat,\
                      nnmodel2_5folds_train_feat,\
                     pd.DataFrame(feat_onehot(train.drop(['label'],axis=1),test.columns)),\
                   #  get_newdata(train)
                      ],axis=1)
    new_te=pd.concat([nnmodel1_5folds_test_feat,\
                      nnmodel2_5folds_test_feat,\
                     pd.DataFrame(feat_onehot(test,test.columns)),\
                   #  get_newdata(test)
                      ],axis=1)
    
    if not os.path.exists('nn_model3_20180826.h5'):
        #调用第一个神经网络模型
        print('第一个模型运行中...')
        model=nn_model3(new_tr,to_categorical(label))
      #  model1.summary()
        results=model.predict(new_te) ##预测
        print(Counter(np.argmax(results,axis=1)))
        model.save('nn_model3_20180826.h5') ##保存模型，下次可以直接调用
        
    else:
        #加载之前保存的模型
        model=load_model('nn_model3_20180826.h5')
        #预测
        print('第一个模型运行中...\n')
        model.summary()
        results=model.predict(new_te)
        print('第一个模型的结果统计',Counter(np.argmax(results,axis=1)))
    #    np.savetxt("dsjyycxds_preliminary1.txt",np.argmax(pre1,axis=1).astype(int),fmt="%d")
    
    ##后处理
    #提前载入stacking模型：
    model1_0=load_model('stack_model1/5_folds_stack_model0.h5')
    model1_1=load_model('stack_model1/5_folds_stack_model1.h5')
    model1_2=load_model('stack_model1/5_folds_stack_model2.h5')
    model1_3=load_model('stack_model1/5_folds_stack_model3.h5')
    model1_4=load_model('stack_model1/5_folds_stack_model4.h5')
    model2_0=load_model('stack_model2/5_folds_stack_model0.h5')
    model2_1=load_model('stack_model2/5_folds_stack_model1.h5')
    model2_2=load_model('stack_model2/5_folds_stack_model2.h5')
    model2_3=load_model('stack_model2/5_folds_stack_model3.h5')
    model2_4=load_model('stack_model2/5_folds_stack_model4.h5')
    
    rep=results.tolist() #转化成list
    final_results=[]
    for i in range(len(results)):
        if np.max(results[i])<0.95:
            initial_sample=gather_data(test[i:i+1]).reset_index(drop=True)
            new_sample=np.hstack((
                   0.2*(model1_0.predict(initial_sample)+\
                       model1_1.predict(initial_sample)+\
                       model1_2.predict(initial_sample)+\
                       model1_3.predict(initial_sample)+\
                       model1_4.predict(initial_sample)),\
                   0.2*(model2_0.predict(initial_sample)+\
                       model2_1.predict(initial_sample)+\
                       model2_2.predict(initial_sample)+\
                       model2_3.predict(initial_sample)+\
                       model2_4.predict(initial_sample)),\
                       feat_onehot(initial_sample,test.columns)
                      ))
            res=model.predict(new_sample)
            final_results.append(
                    #np.argmax(res[np.argmax(res.max(axis=1))])
                    sorted(Counter(np.argmax(res,axis=1)).items(), key=lambda d: d[1])[-1][0]
                    )
        else:
            final_results.append(np.argsort(results[i])[-1])
        
    np.savetxt("dsjyycxds_semifinal.txt",np.array(final_results).astype(int),fmt="%d") 
