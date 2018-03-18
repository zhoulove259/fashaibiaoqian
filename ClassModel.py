from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import numpy as np
import pandas as pd
import keras.backend as K
from keras import layers
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import Dense,Dropout
from keras.optimizers import Adam,SGD,Nadam
from keras.layers import Input, Embedding, Reshape, Add
from keras.layers import Flatten, merge, Lambda
from keras.models import Model
from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import OneHotEncoder, StandardScaler,LabelEncoder
from sklearn.metrics import accuracy_score,log_loss
from sklearn.model_selection import train_test_split
from keras.applications.inception_resnet_v2 import InceptionResNetV2,preprocess_input

trainPath='E:/AIdata/fashionaibiaoqian/fashionAI_attributes_train_20180222/base/'
testPath='E:/AIdata/fashionaibiaoqian/fashionAI_attributes_test_a_20180222/rank/'
df_train = pd.read_csv(trainPath+'Annotations/label.csv', header=None)
df_train.columns = ['image_id', 'class', 'label']
df_train.head()
width=299
lable_count={'coat_length':8,'collar_design':5,'lapel_design':5,
             'neck_design':5,'neckline_design':10,'pant_length':6,
             'skirt_length':6,'sleeve_length':9}
base_model=InceptionResNetV2(weights='imagenet',input_shape=(width,width,3),include_top=False,pooling='avg')
input_tensor=Input((width,width,3))
x=input_tensor
x=Lambda(preprocess_input)(x)
x=base_model(x)
x=Dropout(0.5)(x)
x=[Dense(count,activation='softmax',name=name)(x) for name,count in lable_count.items()]
model=Model(input_tensor,x)
print(model.summary())
plot_model(model, to_file='model.png', show_shapes=True)
model.compile(optimizer=Adam(0.001),loss='categorical_crossentropy',metrics=['acc'])
n=len(df_train)
y=[np.zeros((n,lable_count[x]))for x in lable_count.keys()]
for key in lable_count.keys():
    df=df_train[df_train['class']==key+'_labels']
    index=df.index.tolist()
    print(np.min(index),np.max(index))
