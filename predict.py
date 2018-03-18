from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import os
import cv2
import random
import numpy as np
import pandas as pd
from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras.optimizers import *
from keras.applications import *
from keras.regularizers import *
from keras.applications.inception_resnet_v2 import InceptionResNetV2,preprocess_input
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

testPath='/dl/data/fashionAI_attributes_test_a_20180222/rank/'
model_type={'collar':'collar_design_labels', 'neckline':'neckline_design_labels','skirt': 'skirt_length_labels',
           'sleeve':'sleeve_length_labels', 'neck':'neck_design_labels','coat': 'coat_length_labels', 'lapel':'lapel_design_labels',
           'pant':'pant_length_labels'}
lable_count={'coat':8,'collar':5,'lapel':5,
             'neck':5,'neckline':10,'pant':6,
             'skirt':6,'sleeve':9}
df_test = pd.read_csv(testPath+'Tests/question.csv', header=None)
df_test.columns = ['image_id', 'class', 'x']
del df_test['x']
print(df_test.head())
for key in model_type:
    width = 299
    cur_class = model_type[key]

    cnn_model = InceptionResNetV2(include_top=False, input_shape=(width, width, 3), weights='imagenet',pooling='avg')
    inputs = Input((width, width, 3))

    x = inputs
    x = Lambda(preprocess_input)(x)
    x = cnn_model(x)
    x = Dropout(0.5)(x)
    x = Dense(lable_count[key], activation='softmax', name='softmax')(x)
    model = Model(inputs, x)

    model.load_weights('./models/{0}.best.h5'.format(key))
    df_load = df_test[(df_test['class'] == cur_class)].copy()
    df_load.reset_index(inplace=True)
    del df_load['index']

    print('{0}: {1}'.format(cur_class, len(df_load)))
    print(df_load.head())

    n = len(df_load)
    X_test = np.zeros((n, width, width, 3), dtype=np.uint8)

    for i in tqdm(range(n)):
        image=cv2.imread(testPath + '{0}'.format(df_load['image_id'][i]))
        img_height = image.shape[0]
        img_width = image.shape[1]
        crop_height = img_height*0.875
        crop_width = img_width * 0.875
        image=image[int((img_height-crop_height)/2):int((img_height+crop_height)/2),int((img_width-crop_width)/2):int((img_width+crop_width)/2)]
        X_test[i] = cv2.resize(image, (width, width))
    test_np = model.predict(X_test, batch_size=32)
    print(test_np.shape)

    result = []

    for i, row in df_load.iterrows():
        tmp_list = test_np[i]
        tmp_result = ''
        for tmp_ret in tmp_list:
            tmp_result += '{:.4f};'.format(tmp_ret)

        result.append(tmp_result[:-1])

    df_load['result'] = result
    print(df_load.head())

    df_load.to_csv('./result/{0}.csv'.format(key), header=None, index=False)
    print(key,model_type[key])

data=pd.DataFrame()
l=0
for key in model_type:
    d=pd.read_csv('./result/{0}.csv'.format(key),header=None)
    l=l+len(d)
    d.columns= ['image_id', 'class', 'x']
    data=data.append(d)
data.to_csv('./result/result.csv', header=None, index=False)
print(l)