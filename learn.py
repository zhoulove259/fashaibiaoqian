from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import os
import cv2
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
trainPath='E:/AIdata/fashionaibiaoqian/fashionAI_attributes_train_20180222/base/'
testPath='E:/AIdata/fashionaibiaoqian/fashionAI_attributes_test_a_20180222/rank/'
df_train = pd.read_csv(trainPath+'Annotations/label.csv', header=None)
df_train.columns = ['image_id', 'class', 'label']
print(df_train.head())
df_test = pd.read_csv(testPath+'Tests/test_a.csv', header=None)
df_test.columns = ['image_id', 'class', 'x']
del df_test['x']
print(df_test.head())
lable_count={'collar_design_labels':[3,6], 'neckline_design_labels':0, 'skirt_length_labels':0,
           'sleeve_length_labels':0, 'neck_design_labels':0, 'coat_length_labels':0, 'lapel_design_labels':0,
           'pant_length_labels':0}
classes = ['collar_design_labels', 'neckline_design_labels', 'skirt_length_labels',
           'sleeve_length_labels', 'neck_design_labels', 'coat_length_labels', 'lapel_design_labels',
           'pant_length_labels']
len(classes)
for i in range(len(classes)):
    cur_class = classes[i]
    df_load = df_train[(df_train['class'] == cur_class)].copy().sample(100,random_state=0)
    df_load.reset_index(inplace=True)
    del df_load['index']

    print('{0}: {1}'.format(cur_class, len(df_load)))
    print(df_load.head())
    print(df_load[(df_load.index == 2)])

    n = len(df_load)
    n_class = len(df_load['label'][0])
    width = 299
    X = np.zeros((n, width, width, 3), dtype=np.uint8)
    y = np.zeros((n, n_class), dtype=np.uint8)
    X_temp=X.copy()
    y_temp=y.copy()
    for i in tqdm(range(n)):
        tmp_label = df_load['label'][i]
        if len(tmp_label) > n_class:
            print(df_load['image_id'][i])
        image=cv2.imread(trainPath + '{0}'.format(df_load['image_id'][i]))
        img_height = image.shape[0]
        img_width = image.shape[1]
        crop_height = img_height*0.875
        crop_width = img_width * 0.875
        image=image[int((img_height-crop_height)/2):int((img_height+crop_height)/2),int((img_width-crop_width)/2):int((img_width+crop_width)/2)]
        X[i] = cv2.resize(image, (width, width))
        X_temp[i] = cv2.flip(X[i], 1)
        y[i][tmp_label.find('y')] = 1
        y_temp[i]=y[i]

    X = np.append(X, X_temp, axis=0)
    y = np.append(y, y_temp, axis=0)
    plt.figure(figsize=(12, 7))
    for i in range(8):
        random_index = random.randint(0, n - 1)
        plt.subplot(2, 4, i + 1)
        plt.imshow(X[random_index][:, :, ::-1])
        plt.title(y[random_index])

    cnn_model = InceptionResNetV2(include_top=False, input_shape=(width, width, 3), weights='imagenet',pooling='avg')
    inputs = Input((width, width, 3))

    x = inputs
    x = Lambda(preprocess_input)(x)
    x = cnn_model(x)
    x = Dropout(0.5)(x)
    x = Dense(n_class, activation='softmax', name='softmax')(x)
    model = Model(inputs, x)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.12, random_state=42)
    print(X_train.shape, y_train.shape)

    adam = Adam(lr=0.001)
    prefix_cls = cur_class.split('_')[0]

    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    checkpointer = ModelCheckpoint(filepath='./models/{0}.best.h5'.format(prefix_cls), verbose=1,
                                   save_best_only=True)

    h = model.fit(X_train, y_train, batch_size=3, epochs=80,
                  callbacks=[EarlyStopping(patience=0), checkpointer],
                  shuffle=True,
                  validation_split=0.1)
    model.train_on_batch()
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')

    plt.subplot(1, 2, 2)
    plt.plot(h.history['acc'])
    plt.plot(h.history['val_acc'])
    plt.legend(['acc', 'val_acc'])
    plt.ylabel('acc')
    plt.xlabel('epoch')

    print(model.evaluate(X_train, y_train, batch_size=3))
    print(model.evaluate(X_valid, y_valid, batch_size=3))

    df_load = df_test[(df_test['class'] == cur_class)].copy().sample(10,random_state=0)
    df_load.reset_index(inplace=True)
    del df_load['index']

    print('{0}: {1}'.format(cur_class, len(df_load)))
    print(df_load.head())

    n = len(df_load)
    X_test = np.zeros((n, width, width, 3), dtype=np.uint8)

    for i in tqdm(range(n)):
        X_test[i] = cv2.resize(cv2.imread(testPath + '{0}'.format(df_load['image_id'][i])), (width, width))
    test_np = model.predict(X_test, batch_size=3)
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

    df_load.to_csv('./result/{0}_0307a.csv'.format(prefix_cls), header=None, index=False)
    print(prefix_cls)