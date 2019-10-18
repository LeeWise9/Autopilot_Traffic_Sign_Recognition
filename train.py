# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 14:25:17 2019
@author: lenovo
"""
import cv2
import pickle
import numpy as np
import help_func as h
import matplotlib.pyplot as plt
from keras import utils
from keras.models import Model,load_model
from keras.layers import Flatten,Conv2D,MaxPooling2D,Input,Dense,Dropout
from keras.layers.merge import concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

np.random.seed(0)
# 图像预处理
def preprocess_features(X):
    # RGB彩色图-->YUV亮度[:,:,0]
    X = np.array([np.expand_dims(cv2.cvtColor(rgb_img,cv2.COLOR_RGB2YUV)[:,:,0],2)for rgb_img in X])
    #直方图均衡化
    X = np.array([np.expand_dims(cv2.equalizeHist(np.uint8(img)),2)for img in X])
    X = np.float32(X)
    mean_img = np.mean(X, axis=0)
    std_img = (np.std(X, axis=0) + np.finfo('float32' ).eps)  # 为了使除法分母不为0，添加一个极小值
    X -= mean_img   # 图片去均值
    X /= std_img    # 标准化
    return X

def show_samples_from_generator(image_datagen, X_train, y_train):
    # 对图片做变换处理
    img_rgb = X_train[0]
    # 画图
    plt.figure(figsize=(1, 1))
    plt.imshow(img_rgb)
    plt.title('Example of RGB image (class = {})'.format(y_train[0]))
    plt.show()
    rows, cols = 4, 10
    fig, ax_array = plt.subplots(rows, cols)
    for ax in ax_array.ravel():
        augmented_img, _ = image_datagen.flow(np.expand_dims(img_rgb, 0), y_train[0:1]).next()
        ax.imshow(np.uint8(np.squeeze(augmented_img)))
    plt.setp([a.get_xticklabels() for a in ax_array.ravel()], visible=False)
    plt.setp([a.get_yticklabels() for a in ax_array.ravel()], visible=False)
    plt.suptitle('Random examples of data augmentation (starting from the previous image)')
    plt.show()
    
def get_image_generator():
    # 图像增强
    image_datagen = ImageDataGenerator(rotation_range=15.,      # 旋转±15°
                                       width_shift_range=0.1,   # 横向平移
                                       height_shift_range=0.1,  # 纵向平移
                                       zoom_range=0.2)          # 缩放
    return image_datagen
    
def show_image_generator_effect():
    X_train, y_train = h.load_traffic_sign_data('./traffic-signs-data/train.p')
    n_train = X_train.shape[0]
    image_shape = X_train[0].shape
    n_classes = np.unique(y_train).shape[0]
    print("Number of training examples =", n_train)
    print("Image data shape  =", image_shape)
    print("Number of classes =", n_classes)
    image_generator = get_image_generator()
    show_samples_from_generator(image_generator,X_train,y_train)
    
 
def build_model(dropout_rate):
    input_shape = (32, 32, 1)
    input_ = Input(shape=input_shape)
    cv2d_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(input_)
    pool_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(cv2d_1)
    dropout_1 = Dropout(dropout_rate)(pool_1)
    flatten_1 = Flatten()(dropout_1)
    
    cv2d_2 = Conv2D(64, (3,3), padding='same', activation='relu')(dropout_1)
    pool_2 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(cv2d_2)
    cv2d_3 = Conv2D(64, (3,3), padding='same', activation='relu')(pool_2)
    dropout_2 = Dropout(dropout_rate)(cv2d_3)
    flatten_2 = Flatten()(dropout_2)
    
    concat_1 = concatenate([flatten_1, flatten_2])
    dense_1 = Dense(64, activation='relu')(concat_1)
    output = Dense(43, activation='softmax')(dense_1)
    model = Model(inputs=input_, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def training_setting(model, image_datagen, x_train, y_train, x_validation, y_validation):
    filepath = "./Models/weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='auto')
    callbacks_list = [checkpoint]    # 设置回调函数函数--检查点
    image_datagen.fit(x_train)
    history = model.fit_generator(image_datagen.flow(x_train, y_train, batch_size=512),
                                  steps_per_epoch=5000, epochs=5,
                                  validation_data=(x_validation, y_validation),
                                  callbacks=callbacks_list,verbose=1)
    print(history.history.keys())
    plt.plot(history.history['acc'],label='acc')
    plt.plot(history.history['val_acc'],label='val_acc')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()
    
    plt.plot(history.history['loss'],label='train')
    plt.plot(history.history['val_loss'],label='test')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()
    with open('/trainHistoryDict.p', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)     # 保存训练结果（字典）到文件
    return history

def train_model():
    # 加载数据
    X_train, y_train = h.load_traffic_sign_data('./traffic-signs-data/train.p')
    n_train = X_train.shape[0]               # 数据条数
    n_classes = np.unique(y_train).shape[0]  # 类别数
    image_shape = X_train[0].shape           # 图片形状（大小）
    print("Number of training examples =", n_train)
    print("Number of classes =", n_classes)
    print("Image data shape  =", image_shape)

    X_train_norm = preprocess_features(X_train)         # 对图片预处理
    y_train = utils.to_categorical(y_train, n_classes)  # 标签转为独热编码

    VAL_RATIO = 0.2  # 划分数据集
    X_train_norm, X_val_norm, y_train, y_val = train_test_split(X_train_norm, y_train,
                                                                test_size=VAL_RATIO,
                                                                random_state=0)
    model = build_model(dropout_rate = 0.0)
    image_generator = get_image_generator()  # 图像增强函数
    training_setting(model, image_generator, X_train_norm, y_train, X_val_norm, y_val)


if __name__ == "__main__":
    show_image_generator_effect()
    train_model()
    
