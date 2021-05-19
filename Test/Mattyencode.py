from __future__ import print_function
import os
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, Flatten, Dense, LSTM, TimeDistributed, \
    Bidirectional
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import multiply, Permute, Reshape
from tensorflow.keras.models import load_model
import tensorflow.keras.optimizers as optimizers
from sklearn import metrics
import numpy as np
# import tensorflow.config as config
from tensorflow.keras.utils import multi_gpu_model
from sklearn.metrics import confusion_matrix
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn import manifold

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# tf.debugging.set_log_device_placement(True)
# gpus = tf.config.experimental.list_logical_devices('GPU')
# if gpus:
#    c=[]
#    for gpu in gpus:
#        with tf.device(gpu.name):
#            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
#            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
#            c.append(tf.matmul(a, b))
#        with tf.device('/CPU:0'):
#            print(tf.add_n(c))

modelis = "RAI"

if (modelis == "RDI"):
    h5_dir = 'D:\\real-time-radar\\RDIdata\\'
elif(modelis == "RAI"):
    h5_dir = 'D:\\real-time-radar\\RAIdata\\'
# h5_dir = 'D:/pycharm_progject/yenli/data/cross-scene/3t4rRAI/'
# model_dir = '/home/pstudent/DeepSoli Models/gesture_data/Range-Doppler_fair comparison/final data/3t4r_RDI/'
# vali_dir='/home/pstudent/DeepSoli Models/gesture_data/Range-Doppler_fair comparison/final data/lab210_otherpoints_NewConfig_15dBoff/'
# blank_dir='/home/pstudent/DeepSoli Models/gesture_data/Range-Doppler_fair comparison/3t4r_H104_blank_NewConfig/'
# h5_dir = '/home/pstudent/DeepSoli Models/gesture_data/Range-Angle_fair comparison/3t4r/scene_mixed/'
# h5_dir = '/home/pstudent/DeepSoli Models/gesture_data/transformto_h5/'
# output_dir = h5_dir + 'whole set as train/'

x_train = np.load(h5_dir + 'x_train.npy')
y_train = np.load(h5_dir + 'y_train.npy')

# x_test_blank = np.load(blank_dir + 'x_blank.npy')

x_test = np.load(h5_dir + 'x_test.npy')
y_test = np.load(h5_dir + 'y_test.npy')
# print (x_test.shape)
# print (y_test.shape)

x_train = np.transpose(x_train, (0, 1, 3, 4, 2))
x_test = np.transpose(x_test, (0, 1, 3, 4, 2))

if (modelis == "RAI"):
    x_train = x_train[:, :, :, :, 0]
    x_test = x_test[:, :, :, :, 0]

# x_train = x_train[:,:,:,:,0]
# x_train = x_train[..., np.newaxis]
# x_test = x_test[:,:,:,:,0]
# x_test = x_test[..., np.newaxis]

# ============================================================
# x_train_new = np.load(h5_dir + 'x_train.npy')
# y_train_new = np.load(h5_dir + 'y_train.npy')

# x_test_blank = np.load(blank_dir + 'x_blank.npy')

# x_test_new = np.load(h5_dir + 'x_test.npy')
# y_test_new = np.load(h5_dir + 'y_test.npy')
# print (x_test.shape)
# print (y_test.shape)

# x_train = np.reshape(x_train, [1440, 64, 32, 32, -1])
# x_test = np.reshape(x_test, [1440, 64, 32, 32, -1])

x_train_new = x_train
y_train_new = y_train

x_test_new = x_test
y_test_new = y_test
#
# x_train_new = x_train_new[:,:,:,:,0]
# x_train_new = x_train_new[..., np.newaxis]
# x_test_new = x_test_new[:,:,:,:,0]
# x_test_new = x_test_new[..., np.newaxis]
# ============================================================

# temp_train_x = np.zeros((1440,64,32,32,1))
# temp_train_y = np.zeros((1440,64,12))
# temp_test_x = np.zeros((1440,64,32,32,1))
# temp_test_y = np.zeros((1440,64,12))

# x_train_new = x_train[np.newaxis,0,...]
# y_train_new = y_train[np.newaxis,0,...]
# x_test_new = x_test[np.newaxis,8,...]
# y_test_new = y_test[np.newaxis,8,...]

# for i in range(0, 1440):
##    if i % 10 == 9 or i % 10 == 8 or i % 10 == 7 or i % 10 == 6 or i % 10 == 5:
##    if i % 10 == 9 or i % 10 == 8 or i % 10 == 7 or i % 10 == 6:
##    if i % 10 == 9 or i % 10 == 8 or i % 10 == 7:
#    if i % 10 == 9 or i % 10 == 8:
##    if i % 10 == 9:
#        x_test_new[i,...] = x_train[i,...]
#        y_test_new[i,...] = y_train[i,...]
#        x_train_new[i,...] = x_test[i,...]
#        y_train_new[i,...] = y_test[i,...]
#
# for i in range(0, 1440):
#    if i % 10 == 9 or i % 10 == 8:
#        temp_test_x[i,...] = x_test[i,...]
#        temp_test_y[i,...] = y_test[i,...]

for i in range(0, 1440):  # train as 80%
    #    if i % 10 == 9 or i % 10 == 8 or i % 10 == 7 or i % 10 == 6 or i % 10 == 5:
    #    if i % 10 == 9 or i % 10 == 8 or i % 10 == 7 or i % 10 == 6:
    #    if i % 10 == 9 or i % 10 == 8 or i % 10 == 7:

    if i % 10 == 0 or i % 10 == 1 or i % 10 == 2 or i % 10 == 3 or i % 10 == 4:
        #    if i % 10 == 9:
        x_train_new[i, ...] = x_test[i, ...]
        y_train_new[i, ...] = y_test[i, ...]
#        x_train_new = np.delete(x_train_new, i, 0)
#        y_train_new = np.delete(y_train_new, i, 0)

for i in range(0, 1440):  # test as 20%
    #    if i % 10 == 0 or i % 10 == 1 or i % 10 == 2 or i % 10 == 3 or i % 10 == 4 or i % 10 == 5 or i % 10 == 6 or i % 10 == 7:
    if i % 10 == 5 or i % 10 == 6 or i % 10 == 7 or i % 10 == 8 or i % 10 == 9:
        x_test_new[i, ...] = x_train[i, ...]
        y_test_new[i, ...] = y_train[i, ...]
#        x_test_new = np.delete(x_test_new, i, 0)
#        y_test_new = np.delete(y_test_new, i, 0)

del x_train, y_train, x_test, y_test
# del temp_train_x, temp_train_y, temp_test_x, temp_test_y

# ===================================================================#
## 3t1r RDI
# x_train = np.concatenate((x_train[:,:,:,:,0, np.newaxis],
#                          x_train[:,:,:,:,4, np.newaxis],
#                          x_train[:,:,:,:,8, np.newaxis]), axis=4)
#
# x_test = np.concatenate((x_test[:,:,:,:,0, np.newaxis],
#                          x_test[:,:,:,:,4, np.newaxis],
#                          x_test[:,:,:,:,8, np.newaxis]), axis=4)
# ===================================================================#

# x_train = np.concatenate((x_train, x_test), axis=0)
# y_train = np.concatenate((y_train, y_test), axis=0)

# x_test = np.transpose(x_test, (0,1,3,4,2))

# x_test = np.load(vali_dir + 'x_test.npy')
# y_test = np.load(vali_dir + 'y_test.npy')
# x_test = np.load(h5_dir + 'x_test.npy')
# y_test = np.load(h5_dir + 'y_test.npy')
# x_test = np.transpose(x_test, (0,1,3,4,2))
# x_test = x_test[:,:,:,32:,:]
# x_train = (x_train > 0) * x_train
# x_test = (x_test > 0) * x_test
# x_test = x_test[:,:,:,32:,:]
# x_train = x_train/255
# x_test = x_test/255

# x_train = x_train*255
# x_test = x_test*255


# x_train = x_train/np.max(x_train)
# x_test = x_test/np.max(x_test)

# x_train = x_train*(-1)
# x_test = x_test*(-1)

# x_train=x_train[:,:,:,:,:]
# x_test=x_test[:,:,:,:,:]

# x_train = x_train[..., np.newaxis]
# x_test = x_test[..., np.newaxis]


# x_train = x_train/255
# x_test = x_test/255

# ==============================================================================

# ==============================================================================
## 3t1r RDI
# x_train_3t_1 = np.concatenate((x_train[:,:,:,:,0,np.newaxis], x_train[:,:,:,:,1,np.newaxis],
#                              x_train[:,:,:,:,2,np.newaxis], x_train[:,:,:,:,3,np.newaxis]), axis=4)
# x_test_3t_1 = np.concatenate((x_test[:,:,:,:,0,np.newaxis], x_test[:,:,:,:,1,np.newaxis],
#                              x_test[:,:,:,:,2,np.newaxis], x_test[:,:,:,:,3,np.newaxis]), axis=4)
# x_train = x_train_3t_1
# x_test = x_test_3t_1
#
# del x_train_3t_1, x_test_3t_1

## 1t2r RDI

# x_train_4r_1 = np.concatenate((x_train[:,:,:,:,0,np.newaxis], x_train[:,:,:,:,1,np.newaxis]
#                                ), axis=4)
#
# x_test_4r_1 = np.concatenate((x_test[:,:,:,:,0,np.newaxis], x_test[:,:,:,:,1,np.newaxis]
#                                ), axis=4)
#
# x_train = x_train_4r_1
# x_test = x_test_4r_1
# del x_train_4r_1, x_test_4r_1
# ==============================================================================

num_train = x_train_new.shape[0]
width_train = x_train_new.shape[2]
height_train = x_train_new.shape[3]
channel_train = x_train_new.shape[4]

x_train_mean_perChannel = np.zeros((num_train, 64, width_train, height_train, channel_train))
for i in range(0, channel_train):
    temp = np.mean(x_train_new[:, :, :, :, i])
    temp = np.repeat(temp, height_train, axis=0)
    temp_2 = temp[np.newaxis, ...]
    temp_2 = np.repeat(temp_2, width_train, axis=0)
    temp_2 = temp_2[np.newaxis, ...]
    temp_2 = np.repeat(temp_2, 64, axis=0)
    temp_3 = temp_2[np.newaxis, ...]
    temp_3 = np.repeat(temp_3, num_train, axis=0)
    x_train_mean_perChannel[:, :, :, :, i] = temp_3
    print(i)

x_train_std = np.zeros((1, channel_train))
for j in range(0, channel_train):
    x_train_std[:, j] = np.std(x_train_new[:, :, :, :, j])

x_train_new = (x_train_new - x_train_mean_perChannel) / x_train_std

################################################################
# x_test_new = (x_test_new - x_train_mean_perChannel[:np.size(x_test_new, 0),...])/x_train_std
x_test_new = (x_test_new - x_train_mean_perChannel) / x_train_std
################################################################
del x_train_mean_perChannel, x_train_std
del temp, temp_2, temp_3
##
###===================================================================================++#
##
#
# num_test = x_test.shape[0]
# width_test= x_test.shape[2]
# height_test = x_test.shape[3]
# channel_test = x_test.shape[4]
#
# x_test_mean_perChannel=np.zeros((num_test,64,width_test,height_test,channel_test))
# for i in range(0, channel_test):
#    temp = np.mean(x_test[:,:,:,:,i])
#    temp = np.repeat(temp, height_test, axis=0)
#    temp_2 = temp[np.newaxis,...]
#    temp_2 = np.repeat(temp_2, width_test, axis=0)
#    temp_2 = temp_2[np.newaxis,...]
#    temp_2 = np.repeat(temp_2, 64, axis=0)
#    temp_3 = temp_2[np.newaxis,...]
#    temp_3 = np.repeat(temp_3, num_test, axis=0)
#    x_test_mean_perChannel[:,:,:,:,i] = temp_3
#    print(i)
#
# x_test_std = np.zeros((1, channel_test))
# for j in range(0, channel_test):
#    x_test_std[:,j] = np.std(x_test[:,:,:,:,j])
#
# x_test = (x_test - x_test_mean_perChannel)/x_test_std
# del x_test_mean_perChannel, x_test_std
# del temp, temp_2, temp_3

# ==============================================================================
# x_train_subbed = np.zeros((x_train.shape))
# x_test_subbed = np.zeros((x_test.shape))
#
# for i in range(0, len(x_test)):
#    for j in range(0, 64):
#        for k in range(0, 11):
#            x_test_subbed[i,j,:,:,k] = x_test[i,j,:,:,k] - x_test[i,0,:,:,k]
#
# for i in range(0, len(x_train)):
#    for j in range(0, 64):
#        for k in range(0, 11):
#            x_train_subbed[i,j,:,:,k] = x_train[i,j,:,:,k] - x_train[i,0,:,:,k]
#
# x_test = x_test_subbed
# x_train = x_train_subbed
# del x_train_subbed, x_test_subbed
# ===============================================================================
# x_all = np.concatenate((x_train, x_test), axis=0)
#
# num_all = x_all.shape[0]
# width_all= x_all.shape[2]
# height_all = x_all.shape[3]
# channel_all = x_all.shape[4]
#
# x_all_mean_perChannel=np.zeros((num_all,64,width_all,height_all,channel_all))
# for i in range(0, channel_all):
#    temp = np.mean(x_all[:,:,:,:,i])
#    temp = np.repeat(temp, height_all, axis=0)
#    temp_2 = temp[np.newaxis,...]
#    temp_2 = np.repeat(temp_2, width_all, axis=0)
#    temp_2 = temp_2[np.newaxis,...]
#    temp_2 = np.repeat(temp_2, 64, axis=0)
#    temp_3 = temp_2[np.newaxis,...]
#    temp_3 = np.repeat(temp_3, num_all, axis=0)
#    x_all_mean_perChannel[:,:,:,:,i] = temp_3
#    print(i)
#
# x_all_std = np.zeros((1, channel_all))
# for j in range(0, channel_all):
#    x_all_std[:,j] = np.std(x_all[:,:,:,:,j])
#
# x_train = (x_train - x_all_mean_perChannel[:len(x_train),...])/x_all_std
# x_test = (x_test - x_all_mean_perChannel[:len(x_test),...])/x_all_std
#
# del x_all_mean_perChannel, x_all_std , x_all
# del temp, temp_2, temp_3
# ==============================================================================
# model = load_model(model_dir + 'model_0807NewConfig_bilstm_RDI_gaussian_32x32_batch12_Adam.h5')
# model.load_weights(model_dir + 'weights_3t4r_lstm_RangeDoppler_32x32_gaussed_batch12_ProgressSgd.h5', by_name = True)
# score = model.evaluate(x_test, y_test, batch_size=12)
# predictions = model.predict(x_test)
# ==============================================================================
# y_train = np.utils.to_categorical(y_train, 7)
# y_test = np.utils.to_categorical(y_test, 7)

# datagen = ImageDataGenerator(
#        featurewise_center=True,
#        featurewise_std_normalization=True,
#        zoom_range=0.2)

# datagen.fit(x_train)

# x_train = x_train[..., np.newaxis]
# x_test = x_test[..., np.newaxis]
# gpu_options = tf.GPUOptions(allow_growth=True)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#
# tf.keras.backend.set_session(sess)


# x_train = (x_train > 1) * x_train
# x_test = (x_test > 1) * x_test

# =============================================================================
epochs = 40
lr_power = 0.9
lr_base = 1e-2


def lr_scheduler(epoch):
    # def lr_scheduler(epoch, mode='progressive_drops'):
    '''if lr_dict.has_key(epoch):
        lr = lr_dict[epoch]
        print 'lr: %f' % lr'''

    #     if mode is 'power_decay':
    #         # original lr scheduler
    #         lr = lr_base * ((1 - float(epoch) / epochs) ** lr_power)
    #     if mode is 'exp_decay':
    #         # exponential decay
    #         lr = (float(lr_base) ** float(lr_power)) ** float(epoch + 1)
    #     # adam default lr
    #     if mode is 'adam':
    #         lr = 0.001
    #
    #     if mode is 'progressive_drops':keras.layers
    #         # drops as progression proceeds, good for sgd
    #         if epoch > 0.9 * epochs:
    #             lr = 1e-5
    #         elif epoch > 0.75 * epochs:
    #             lr = 1e-4
    #         elif epoch > 0.5 * epochs:
    #             lr = 1e-3
    #         else:
    #             lr = 1e-2
    if epoch > 0.9 * epochs:
        lr = 1e-5
    elif epoch > 0.75 * epochs:
        lr = 1e-4
    elif epoch > 0.5 * epochs:
        lr = 1e-3
    else:
        lr = 1e-2
    print('lr: %f' % lr)
    return lr


#     if epoch > 0.75 * epochs:
#        lr = 1e-5
#     elif epoch > 0.5 * epochs:
#        lr = 1e-4
#     elif epoch > 0.25 * epochs:
#        lr = 1e-3
#     else:
#        lr = 1e-2
#     print('lr: %f' % lr)
#     return lr

scheduler = LearningRateScheduler(lr_scheduler)
# =============================================================================

# =============================================================================
# x_train = x_train.astype('float32') / 255. - 0.5
# x_test = x_test.astype('float32') / 255. - 0.5
# =============================================================================
with tf.device('/gpu:0'):
    model = Sequential()

    model.add(TimeDistributed(Conv2D(32, (3, 3), strides=1, padding='valid'), input_shape=x_train_new.shape[1:]))
    # model.add(TimeDistributed(MaxPooling2D((3, 3), strides=1, padding='valid', data_format='channels_last')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    #    model.add(TimeDistributed(Dropout(0.4)))

    model.add(TimeDistributed(Conv2D(64, (3, 3))))
    # model.add(TimeDistributed(MaxPooling2D((3, 3), strides=1, padding='valid', data_format='channels_last')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Dropout(0.4)))

    model.add(TimeDistributed(Conv2D(128, (3, 3))))
    # model.add(TimeDistributed(MaxPooling2D((3, 3), strides=1, padding='valid', data_format='channels_last')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Dropout(0.4)))

    model.add(TimeDistributed(Flatten()))

    model.add(TimeDistributed(Dense(512)))
    #    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Dropout(0.5)))

    model.add(TimeDistributed(Dense(512)))
    #    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Dropout(0.5)))
    #    model.add(TimeDistributed(Dropout(0.5)))
    # model.add(TimeDistributed(Lambda(lambda x: tf.expand_dims(model.output, axis=-1))))
    # model.add(Bidirectional(LSTM(512, return_sequences=True)))
    model.add(LSTM(512, return_sequences=True))
    model.add(TimeDistributed(Dropout(0.5)))
    #    model.add(LSTM(512, return_sequences=True))
    # model.add(TimeDistributed(BatchNormalization()))

    #    model.add(TimeDistributed(SeqSelfAttention(attention_activation='sigmoid')))
    model.add(TimeDistributed(Dense(12)))
    #    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('softmax')))

# nadam = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)


################################################################model.save(h5_dir + 'model_uni-lstm_RangeAngle_64x256_255normalized_12label_80-20_batch12_ProgressSgd.h5')
# model.save_weights(h5_dir + 'weights_uni-lstm_RangeAngle_64x256_255normalized_12label_80-20_batch12_ProgressSgd.h5')

sgd = optimizers.SGD(lr=0.0, momentum=0.9, decay=0.0, nesterov=False)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

history = model.fit(x_train_new, y_train_new,
                    batch_size=12, epochs=40, shuffle=True, verbose=1,
                    callbacks=[scheduler],
                    validation_data=(x_test_new, y_test_new)
                    )

# score = model.evaluate(x_test, y_test, batch_size=12)

# history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=7),
#                              epochs=60, shuffle=True, verbose=1,
##          callbacks=[scheduler],
#          validation_data = (x_test, y_test))

# model.fit(x_train, y_train, batch_size=7, epochs=40, shuffle=True, verbose=1)

predictions = model.predict(x_test_new)
# predictions_classes = model.predict_classes(x_test)

##################################################################################
#


#
##print(history.history.keys())
#
#################################################################################

######################################################
#


######################################################
# y_pred = (predictions > 0.5)
# y_p=(y_pred>0).astype(int)
#######################################################
# xx=[]
# yy=[]
# yy_temp=[]

######################################################

# label=[1,2,3,4,5,6,7]
#
# for i in range(0, len(y_test)):
#    for j in range(0, 7):
#       if y_test[i,1,j] ==1 :
#           xx.append(j+1)
#
# for i in range(0, len(y_test)):
#    y_p_slice = y_p[i,:,:]
#    yy_temp=[]
#    for j in range(0, 64):
#        for k in range(0, 7):
#            if y_p_slice[j,k] == 1 :
#               yy_temp.append(k+1)
#               label_of_frame = max(set(yy_temp), key=yy_temp.count)
#
#    yy.append(label_of_frame)

######################################################

# label=[1,2,3,4,5,6,7,8,9,10,11,12]
#
# for i in range(0, len(y_test)):
#    for j in range(0, 12):
#       if y_test[i,1,j] ==1 :
#           xx.append(j+1)
#
# for i in range(0, len(y_test)):
#    y_p_slice = y_p[i,:,:]
#    yy_temp=[]
#    for j in range(0, 64):
#        for k in range(0, 12):
#            if y_p_slice[j,k] == 1 :
#               yy_temp.append(k+1)
#               label_of_frame = max(set(yy_temp), key=yy_temp.count)
#
#    yy.append(label_of_frame)
#
#######################################################
###matrix = metrics.confusion_matrix(y_test.argmax(axis=1), y_p.argmax(axis=1))
# mat = metrics.confusion_matrix(xx, yy,labels=label)
### =============================================================================
# plt.matshow(mat)
# plt.colorbar()
# plt.xlabel('predicted')
# plt.ylabel('answer')
# plt.xticks(np.arange(mat.shape[1]),label)
# plt.yticks(np.arange(mat.shape[1]),label)
# plt.show()
# =============================================================================
# mat_percentage =mat.astype(float)
########################################################

# for i in range(0,7):
#       for j in range(0,7):
#            s = float(sum(mat[i,:]))
#            mat_percentage[i,j]= mat_percentage[i,j]/s

########################################################

# for i in range(0, 12):
#       for j in range(0,12):
#            s = float(sum(mat[i,:]))
#            mat_percentage[i,j]= mat_percentage[i,j]/s

########################################################
# mat_percentage = []
# for i in range(0, len(predictions[:,1,1])):
#    for j in range(0, 12):
#        mat_percentage[] = sum(predictions[i,:,j])

########################################################

# mat_percentage =mat.astype(float)
########################################################

# for i in range(0,7):#frames_prob_sum=[]
# temp=[]
# mat_percentage = np.zeros((12, 12))
#
# for i in range(0, 64):
#    frames_prob_sum = np.sum(predictions, axis=1) # sum as (test_num, label)
#
# count=0;
#
################## ladder-shaped 1-to-7 label###########
##for j in range(0, 24):
##    if j==0:
##        temp = frames_prob_sum[count:count+7,:]
##    else:
##        temp = temp + frames_prob_sum[count:count+7,:]
##    count+=7
##mat_percentage[0:7, :] = temp/24
########################################################
# for j in range(0, 7):
#    temp = sum(frames_prob_sum[count:count+24,:])
#    mat_percentage[j, :] = temp/24
#    count+=24
#
# for j in range(7, 10):
#    temp = sum(frames_prob_sum[count:count+12,:])
#    mat_percentage[j, :] = temp/12
#    count=count+12
#
# for j in range(10, 12):
#    temp = sum(frames_prob_sum[count:count+32,:])
#    mat_percentage[j, :] = temp/32
#    count=count+32
#
# mat_percentage = mat_percentage/64
#
# print('The vali accuracy derived from confusion matrix is:\n')
# print(np.trace(mat_percentage)/12)
#       for j in range(0,7):
#            s = float(sum(mat[i,:]))
#            mat_percentage[i,j]= mat_percentage[i,j]/s

########################################################

# for i in range(0, 12):
#       for j in range(0,12):
#            s = float(sum(mat[i,:]))
#            mat_percentage[i,j]= mat_percentage[i,j]/s

########################################################
# ==============================================================================
# mat_percentage = np.zeros((12, 12))
# for i in range(0, 24*7):
#     for j in range(0, 7):
#         mat_percentage[i%7, j] = sum(predictions[i,:,j])/64
#
#
# for i in range(24*7, 24*7 + 10):
#     for j in range(0, 12):
#         mat_percentage[7, j] = sum(predictions[i,:,j])/64
#
# for i in range(24*7 + 10, 24*7 + 20):
#     for j in range(0, 12):
#         mat_percentage[8, j] = sum(predictions[i,:,j])/64
#
# for i in range(24*7 + 20, 24*7 + 30):
#     for j in range(0, 12):
#         mat_percentage[9, j] = sum(predictions[i,:,j])/64
#
# for i in range(24*7 + 30, 24*7 + 62):
#     for j in range(0, 12):
#         mat_percentage[10, j] = sum(predictions[i,:,j])/64
#
# for i in range(24*7 + 62, 24*7 + 94):
#     for j in range(0, 12):
#         mat_percentage[11, j] = sum(predictions[i,:,j])/64
# ==============================================================================
## confusion for scene-mixing

# frames_prob_sum=[]
# temp=[]
# mat_percentage = np.zeros((12, 12))
#
# for i in range(0, 64):
#    frames_prob_sum = np.sum(predictions, axis=1) # sum as (test_num, label)
#
# count=0;
# for j in range(0, 7):
#    temp = sum(frames_prob_sum[count:count+42,:])
#    mat_percentage[j, :] = temp/42
#    count+=42
#
# for j in range(7, 10):
#    temp = sum(frames_prob_sum[count:count+24,:])
#    mat_percentage[j, :] = temp/24
#    count+=24
#
# for j in range(10, 12):
#    temp = sum(frames_prob_sum[count:count+52,:])
#    mat_percentage[j, :] = temp/52
#    count+=52
#
# mat_percentage = mat_percentage/64
# print('\nThe vali accuracy derived from confusion matrix is:\n')
# print(np.trace(mat_percentage)/12)


##############################################################
## confusion for lab210

# frames_prob_sum=[]
# temp=[]
# mat_percentage = np.zeros((12, 12))
#
# for i in range(0, 64):
#    frames_prob_sum = np.sum(predictions, axis=1) # sum as (test_num, label)
#
# count=0;
#
################## ladder-shaped 1-to-7 label###########
##for j in range(0, 24):
##    if j==0:
##        temp = frames_prob_sum[count:count+7,:]
##    else:
##        temp = temp + frames_prob_sum[count:count+7,:]
##    count+=7
##mat_percentage[0:7, :] = temp/24
########################################################
# for j in range(0, 7):
#    temp = sum(frames_prob_sum[count:count+24,:])
#    mat_percentage[j, :] = temp/24
#    count+=24
#
# for j in range(7, 10):
#    temp = sum(frames_prob_sum[count:count+12,:])
#    mat_percentage[j, :] = temp/12
#    count=count+12
#
# for j in range(10, 12):
#    temp = sum(frames_prob_sum[count:count+32,:])
#    mat_percentage[j, :] = temp/32
#    count=count+32
#
# mat_percentage = mat_percentage/64
#
# print('The vali accuracy derived from confusion matrix is:\n')
# print(np.trace(mat_percentage)/12)
# np.save(h5_dir + 'mat_percentage', mat_percentage)
#########################################################

##confusion for H104

# frames_prob_sum=[]
# temp=[]
# mat_percentage = np.zeros((12, 12))
#
# for i in range(0, 64):
#    frames_prob_sum = np.sum(predictions, axis=1) # sum as (test_num, label)
#
# count=0;
# for j in range(0, 7):
#    temp = sum(frames_prob_sum[count:count+18,:])
#    mat_percentage[j, :] = temp/18
#    count+=18
#
# for j in range(7, 10):
#    temp = sum(frames_prob_sum[count:count+12,:])
#    mat_percentage[j, :] = temp/12
#    count+=12
#
# for j in range(10, 12):
#    temp = sum(frames_prob_sum[count:count+20,:])
#    mat_percentage[j, :] = temp/20
#    count+=20
#
# mat_percentage = mat_percentage/64
# print('The vali accuracy derived from confusion matrix is:\n')
# print(np.trace(mat_percentage)/12)

# ==============================================================================
# confusion for lab210 other points
# frames_prob_sum=[]
# temp=[]
# mat_percentage = np.zeros((12, 12))
#
# for i in range(0, 64):
#    frames_prob_sum = np.sum(predictions, axis=1) # sum as (test_num, label)
#
# count=0;
# for j in range(0, 12):
#    temp = sum(frames_prob_sum[count:count+20,:])
#    mat_percentage[j, :] = temp/20
#    count+=20
#
# mat_percentage = mat_percentage/64
# print('The vali accuracy derived from confusion matrix is:\n')
# print(np.trace(mat_percentage)/12)


# ==============================================================================
## confusion matrix for H210 whole testing

# frames_prob_sum=[]
# temp=[]
# mat_percentage = np.zeros((12, 12))
#
# for i in range(0, 64):
#    frames_prob_sum = np.sum(predictions, axis=1) # sum as (test_num, label)
#
# count=0;
# for j in range(0, 7):
#    temp = sum(frames_prob_sum[count:count+120,:])
#    mat_percentage[j, :] = temp/120
#    count+=120
#
# for j in range(7, 10):
#    temp = sum(frames_prob_sum[count:count+60,:])
#    mat_percentage[j, :] = temp/60
#    count+=60
#
# for j in range(10, 12):
#    temp = sum(frames_prob_sum[count:count+160,:])
#    mat_percentage[j, :] = temp/160
#    count+=160
#
# mat_percentage = mat_percentage/64
# print('The vali accuracy derived from confusion matrix is:\n')
# print(np.trace(mat_percentage)/12)

# ==============================================================================
##confusion matrix for H104 whole testing

# frames_prob_sum=[]
# temp=[]
# mat_percentage = np.zeros((12, 12))
#
# for i in range(0, 64):
#    frames_prob_sum = np.sum(predictions, axis=1) # sum as (test_num, label)
#
# count=0;
# for j in range(0, 7):
#    temp = sum(frames_prob_sum[count:count+90,:])
#    mat_percentage[j, :] = temp/90
#    count+=90
#
# for j in range(7, 10):
#    temp = sum(frames_prob_sum[count:count+60,:])
#    mat_percentage[j, :] = temp/60
#    count+=60
#
# for j in range(10, 12):
#    temp = sum(frames_prob_sum[count:count+100,:])
#    mat_percentage[j, :] = temp/100
#    count+=100
#
# mat_percentage = mat_percentage/64
# print('The vali accuracy derived from confusion matrix is:\n')
# print(np.trace(mat_percentage)/12)
# ==============================================================================
## Aug. NewConfig data_cross-scene
#
# frames_prob_sum=[]
#
# mat_percentage = np.zeros((12, 12))
#
# for i in range(0, 64):
#    frames_prob_sum = np.sum(predictions, axis=1) # sum as (test_num, label)
#
# count=0;
# for j in range(0, 12):
#    count=10*j
#    temp_labelSum=[]
#    temp=np.zeros((1, 12))
#    for k in range(0, 12):
#        temp_labelSum = sum(frames_prob_sum[count:count+10,:])
#        count+=120
#        temp = temp + temp_labelSum
#    mat_percentage[j, :] = temp
#
# mat_percentage = mat_percentage/(64*12*10)
# print('The vali accuracy derived from confusion matrix is:\n')
# print(np.trace(mat_percentage)/12)

# ==============================================================================
# Aug. NewConfig data_cross-scene

frames_prob_sum = []

mat_percentage = np.zeros((12, 12))

for i in range(0, 64):
    frames_prob_sum = np.sum(predictions, axis=1)  # sum as (test_num, label)

count = 0;
# np1 = np.zeros([12])
# np_count = 0

for j in range(0, 12):
    temp_labelSum = []
    temp = np.zeros((1, 12))

    temp_labelSum = sum(frames_prob_sum[count:count + 10, :])
    count += 10
    temp = temp + temp_labelSum
    mat_percentage[j, :] = temp
mat_percentage = mat_percentage / (64 * 10)

print('The vali accuracy derived from confusion matrix is:\n')
print(np.trace(mat_percentage) / 12)

tempdiag = np.diag(mat_percentage)
print(tempdiag)
df = pd.DataFrame(tempdiag)
df = df.T
df.to_excel('C:\\Users\\user\\Desktop\\excel_output.xls')
# ==========================================================================
# Aug. NewConfig data original cases
# frames_prob_sum=[]
#
# mat_percentage = np.zeros((12, 12))
#
# for i in range(0, 64):
#    frames_prob_sum = np.sum(predictions, axis=1) # sum as (test_num, label)
#
# count=0;
# for j in range(0, 12):
#
#    temp_labelSum=[]
#    temp=np.zeros((1, 12))
#
#    temp_labelSum = sum(frames_prob_sum[count:count+48,:])
#    count+=48
#    temp = temp_labelSum
#    mat_percentage[j, :] = temp
#
# mat_percentage = mat_percentage/(64*48)
# print('The vali accuracy derived from confusion matrix is:\n')
# print(np.trace(mat_percentage)/12)
# ==============================================================================
## when gesture 3 and 5 is removed

# frames_prob_sum=[]
# temp=[]
# mat_percentage = np.zeros((10, 10))
#
# for i in range(0, 64):
#    frames_prob_sum = np.sum(predictions, axis=1) # sum as (test_num, label)
#
# count=0;
# for j in range(0, 5):
#    temp = sum(frames_prob_sum[count:count+90,:])
#    mat_percentage[j, :] = temp/90
#    count+=90
#
##for j in len(3, 5, 6):
##    temp = sum(frames_prob_sum[count:count+90,:])
##    mat_percentage[j, :] = temp/90
##    count+=90
#
# for j in range(5, 8):
#    temp = sum(frames_prob_sum[count:count+60,:])
#    mat_percentage[j, :] = temp/60
#    count+=60
#
# for j in range(8, 10):
#    temp = sum(frames_prob_sum[count:count+100,:])
#    mat_percentage[j, :] = temp/100
#    count+=100
#
# mat_percentage = mat_percentage/64
# print('The vali accuracy derived from confusion matrix is:\n')
# print(np.trace(mat_percentage)/10)
# ==============================================================================

########################################################
# tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
# X_tsne = tsne.fit_transform(xx)
#
# x_min, x_max = X_tsne.min(0), X_tsne.max(0)
# X_norm = (X_tsne - x_min) / (x_max - x_min)
# plt.figure(figsize=(8, 8))
# for i in range(X_norm.shape[0]):
#    plt.text(X_norm[i, 0], X_norm[i, 1], str(yy[i]), color=plt.cm.Set1(yy[i]),
#             fontdict={'weight': 'bold', 'size': 9})
# plt.xticks([])
# plt.yticks([])
# plt.show()

if (modelis == "RDI"):
    output_dir = 'D:\\real-time-radar\\RDIdata\\'
elif(modelis == "RAI"):
    output_dir = 'D:\\real-time-radar\\RAIdata\\'
# output_dir = h5_dir

np.save(output_dir + 'mat_percentage.npy', mat_percentage)
np.save(output_dir + 'predictions.npy', predictions)
# summarize history for accuracy
fig_acc = plt.gcf()
plt.plot(history.history['accuracy'], linestyle='--')
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.draw()

fig_acc.savefig(output_dir + 'acc.png')
# fig_acc.savefig(output_dir +'acc.png')

# summarize history for loss
fig_loss = plt.gcf()
plt.plot(history.history['loss'], linestyle='--')
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.draw()

fig_loss.savefig(output_dir + 'loss.png')
# fig_loss.savefig(output_dir + 'loss.png')


model.save(output_dir + 'model_NewConfig_bilstm_3t4rRAI_32x32_Gaussian_batch12_ProgressSGD.h5')
model.save_weights(output_dir + 'weightsNewConfig_bilstm_3t4rRAI_32x32_Gaussian_batch12_ProgressSGD.h5')

# model = load_model(output_dir + 'model_NewConfig_bilstm_1t4rRDI_32x32_Gaussian_batch12_ProgressSGD.h5')
# model.load_weights(model_dir + 'weights_3t4r_lstm_RangeDoppler_32x32_gaussed_batch12_ProgressSgd.h5', by_name = True)
# score = model.evaluate(x_test, y_test, batch_size=12)
# predictions = model.predict(x_test_new)

#  data numerical 50% to 50%
# 100-0: 61.22%   64.04%
# 90-10: 74.49%   80.56%
# 80-20: 79.51%   84.11%
# 70-30: 81.53%   86.00%
# 60-40: 82.01%   86.46%
# 50-50: 80.27%   85.27%

# 40-60:
# 30-70:
# 20-80:
# 10-90:


#  data numerical 80% to 20%
# 100-0: