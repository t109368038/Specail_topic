from __future__ import print_function
import os
import sys

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
from tensorflow_core.python.keras.layers import MaxPooling2D
from preprocessing import data_Standardization
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from sklearn.model_selection import train_test_split
from tensorflow import keras

from sklearn.utils import shuffle

# modelis = "RAI"
modelis = "RDI"

def plot_confusion_matrix(cm_normalized, classes, save_file=False):
    plt.figure(figsize=(12, 8), dpi=60)
    x_location = np.array(range(classes))
    x, y = np.meshgrid(x_location, x_location)

    np1 = np.zeros([12])
    np_count = 0
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c > 0.01:
            if (x_val == y_val):
                plt.text(x_val, y_val, "%0.2f" % (c * 100,), color='white', fontsize=12, \
                         va='center', ha='center')
                np1[np_count] = c*100
                np_count += 1
            else:
                plt.text(x_val, y_val, "%0.2f" % (c * 100,), color='black', fontsize=12, \
                         va='center', ha='center')
        plt.xticks(x_location, np.arange(classes))
    print(np1)
    df = pd.DataFrame(np1)
    df = df.T
    df.to_excel('C:\\Users\\user\\Desktop\\excel_output.xls')

    plt.yticks(x_location, np.arange(classes))
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')
    plt.tick_params(axis=u'both', which=u'both', length=0)
    plt.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
    plt.colorbar()
    plt.show()
    if save_file:
        plt.savefig()

if (modelis == "RDI"):
    h5_dir = 'D:\\real-time-radar\\RDIdata\\'
    h5_dir1 = 'D:\\real-time-radar\\gest_August\\cross-scene\\3t4rRDI\\'
elif (modelis == "RAI"):
    h5_dir = 'D:\\real-time-radar\\RAIdata\\'
    h5_dir1 = 'D:\\real-time-radar\\gest_August\\cross-scene\\3t4rRAI\\'
#
x_train = np.load(h5_dir1 + 'x_train.npy')
y_train = np.load(h5_dir1 + 'y_train.npy')

# x_test_blank = np.load(blank_dir + 'x_blank.npy')

x_test = np.load(h5_dir1 + 'x_test.npy')
y_test = np.load(h5_dir1 + 'y_test.npy')
# print (x_test.shape)
# print (y_test.shape)

x_train = np.transpose(x_train, (0, 1, 3, 4, 2))
x_test = np.transpose(x_test, (0, 1, 3, 4, 2))

if (modelis == "RAI"):
    x_train = x_train[:, :, :, :, 1]
    x_test = x_test[:, :, :, :, 1]
    x_train = np.reshape(x_train, [1440, 64, 32, 32, -1])
    x_test = np.reshape(x_test, [1440, 64, 32, 32, -1])
elif(modelis =="RDI"):
    x_train = x_train[:, :, :, :, 0:3]
    x_test = x_test[:, :, :, :,  0:3]


x_train_new = x_train
y_train_new = y_train

x_test_new = x_test
y_test_new = y_test

print("Training Data Shape：", np.shape(x_train))
print("Testing Data Shape：", np.shape(x_test))
print("Training Label Shape：", np.shape(y_train))
print("Testing Label Shape：", np.shape(y_test))
# ----------------------------------------------------------------
for i in range(0, 1440):  # train as 80%
    #    if i % 10 == 9 or i % 10 == 8 or i % 10 == 7 or i % 10 == 6 or i % 10 == 5:
    #    if i % 10 == 9 or i % 10 == 8 or i % 10 == 7 or i % 10 == 6:
    #    if i % 10 == 9 or i % 10 == 8 or i % 10 == 7:z/
    if i % 10 == 0 or i % 10 == 1 or i % 10 == 2 or i % 10 == 3 or i % 10 == 4:
    # if i % 10 == 5 or i % 10 == 6 or i % 10 == 7 or i % 10 == 8 or i % 10 == 9:
        #    if i % 10 == 9:
        x_train_new[i, ...] = x_test[i, ...]
        y_train_new[i, ...] = y_test[i, ...]
#        x_train_new = np.delete(x_train_new, i, 0)
#        y_train_new = np.delete(y_train_new, i, 0)
# for i in range(0, 1440):  # test as 20%
#        if i % 10 == 0 or i %  10 == 1 or i % 10 == 2 or i % 10 == 3 or i % 10 == 4 or i % 10 == 5 or i % 10 == 6 or i % 10 == 7:
    if i % 10 == 5 or i % 10 == 6 or i % 10 ==7 or i % 10 == 8 or i % 10 == 9:
        x_test_new[i, ...] = x_train[i, ...]
        y_test_new[i, ...] = y_train[i, ...]
#        x_test_new = np.delete(x_test_new, i, 0)
#        y_test_new = np.delete(y_test_new, i, 0)

x_test_new = data_Standardization(x_test_new)
x_train_new = data_Standardization(x_train_new)
del x_train, y_train, x_test, y_test
# ----------------------------------------------------------------

# print("x_test is  : " + str(y_train.shape))
# print("x_train is  : " + str(y_test.shape))

# x_train = np.reshape(x_train, [1440, 64, 32, 32, -1])
# x_test = np.reshape(x_test, [1440, 64, 32, 32, -1])
# x_train_s0, x_test_s0, y_train_s0, y_test_s0 = train_test_split(x_train, y_train, test_size=0.5, random_state=2)
# x_train_s1, x_test_s1, y_train_s1, y_test_s1 = train_test_split(x_test, y_test, test_size=0.5, random_state=2)

# x_train_new = np.concatenate([x_train_s0, x_train_s1], axis=0)
# y_train_new = np.concatenate([y_train_s0, y_train_s1], axis=0)
# x_test_new = np.concatenate([x_test_s0, x_test_s1], axis=0)
# y_test_new = np.concatenate([y_test_s0, y_test_s1], axis=0)

# x_train_new = data_normalize_toall(x_train_new)
# x_test_new = data_normalize_toall(x_test_new)
# x_train_new, y_train_new, x_test_new, y_test_new = shuffle(x_train_new, y_train_new, x_test_new, y_test_new, random_state=2)

#
# print("x_train_new is  : " + str(x_train_new.shape))
# print("x_test_new is  : " + str(x_test_new.shape))
# x_train_new = x_train
# y_train_new = y_train
#
# x_test_new = x_test
# y_test_new = y_test
#
# ----------------------------------------------------------------
epochs = 40
lr_power = 0.9
lr_base = 1e-2

def lr_scheduler(epoch):
    # def lr_scheduler(epoch, mode='progressive_drops'):
    '''if lr_dict.has_key(epoch):
        lr = lr_dict[epoch]
        print 'lr: %f' % lr'''

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


scheduler = LearningRateScheduler(lr_scheduler)

with tf.device('/gpu:0'):
    model = Sequential()

    model.add(TimeDistributed(Conv2D(32, (3, 3), strides=1, padding='valid'), input_shape=x_train_new.shape[1:]))
    model.add(TimeDistributed(MaxPooling2D((3, 3), strides=1, padding='valid', data_format='channels_last')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    #    model.add(TimeDistributed(Dropout(0.4)))

    model.add(TimeDistributed(Conv2D(64, (3, 3))))

    model.add(TimeDistributed(MaxPooling2D((3, 3), strides=1, padding='valid', data_format='channels_last')))

    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Dropout(0.4)))

    model.add(TimeDistributed(Conv2D(128, (3, 3))))

    model.add(TimeDistributed(MaxPooling2D((3, 3), strides=1, padding='valid', data_format='channels_last')))

    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Dropout(0.4)))

    model.add(TimeDistributed(Flatten()))

    model.add(TimeDistributed(Dense(512)))
    #model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Dropout(0.5)))

    model.add(TimeDistributed(Dense(512)))
    # model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Dropout(0.5)))
    #    model.add(TimeDistributed(Dropout(0.5)))
    # model.add(TimeDistributed(Lambda(lambda x: tf.expand_dims(model.output, axis=-1))))
    model.add(Bidirectional(LSTM(512, return_sequences=True)))
    # model.add(LSTM(512, return_sequences=True))
    # model.add(LSTM(512, return_sequences=True))
    # model.add(LSTM(512, return_sequences=True))
    model.add(TimeDistributed(Dropout(0.5)))
       # model.add(LSTM(512, return_sequences=True))
    # model.add(TimeDistributed(BatchNormalization()))

       # model.add(TimeDistributed(SeqSelfAttention(attention_activation='sigmoid')))
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
# model = keras.models.load_model("D:\\train_data\\Matt_Yen\\" + 'my_model.h5')
# model.summary()

prediction = model.predict(x_test_new)

y_test_new = np.reshape(y_test_new, [-1, 12])
y_cat = np.argmax(y_test_new, axis=1)

prediction = np.reshape(prediction, [-1, 12])
prediction_cat = np.argmax(prediction, axis=1)

print('----------------------------------')
print("Plot Confusion Matrix")
cm = confusion_matrix(y_cat, prediction_cat)  # get confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # normalize cm
print('shape of cm_normalized :'+str(cm_normalized))
plot_confusion_matrix(cm_normalized, 12, False)  # plot confusion matrix
print("Program Terminate")

# print('shape of y_test :'+str(np.shape(y_test)))
# print('shape of y_cat :'+str(np.shape(y_cat)))

if (modelis == "RDI"):
    output_dir = 'D:\\real-time-radar\\RDIdata\\'
elif (modelis == "RAI"):
    output_dir = 'D:\\real-time-radar\\RAIdata\\'
# output_dir = h5_dir
# summarize history for accuracy
# fig_acc = plt.gcf()
# plt.plot(history.history['accuracy'], linestyle='--')
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# plt.draw()
#
# fig_acc.savefig(output_dir + 'acc.png')
# # fig_acc.savefig(output_dir +'acc.png')
#
# # summarize history for loss
# fig_loss = plt.gcf()
# plt.plot(history.history['loss'], linestyle='--')
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# plt.draw()
#
# fig_loss.savefig(output_dir + 'loss.png')
# # fig_loss.savefig(output_dir + 'loss.png')


model.save(output_dir + 'model_NewConfig_bilstm_3t4rRAI_32x32_Gaussian_batch12_ProgressSGD.h5')
model.save_weights(output_dir + 'weightsNewConfig_bilstm_3t4rRAI_32x32_Gaussian_batch12_ProgressSGD.h5')


sys.exit(0)


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
