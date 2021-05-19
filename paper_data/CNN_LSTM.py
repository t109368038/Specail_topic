import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tensorflow
import tensorflow.keras.optimizers as optimizers
import sys
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, \
    Flatten, Dense, LSTM, TimeDistributed, Bidirectional, Activation,LayerNormalization
from sklearn.metrics import confusion_matrix
from preprocessing import data_Standard,data_normalize,data_normalize_toall,data_Standard_toall,data_Standardization,new_read_dataset
from preprocessing import data_Standard,data_normalize,data_normalize_toall,data_Standard_toall,data_Standardization,new_read_dataset


def read_dataset(data_path, sense, gesture, data_type, times, frame_num):
    data = []
    label = []
    print('Sense:', sense)
    print('==============================')
    for g in range(gesture):
        filename = data_type + '_S' + str(sense) + 'G' + str(g) + '.npy'
        print('Read File:' + filename)
        data_tmp = np.load(data_path + filename)
        data.extend(data_tmp)
        label.extend(np.zeros(gesture * times * frame_num) + g)
    print('%%%%%%%%%%%%%%%%')
    print((np.shape(label)))
    print('%%%%%%%%%%%%%%%%')
    print('==============================')
    if data_type == 'RAI':
        data = np.reshape(data, [np.shape(data)[0], np.shape(data)[1], np.shape(data)[2], np.shape(data)[3], -1])
        # data = np.reshape(data, [np.shape(data)[0], np.shape(data)[1], -1])
    #     data = data[0:720,:,:,:,:]
    #     print('%%%%%%%%%%%%%%%%')
    #     print((data.shape))
    #     print('%%%%%%%%%%%%%%%%')
    label = np.reshape(label, [np.shape(data)[0], 64])


    label = to_categorical(label)
    print('%%%%%%%%%%%%%%%%')
    print((np.shape(label)))
    print('%%%%%%%%%%%%%%%%')
    return data, label


def lr_scheduler(epoch):
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

def mattyen_rand(x_train,x_test,y_train,y_test):
    x_train_new = x_train
    y_train_new = y_train
    x_test_new = x_test
    y_test_new = y_test
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

    return  x_train_new,x_test_new,y_train_new,y_test_new

config = tensorflow.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tensorflow.compat.v1.InteractiveSession(config=config)

# data_path = '../data/Studio_data/2t4r_v132/'
data_path = 'D:/train_data/Matt_Yen/after_p_1t4r/'
# data_path = 'C:/data/Studio_data/2t4r_good/'

# data_path = 'C:/data/Studio_data/2t4r_v132/'
epochs = 40
sense = [0, 1]
gesture = 12
times = 10
frame_num = 64
batch_sizes = 12
dataset_type = "RAI"

print("Training Settings：")
print('==============================')
print("Epochs：", epochs)
print("Gesture Classes：", gesture)
print("Batch Size:", batch_sizes)
print("Input Data：", dataset_type)
print('==============================')

data_s0, label_s0 = new_read_dataset(data_path, sense[0], gesture, dataset_type, times, frame_num)
data_s0 = np.array(data_s0)
data_s1, label_s1 = new_read_dataset(data_path, sense[1], gesture, dataset_type, times, frame_num)
data_s1 = np.array(data_s1)


# print('=============standarding=================')
# data_s0 = data_Standard(data_s0)# 分開做
# data_s1 = data_Standard(data_s1)
# data_s0 = data_Standard_toall(data_s0)#一起做
# data_s1 = data_Standard_toall(data_s1)
# print('=============Normalizing=================')
# data_s0 = data_normalize(data_s0)  # 分開做
# data_s1 = data_normalize(data_s1)
# data_s0 = data_normalize_toall(data_s0) #一起做
# data_s1 = data_normalize_toall(data_s1)
# print('=============Matt Yan method=================')
# data_s0 = data_Standardization(data_s0)
# data_s1 = data_Standardization(data_s1)


print("size of data_s0"+str(np.shape(data_s0)))
print("size of label_s0"+str(np.shape(label_s0)))
print("size of data_s1"+str(np.shape(data_s1)))
print("size of label_s1"+str(np.shape(label_s1)))

# x_train, x_test, y_train, y_test = train_test_split(data_s0, label_s0, test_size=0.2, random_state=2)
data_s0, data_s1, label_s0, label_s1 = shuffle(data_s0, data_s1, label_s0, label_s1, random_state=2)
# for cross sense
# x_train_s0, x_test_s0, y_train_s0, y_test_s0 = train_test_split(data_s0, label_s0, test_size=0.5, random_state=2)
# x_train_s1, x_test_s1, y_train_s1, y_test_s1 = train_test_split(data_s1, label_s1, test_size=0.5, random_state=2)
# x_train = np.concatenate([x_train_s0, x_train_s1], axis=0)
# y_train = np.concatenate([y_train_s0, y_train_s1], axis=0)
# x_test = np.concatenate([x_test_s0, x_test_s1], axis=0)
# y_test = np.concatenate([y_test_s0, y_test_s1], axis=0)
x_train,x_test,y_train,y_test=mattyen_rand(data_s0,data_s1,label_s0,label_s1)
del data_s0,label_s0,data_s1,label_s1

# x_train = data_s0
# y_train = label_s0
# x_test = data_s1
# y_test = label_s1
print("Training x_train Shape：", np.shape(x_train))
print("Testing x_test Shape：", np.shape(x_test))
print("Training y_train Shape：", np.shape(y_train))
print("Testing y_test Shape：", np.shape(y_test))


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


scheduler = keras.callbacks.LearningRateScheduler(lr_scheduler)

# model ==============================
model = Sequential()

model.add(TimeDistributed(Conv2D(32, (3, 3), strides=1, padding='valid'), input_shape=x_train.shape[1:]))
# model.add(TimeDistributed(MaxPooling2D((3, 3), strides=1, padding='valid', data_format='channels_last')))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Activation('relu')))

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

model.add(TimeDistributed(Activation('relu')))
model.add(TimeDistributed(Dropout(0.5)))

model.add(TimeDistributed(Dense(512)))
model.add(TimeDistributed(Activation('relu')))

model.add(TimeDistributed(Dropout(0.5)))
# model.add(TimeDistributed(Lambda(lambda x: tf.expand_dims(model.output, axis=-1))))
model.add(Bidirectional(LSTM(512, return_sequences=True)))   # 雙向 LSTM
model.add(TimeDistributed(Dropout(0.5)))
# model.add(TimeDistributed(BatchNormalization()))

model.add(TimeDistributed(Dense(gesture)))
model.add(TimeDistributed(Activation('softmax')))

sgd = optimizers.SGD(lr=0.0, momentum=0.9, decay=0.0, nesterov=False)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
# model ==============================
history = model.fit(x_train, y_train,
                    batch_size=batch_sizes, epochs=epochs, shuffle=False, verbose=1,
                    callbacks=[scheduler],
                    validation_data=(x_test, y_test),
                    )

model.save(data_path+'my_model.h5')

print("Training Finished, Let's look the prediction results")
prediction = model.predict(x_test)
print('shape of predictiont :'+str(np.shape(prediction)))
prediction = np.reshape(prediction, [-1, gesture])
print('shape of predictiont :'+str(np.shape(prediction)))
prediction_cat = np.argmax(prediction, axis=1)
print('shape of prediction_cat :'+str(np.shape(prediction_cat)))


y_test = np.reshape(y_test, [-1, gesture])
y_cat = np.argmax(y_test, axis=1)
print('shape of y_test :'+str(np.shape(y_test)))
print('shape of y_cat :'+str(np.shape(y_cat)))

print('----------------------------------')
print(prediction_cat)
text_file = open("C:\\Users\\user\\Desktop\\Output.txt", "w")
text_file.write(str(prediction_cat))
text_file.close()


print('----------------------------------')
print("Plot Confusion Matrix")
cm = confusion_matrix(y_cat, prediction_cat)  # get confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # normalize cm
print('shape of cm_normalized :'+str(cm_normalized))
plot_confusion_matrix(cm_normalized, gesture, False)  # plot confusion matrix
print("Program Terminate")
sys.exit(0)
