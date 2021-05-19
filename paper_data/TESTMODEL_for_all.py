import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.optimizers as optimizers
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, \
    Flatten, Dense, LSTM, TimeDistributed, Bidirectional, Activation
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.preprocessing import OneHotEncoder

import DSP
import os
from read_binfile import read_bin_file


def plot_confusion_matrix(cm_normalized, classes, save_file=False):
    plt.figure(figsize=(12, 8), dpi=60)
    x_location = np.array(range(classes))
    x, y = np.meshgrid(x_location, x_location)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c > 0.01:
            plt.text(x_val, y_val, "%0.2f" % (c * 100,), color='red', fontsize=12, \
                     va='center', ha='center')
        plt.xticks(x_location, np.arange(classes))
    plt.yticks(x_location, np.arange(classes))
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')
    plt.tick_params(axis=u'both', which=u'both', length=0)
    plt.imshow(cm_normalized, interpolation='nearest', cmap='gray_r')
    plt.colorbar()
    plt.show()
    if save_file:
        plt.savefig()


def read_dataset(data_path, sense, gesture, data_type, times, frame_num):
    data = []
    label = []
    print('Sense:', sense)
    print('==============================')
    for g in range(gesture):
        filename = data_type + '_S' + str(sense) + 'G' + str(g) + '.npy'
        data_tmp = np.load(data_path + filename)
        data_len = len(data_tmp[0])
        label_tmp = np.zeros(data_len*data_len)
        print(type(data_tmp))
        label_tmp = label_tmp + g
        data.append(data_tmp)
        label.append(label_tmp)


    print('==============================')

    if data_type == 'RAI':
        data = np.reshape(data, [np.shape(data)[0], np.shape(data)[1], np.shape(data)[2], -1])
        # data = np.reshape(data, [np.shape(data)[0], np.shape(data)[1], -1])
    #     data = data[0:720,:,:,:,:]

    # onehotencoder = OneHotEncoder(categorical_features=[0])
    # data_str_ohe = onehotencoder.fit_transform(data).toarray()
    # label = data_str_ohe
    # label = np.reshape(label, (np.shape(data)[0], 64))
    # label = to_categorical(label)

    return data, label


# data_path = 'C:\\Users\\user\\Desktop\\new\\'
data_path = 'D:/train_data/Matt_Yen/after_p_1t4r/'

epochs = 40
sense = [0, 1]
gesture = 12
times = 10
frame_num = 64
batch_sizes = 12
dataset_type = "RAI"

norm_test, y_test = read_dataset(data_path, sense[0], gesture, dataset_type, times, frame_num)
print('shape of norm_test :' + str(np.shape(norm_test)))
print('shape of y_test :' + str(np.shape(y_test)))

norm_test = np.expand_dims(norm_test, axis=-1)
# print((np.shape(norm_test)))
data_path = 'C:/data/Studio_data/2t4r_good/'
new_model = keras.models.load_model(data_path + 'my_model.h5')
new_model.summary()

print(np.shape(norm_test))
# norm_test = np.transpose(norm_test, [4, 0, 1, 2, 3])
print(np.shape(norm_test))
prediction = new_model.predict(norm_test)
print('shape of predictiont :' + str(np.shape(prediction)))
prediction = np.reshape(prediction, [-1, gesture])
print('shape of predictiont :' + str(np.shape(prediction)))
prediction_cat = np.argmax(prediction, axis=1)
print('shape of prediction_cat :' + str(np.shape(prediction_cat)))

# y_test = np.reshape(y_test, [-1, 4])
print('shape of y_test :' + str(np.shape(y_test)))
# y_cat = np.argmax(y_test, axis=1)
# print('shape of y_cat :' + str(np.shape(y_cat)))
y_cat = y_test

print('----------------------------------')
print(prediction_cat)
text_file = open("C:\\Users\\user\\Desktop\\Output.txt", "w")
text_file.write(str(prediction_cat))
text_file.close()
print('----------------------------------')
print("Plot Confusion Matrix")
cm = confusion_matrix(y_cat, prediction_cat,[0, 1, 2, 3])  # get confusion matrix


cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # normalize cm


print('shape of cm_normalized :' + str((cm_normalized)))
plot_confusion_matrix(cm_normalized, gesture, False)  # plot confusion matrix
print("Program Terminate")
sys.exit(0)

# score=new_model.evaluate(norm_test, y_test, verbose=2)
# pred = new_model.predict([norm_test], batch_size=1, verbose=0)
