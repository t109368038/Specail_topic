import numpy as np
import os
import matplotlib.pyplot as plt
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
    print('==============================')
    if data_type == 'RAI':
        data = np.reshape(data, [np.shape(data)[0], np.shape(data)[1], np.shape(data)[2], np.shape(data)[3], -1])
    label = np.reshape(label, [np.shape(data)[0], 64])
    label = to_categorical(label)
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


def plot_confusion_matrix(cm_std, classes, save_file=False):
    plt.figure(figsize=(12, 8), dpi=60)
    x_location = np.array(range(classes))
    x, y = np.meshgrid(x_location, x_location)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_std[y_val][x_val]
        if c > 0.01:
            plt.text(x_val, y_val, "%0.2f" % (c * 100,), color='red', fontsize=12, \
                     va='center', ha='center')
        plt.xticks(x_location, np.arange(classes))
    plt.yticks(x_location, np.arange(classes))
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')
    plt.tick_params(axis=u'both', which=u'both', length=0)
    plt.imshow(cm_std, interpolation='nearest', cmap='gray_r')
    plt.colorbar()
    plt.show()
    if save_file:
        plt.savefig()


data_path = '../data/Studio_data/2t4r_v132/'
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

data_s0, label_s0 = read_dataset(data_path, sense[0], gesture, dataset_type, times, frame_num)
data_s0 = np.array(data_s0)
data_s1, label_s1 = read_dataset(data_path, sense[1], gesture, dataset_type, times, frame_num)
data_s1 = np.array(data_s1)

# x_train, x_test, y_train, y_test = train_test_split(data_s0, label_s0, test_size=0.2, random_state=2)
data_s0, data_s1, label_s0, label_s1 = shuffle(data_s0, data_s1, label_s0, label_s1, random_state=2)
# for cross sense
x_train_s0, x_test_s0, y_train_s0, y_test_s0 = train_test_split(data_s0, label_s0, test_size=0.8, random_state=2)
x_train_s1, x_test_s1, y_train_s1, y_test_s1 = train_test_split(data_s1, label_s1, test_size=0.2, random_state=2)

x_train = np.concatenate([x_train_s0, x_train_s1], axis=0)
y_train = np.concatenate([y_train_s0, y_train_s1], axis=0)
x_test = np.concatenate([x_test_s0, x_test_s1], axis=0)
y_test = np.concatenate([y_test_s0, y_test_s1], axis=0)

# x_train = data_s0
# y_train = label_s0
# x_test = data_s1
# y_test = label_s1

print("Training Data Shape：", np.shape(x_train))
print("Testing Data Shape：", np.shape(x_test))
print("Training Label Shape：", np.shape(y_train))
print("Testing Label Shape：", np.shape(y_test))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
model.add(Bidirectional(LSTM(512, return_sequences=True)))
model.add(TimeDistributed(Dropout(0.5)))
model.add(TimeDistributed(BatchNormalization()))

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
                    validation_data=(x_test, y_test)
                    )
print("Training Finished, Let's look the prediction results")
prediction = model.predict(x_test)
prediction = np.reshape(prediction, [-1, gesture])
prediction_cat = np.argmax(prediction, axis=1)

y_test = np.reshape(y_test, [-1, gesture])
y_cat = np.argmax(y_test, axis=1)

print("Plot Confusion Matrix")
cm = confusion_matrix(y_cat, prediction_cat)  # get confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # normalize cm
plot_confusion_matrix(cm_normalized, gesture, False)  # plot confusion matrix
print("Program Terminate")
sys.exit(0)
