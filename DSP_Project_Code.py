'''
*Digital Signal Processing Project
*Title :: Speech Recognition using Spcetrogram
*Team Members ::
*   Chittoor Vamsi
*   D Mabu Jaheer Abbas
*   Pattem Gaurav Naga Maheswar
*
*Dataset Source :: 'https://www.kaggle.com/c/tensorflow-speech-recognition-challenge'
'''

'''--------------------------------- Importing all modules needed----------------------------------------'''
# Using the method to_categorical() from tensorflow.keras.utils, a numpy array (or) a vector which has integers that represent different categories,
# can be converted into a numpy array (or) a matrix which has binary values and has columns equal to the number of categories in the data.
from tensorflow.keras.utils import to_categorical

#Used to split the data set into training and testing sets
from sklearn.model_selection import train_test_split

#pandas provides high-performance data manipulation in Python
import pandas as pd

# Seaborn is a library mostly used for statistical plotting in Python.
# It is built on top of Matplotlib and provides beautiful default styles and color palettes to make statistical plots more attractive.
import seaborn as sns

# Matplotlib.pyplot is a plotting library used for 2D graphics.
import matplotlib.pyplot as plt

# NumPy is a commonly used Python data analysis package.
import numpy as np

# The OS module in Python provides functions for interacting with the operating system.
import os

# The signal processing toolbox 
from scipy import signal

#Used to work with .wav files
from scipy.io import wavfile
'''--------------------------------------------------------------------------------------------------------'''

# Assigning audio folder path to audios_dir
audios_dir = r'Train/audio'

# Words used in this project implementation
words=["yes", "no", "up", "down", "left", "right", "off", "stop"]

# Plotting an example Spectrogram of wav file of word 'YES'
sample_rate, samples = wavfile.read(r'Train/audio/yes/00f0204f_nohash_1.wav')
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

plt.figure(figsize=(30,8))
plt.pcolormesh(times, frequencies, spectrogram)
plt.imshow(spectrogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

#.............................................................................................
# Beginning : Copying spectrograms of each wav form into data
data = []
labels = []
dirs = os.listdir(audios_dir)
for d in dirs:
    if d in words:
        print(d)
        files = os.listdir(os.path.join(audios_dir,d))
        audios = [f for f in files if f.endswith('.wav')]
        for file in audios:
            # load the image, swap color channels, and resize it to be a fixed
            # 224x224 pixels while ignoring aspect ratio
            #file = audios[i]
            sample_rate, samples = wavfile.read(os.path.join(audios_dir,d,file))
            frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

            # convert the image data to NumPy arrays while scaling the pixel
            # intensities to the range [0, 255]
            #image = np.array(image)/255.0
            
            # update the data and labels lists, respectively
            if spectrogram.shape[0] >= 128 and spectrogram.shape[1] >=48:
                data.append(spectrogram[:128,:48])
                labels.append(d)
            #else:
             #   print(spectrogram.shape)

data = np.array(data)
print(data.shape)
values,count = np.unique(labels,return_counts=True)
# End : Spectrograms of each wav form copied into data
#.............................................................................................

# Plot the number of Commands(Words) vs the number of samples
plt.figure(figsize=(10,5))
index = np.arange(len(words))
plt.bar(index, count)
plt.xlabel('Commands', fontsize=12)
plt.ylabel('No of samples', fontsize=12)
plt.xticks(index, words, fontsize=15, rotation=60)
plt.title('No. of samples for each command')
plt.show()

# Backing up labels into labels_backup
labels_backup = labels

# Convert the labels to NumPy arrays while scaling the pixel
labels = np.array(labels)

# Encode target labels with value between 0 and n_classes-1.
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
ls = le.fit_transform(labels)

#Converting the labels ls into categorical
labels_categoricals = to_categorical(ls)
print(labels_categoricals)
#Printing the dimensions of labels_categoricals
print(labels_categoricals.shape)

# partition the data into training,cross-validation and testing splits using 60%,20% and 20% of data
(trainX, testX, trainY, testY) = train_test_split(data, labels_categoricals,test_size=0.20, stratify=labels_categoricals, random_state=42)
(trainX, cvX, trainY, cvY) = train_test_split(trainX, trainY,test_size=0.25, stratify=trainY, random_state=42)

print("Train data size : ",trainX.shape)
print("Cross-Validation data size : ",cvX.shape)
print("Test data size : ",testX.shape)

#Deleting the data, labels_categoricals
del data
del labels_categoricals

# Keras is a model-level library, offers high-level building blocks that are useful to develop deep learning models.
# It handles the situation in a modular way by seamlessly plugging many distinct back-end engines to Keras.
# It depends upon the backend engine that is well specialized and optimized tensor manipulation library.
from keras import backend as K

# input image dimensions
img_rows, img_cols = 128, 48
# K.image_data_format() returns the default image data format convention.
if K.image_data_format() == 'channels_first':
    trainX = trainX.reshape(trainX.shape[0], 1, img_rows, img_cols)
    testX = testX.reshape(testX.shape[0], 1, img_rows, img_cols)
    cvX = cvX.reshape(cvX.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    trainX = trainX.reshape(trainX.shape[0],img_rows, img_cols,1)
    testX = testX.reshape(testX.shape[0],img_rows, img_cols,1)
    cvX = cvX.reshape(cvX.shape[0],img_rows, img_cols,1)
    input_shape = (img_rows, img_cols,1)

print("Number of training examples :", trainX.shape[0], "and each image is of shape :",trainX.shape)
print("Number of cross validation examples :", cvX.shape[0], "and each image is of shape :",cvX.shape)
print("Number of testing examples :", testX.shape[0], "and each image is of shape :",testX.shape)

#initialize the initial learning rate, number of epochs to train for and batch size
INIT_LR = 1e-3
EPOCHS = 100
batch_size = 32
num_class = 8

from keras.models import Sequential 
from keras.layers import Dense, Activation
from keras.layers import Dropout,Flatten
from keras.layers import BatchNormalization 
from keras.layers import Conv2D, MaxPooling2D

model = Sequential()
model.add(Conv2D(8, kernel_size=(2, 2),padding='same',activation='relu',input_shape=input_shape))
model.add(Conv2D(16, kernel_size=(2, 2),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Conv2D(32, kernel_size=(2, 2),padding='same',activation='relu'))
model.add(Conv2D(64, kernel_size=(2, 2),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Conv2D(128, kernel_size=(2, 2),padding='same',activation='relu'))
model.add(Conv2D(256, kernel_size=(2, 2),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_class, activation='softmax'))

print(model.summary())

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25, min_delta=0.0001) 
mc = ModelCheckpoint('best_model.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

history = model.fit(trainX, trainY, batch_size=batch_size, epochs=EPOCHS, verbose=1, callbacks=[es,mc],validation_data=(cvX, cvY))

# plot the training loss
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.title("Train/CV Loss on Speech-To-Text Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")

# plot the training accuracy
N = EPOCHS
plt.figure()
plt.plot(np.arange(0, N), history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")
plt.title("Train and CV Accuracy on Speech-To-Text Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")

best_acc = max(history.history["val_accuracy"])
print(best_acc*100)

# make predictions on the testing set
predIdxs = model.predict(testX, batch_size=batch_size)
# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

def plot_Confusion_Matrix(actual_labels,predict_labels,title):
    """This function plot the confusion matrix"""
    # Reference : https://seaborn.pydata.org/generated/seaborn.heatmap.html
    cm = confusion_matrix(actual_labels, predict_labels)
    classNames = words
    cm_data = pd.DataFrame(cm,index = classNames, columns = classNames)
    plt.figure(figsize = (8,8))
    sns.heatmap(cm_data, annot=True,fmt="d")
    plt.title(title)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

from sklearn.metrics import confusion_matrix

plot_Confusion_Matrix(testY.argmax(axis=1), predIdxs,"Confusion Matrix")
