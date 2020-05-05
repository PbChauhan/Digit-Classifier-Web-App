import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

y = train_data.iloc[:,0].values
X = train_data.iloc[:,1:].values
X_test = test_data.iloc[:,:].values

X = X/255
X_test = X_test/255
X = np.reshape(X, (len(X[:,0]), 28, 28))
X_test = np.reshape(X_test, (X_test.shape[0],28,28))

x = 1 #resizing factor
 
X = np.reshape(X, (X.shape[0],28*x,28*x,1))
X_test = np.reshape(X_test, (X_test.shape[0],28*x,28*x,1))


# one hot encoding y_train
from sklearn.preprocessing import OneHotEncoder
y = np.reshape(y, (len(y),1))
encoder = OneHotEncoder(n_values = 10,categorical_features = [0])
y = encoder.fit_transform(y)
y = y.toarray()

#from sklearn.model_selection import train_test_split
#X_train, X_val, y_train, y_val = train_test_split(X,y,test_size = 0.2)

from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rotation_range=0,
                             width_shift_range=0,
                             height_shift_range=0,
                             horizontal_flip=False,
                             zoom_range=0.1,
                             validation_split=0.2)
datagen.fit(X)

'''
import tensorflow as tf
tf.config.list_physical_devices('GPU')
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
'''

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers import Dropout
classifier = Sequential()

classifier.add(Conv2D(32, kernel_size = (3,3), padding = 'same', activation = 'relu', input_shape = (28*x, 28*x, 1)))

classifier.add(Dropout(0.5))

classifier.add(BatchNormalization())

classifier.add(MaxPool2D((2,2)))

classifier.add(Conv2D(64, kernel_size = (3,3), padding = 'same', activation = 'relu'))

classifier.add(Dropout(0.5))

classifier.add(BatchNormalization())

classifier.add(MaxPool2D((2,2)))

classifier.add(Flatten())

classifier.add(Dense(256, activation = 'relu'))

classifier.add(Dense(10,activation = 'softmax'))

classifier.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

classifier.summary()


import tensorflow as tf
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', patience = 5, min_lr = 1e-4, verbose = 1)
with tf.device('/gpu:0'):
    classifier.fit(datagen.flow(X, y,subset="training"),
                   validation_data = datagen.flow(X,y,subset="validation"),
                   callbacks = [lr_callback],verbose = 1, epochs = 20)

y_pred = classifier.predict(X_test)

y_pred = np.argmax(y_pred, axis = 1)


#Visualising the data
while True:
    i = int(input("Enter the image number.."))
    if i<=0 or i>=28000:
        break
    plt.imshow(X_test[i+1,:,:,0])
    print("-- Image --")
    plt.show()
    print("Predicted Value-",y_pred[i+1])
    

#Saving the model for deployment
model_json = classifier.to_json()
with open("model.json",'w') as md:
    md.write(model_json)

classifier.save_weights("model.h5")
