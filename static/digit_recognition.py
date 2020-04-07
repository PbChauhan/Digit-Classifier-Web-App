import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

y = train_data.iloc[:,0].values
X = train_data.iloc[:,1:].values

X = X/255
X = np.reshape(X, (len(X[:,0]), 28, 28))
X2 = np.zeros((X.shape[0], 28*4, 28*4))
for i in range(X.shape[0]):
    plt.imsave("img.png", X[i,:,:])
    cv2_x = cv2.imread("img.png", flags = cv2.IMREAD_GRAYSCALE)
    cv2_x = cv2.resize(cv2_x, (28*4,28*4))
    X2[i,:] = cv2_x
plt.imshow(X2[1000,:,:])
X2 = np.reshape(X2, (X2.shape[0],28*4,28*4,1))
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rotation_range=10,
                             width_shift_range=0.1,
                             height_shift_range=0.2,
                             horizontal_flip=True,
                             zoom_range=0.1)
datagen.fit(X2)    


# one hot encoding y_train
from sklearn.preprocessing import OneHotEncoder
y = np.reshape(y, (len(y),1))
encoder = OneHotEncoder(n_values = 10,categorical_features = [0])
y = encoder.fit_transform(y)
y = y.toarray()



#import tensorflow as tf
#tf.config.list_physical_devices('GPU')
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten

classifier = Sequential()

classifier.add(Conv2D(32, kernel_size = (3,3), activation = 'relu', input_shape = (28*4, 28*4, 1)))

classifier.add(MaxPool2D((2,2)))

classifier.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))

classifier.add(MaxPool2D((2,2)))

classifier.add(Flatten())

classifier.add(Dense(128, activation = 'relu'))

classifier.add(Dense(128, activation = 'relu'))

classifier.add(Dense(10,activation = 'softmax'))

classifier.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

classifier.fit(datagen.flow(X2, y, batch_size=32), epochs = 20)

y_pred = classifier.predict(X_test)

y_pred = np.argmax(y_pred, axis = 1)


from sklearn.metrics import confusion_matrix
cm  = confusion_matrix(y_val,y_pred)
accuracy = sum([cm[i][i] for i in range(10)])/sum(sum(cm)) * 100

X_test = test_data.iloc[:,:].values
X_test = np.reshape(X_test, (len(X_test[:,0]), 28, 28, 1))
y_test_pred = classifier.predict(X_test)
y_test_pred = np.argmax(y_test_pred, axis = 1)

#Visualising the data
while True:
    i = int(input("Enter the image number.."))
    if i<=0 or i>=28000:
        break
    plt.imshow(X_test[i+1,:,:,0])
    print("-- Image --")
    plt.show()
    print("Predicted Value-",y_test_pred[i+1])
    

#Saving the model for deployment
model_json = classifier.to_json()
with open("model.json",'w') as md:
    md.write(model_json)

classifier.save_weights("model.h5")