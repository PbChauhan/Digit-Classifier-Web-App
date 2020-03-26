
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

y = train_data.iloc[:1300,0].values
X = train_data.iloc[:1300,1:].values

X = X/255
X = np.reshape(X, (len(X[:,0]), 28, 28, 1))

from sklearn.model_selection  import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.25)


# one hot encoding y_train
from sklearn.preprocessing import OneHotEncoder
y_train = np.reshape(y_train, (len(y_train),1))
encoder = OneHotEncoder(n_values = 10,categorical_features = [0])
y_train = encoder.fit_transform(y_train)
y_train = y_train.toarray()


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten

classifier = Sequential()

classifier.add(Conv2D(32, kernel_size = (3,3), activation = 'relu', input_shape = (28, 28, 1)))

classifier.add(MaxPool2D((2,2)))

classifier.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))

classifier.add(MaxPool2D((2,2)))

classifier.add(Flatten())

classifier.add(Dense(128, activation = 'relu'))

classifier.add(Dense(128, activation = 'relu'))

classifier.add(Dense(10,activation = 'softmax'))

classifier.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 64, epochs = 20)

y_pred = classifier.predict(X_val)

y_pred = np.argmax(y_pred, axis = 1)


from sklearn.metrics import confusion_matrix
cm  = confusion_matrix(y_val,y_pred)
accuracy = sum([cm[i][i] for i in range(10)])/sum(sum(cm)) * 100

X_test = test_data.iloc[:,:].values
X_test = np.reshape(X_test, (len(X_test[:,0]), 28, 28, 1))
y_test_pred = classifier.predict(X_test)
y_test_pred = np.argmax(y_test_pred, axis = 1)

#Visualising the data
Do = False
while Do:
    i = int(input("Enter the image number.."))
    if i<=0 or i>=28000:
        break
    plt.imshow(X_test[i+1,:,:,0])
    print("-- Image --")
    plt.show()
    print("Predicted Value-",y_test_pred[i+1])
    

