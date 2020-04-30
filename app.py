from keras.models import model_from_json
from flask import Flask, render_template, redirect, request
import matplotlib.pyplot as plt
import numpy as np
import cv2
app = Flask(__name__)


json_file = open('model.json', 'r')
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
loaded_model.load_weights('model.h5')

@app.route('/', methods = ['GET','POST'])
def home():
    if request.method == "POST":
        y = request.form.get('val')
        print(y)
    return render_template('index.html')

@app.route('/predict', methods = ['GET', 'POST'])
def pred():
    if request.method == "POST":
        raw_X = request.get_json(force = True)
        raw_X = list(raw_X.split(','))
        li_X =[0]
        for i in raw_X:
            if i.isnumeric():
               li_X.append(i) 
        li_X.append(0)
        arr_X = np.reshape(np.array(li_X, dtype = 'float'), (1,28*4,28*4))[0,:,:]
        arr_X = cv2.resize(arr_X, (28*1,28*1))
        plt.imsave("./static/image/predict.png", arr_X)
        arr_X = np.reshape(arr_X, (1,28*1,28*1,1))
        ans = loaded_model.predict(arr_X)
        ans = np.argmax(ans, axis = 1)
        return str(ans)
    return redirect('/')

app.run(debug=True, use_reloader=False, threaded = False)
    

