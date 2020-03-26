import digit_recognition
from flask import Flask, render_template, redirect, request

app = Flask(__name__)

@app.route('/')
def home():
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
        arr_X = np.array(li_X)
        arr_X = np.reshape(arr_X, (1,28,28,1))
        ans = classifier.predict(arr_X)
        ans = np.argmax(ans, axis = 1)
        return str(ans)
    return "ok"

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)