from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('FinalML_LinearRegression.pkl', 'rb'))


@app.route('/')
def hello_world():  # put application's code here
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    default = 0
    memory = float(request.form.get('memory', default))
    disk = float(request.form.get('disk', default))
    cpu = float(request.form.get('cpu', default))
    time = float(request.form.get('time', default))
    inputs = np.array([[memory, disk, cpu]])

    result = model.predict(inputs)
    price_unit_min = 0.002581645946459407

    predict_cost = price_unit_min * result * time
    # predict_cost = predict_cost.astype(int)
    return render_template('index.html', data=predict_cost)


if __name__ == '__main__':
    app.run(debug=True)
