from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    area = float(request.form['area'])
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])

    input_data = np.array([[area, bedrooms, bathrooms]])
    prediction = model.predict(input_data)[0]

    return render_template('result.html', prediction=round(prediction, 2))

if __name__ == "__main__":
    app.run(debug=True)
