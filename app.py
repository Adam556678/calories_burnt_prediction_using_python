import numpy as np
from flask import Flask, request, render_template
import pickle
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import json

app = Flask(__name__)

# load ANN model
model = pickle.load(open("ann_model.pkl", "rb"))

# load standard scaler
scaler = pickle.load(open("scaler.pkl", "rb"))

# load boxcox trained lambdas
with open("lambda.json", "r") as f:
    boxcox_lambda = json.load(f)

@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # get form values
        age = float(request.form['Age'])
        weight = float(request.form['Weight'])
        heart_rate = float(request.form['Heart_Rate'])
        body_temp = float(request.form['Body_Temp'])

        if body_temp <= 0:
            return "Body temperature must be positive for Box-Cox transformation."
        
        body_temp_transformed = boxcox(body_temp + 1, boxcox_lambda["Body_Temp"])    

        # convert inputs to np array and scale them 
        input_features = np.array([[age, weight, heart_rate, body_temp_transformed]])
        input_scaled = scaler.transform(input_features)

        # predict
        prediction = model.predict(input_scaled)
        
        # inverse predicition
        prediction_inv = inv_boxcox(prediction, boxcox_lambda['Calories'])
        prediction_inv = float(prediction_inv.flatten()[0])

        return render_template("index.html", prediction=prediction_inv) 
    
    return render_template("index.html")

if __name__=="__main__":
    app.run(debug=True)