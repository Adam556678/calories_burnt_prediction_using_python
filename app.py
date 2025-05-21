import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from scipy.special import boxcox1p

app = Flask(__name__)

model = pickle.load(open("xgb_model.pkl", "rb"))

@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        age = float(request.form['Age'])
        weight = float(request.form['Weight'])
        heart_rate = float(request.form['Heart_Rate'])
        body_temp = float(request.form['Body_Temp'])

        if body_temp < 0:
            return "Body temperature must be positive for Box-Cox transformation."
        
        
    
    return render_template("index.html")

if __name__=="__main__":
    app.run(debug=True)