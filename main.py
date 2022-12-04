import pandas as pd
# from flask import *
from flask import Flask, render_template, request, url_for,redirect
import pickle
import numpy as np


app=Flask('__name__')
heart=pd.read_csv("heart_disease_data.csv")
model=pickle.load(open("logisticregressionmodel.pkl",'rb'))





@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/')
def index():
    return redirect('/home')


@app.route('/predict', methods=['POST'])
def predict():
    age = request.form.get('age')
    sex = request.form.get('sex')
    cp = request.form.get('cp')
    trestbps = request.form.get('trestbps')
    chol = request.form.get('chol')
    fbs = request.form.get('fbs')
    restecg = request.form.get('restecg')
    thalach = request.form.get('thalach')
    exang = request.form.get('exang')
    oldpeak = request.form.get('oldpeak')
    slope = request.form.get('slope')
    ca = request.form.get('ca')
    thal = request.form.get('thal')



    prediction = model.predict(
        pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
                     columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak',
                              'slope', 'ca', 'thal']))

    print(prediction[0])

    if (prediction[0]==0):
        st=" 0: The person does not have Heart Diesease"
        return st
    else:
        s=" 1: The person has Heart Diesease"
        return s



if __name__ =="__main__":
    app.run(debug=True)

# age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
