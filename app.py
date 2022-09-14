# Importing essential libraries
from array import array
from unicodedata import name
from flask import Flask, render_template, request
import pickle
import numpy as np
import csv
import numpy
import pandas as pd


# Load the Random Forest CLassifier model
filename = 'heart-disease-prediction-knn-model.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

# a = numpy.array([[1,4,2],[7,9,4],[0,6,2]])

# with open('userdata.csv', 'w', newline='') as file:
#     mywriter = csv.writer(file, delimiter=',')
#     mywriter.writerows(a)

@app.route('/')
def home():
	return render_template('main.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        name= request.form.get("name")
        print(name)
        age = int(request.form['age'])
        sex = request.form.get('sex')
        cp = request.form.get('cp')
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = request.form.get('fbs')
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach']) 
        exang = request.form.get('exang')
        oldpeak = float(request.form['oldpeak'])
        slope = request.form.get('slope')
        ca = int(request.form['ca'])
        thal = request.form.get('thal')
        
        
        data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
        print(data)
        my_prediction = model.predict(data)
        data_count =0
        
        with open(r'history.csv','a', newline ='') as csvfile:
            fieldnames = ['name','age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','prediction']
           
            thewriter = csv.DictWriter(csvfile,fieldnames=fieldnames)
            thewriter.writeheader()

            for fieldnames in data :
                data_count += 1
            thewriter.writerow({'name':name,'age':age,'sex':sex,'cp':cp,'trestbps':trestbps,'chol':chol,'fbs':fbs,'restecg':restecg,'thalach':thalach,'exang':exang,'oldpeak':oldpeak ,'slope':slope,'ca':ca,'thal':thal,'prediction':my_prediction})
        
        
        return render_template('result.html', prediction=my_prediction)

       
        

if __name__ == '__main__':
	app.run(debug=True)


