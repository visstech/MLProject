from flask import Flask,request,jsonify,render_template 
import pickle
import pandas as pd 
from sklearn.preprocessing import StandardScaler
import seaborn as sns 
import matplotlib.pyplot as plt 

app = Flask(__name__)

ElasticNet = pickle.load(open('C:\\ML\MLProject\\ElasticNet_reg.pkl','rb'))
scaler  = pickle.load(open('C:\\ML\MLProject\\scaler.pkl','rb'))

@app.route('/')
def index():
   return render_template('index.html')

@app.route('/predictData',methods=['GET','POST'])
def predict_datapoint():
    print('request method is:',request.method)
    if request.method =='POST':
      Temperature = float(request.form.get('Temperature'))
      RH = float(request.form.get('RH'))
      WS = float(request.form.get('WS'))
      Rain = float(request.form.get('Rain'))
      FFMC = float(request.form.get('FFMC'))
      DMC = float(request.form.get('DMC'))
      ISI = float(request.form.get('ISI'))
      Classes = float(request.form.get('Classes'))
      Region = float(request.form.get('Region'))
      new_data_scaled = scaler.transform([[Temperature,RH,WS,Rain,FFMC,DMC,ISI,Classes,Region]])
      result = ElasticNet.predict(new_data_scaled)
      return render_template('Home.html',results=abs(result[0]))
    else:   
      print('Coming here in else') 
      return render_template('Home.html')     

if __name__ =='__main__' :
    print('Yes coming here')
    app.run(host="0.0.0.0")
 

