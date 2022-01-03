from flask import Flask,request, url_for, redirect, render_template, jsonify
#from pycaret.regression import *
import pandas as pd
import pickle
import numpy as np
from sklearn import preprocessing
# Initalise the Flask app
app = Flask(__name__,template_folder='templates')

# Loads pre-trained model
#model = load_model('/app/model/model.sav')

file = open("app/model/model.sav",'rb')
model = pickle.load(file)

cols = ['age', 'sex', 'bmi', 'children', 'smoker']

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = cols)
    
    # convert smoker categorical column into numerical column
    smoker_codes = {'no': 0, 'yes': 1}
    data_unseen['smoker_code'] = data_unseen.smoker.map(smoker_codes)
    
    sex_codes = {'female': 0, 'male': 1}
    data_unseen['sex_code'] = data_unseen.sex.map(sex_codes)
    #prediction = predict_model(model, data=data_unseen, round = 0)
    print("sex ",data_unseen['sex_code'])
    print("smoker ",data_unseen['smoker_code'])
    print("age ",data_unseen['age'])
    print("bmi ",data_unseen['bmi'])
    print("children ",data_unseen['children'])
    
    
    
    # enc = preprocessing.OneHotEncoder()
    # enc.fit(data_unseen[['region']])
    # one_hot = enc.transform(data_unseen[['region']]).toarray()
    # data_unseen[['northeast', 'northwest', 'southeast', 'southwest']] = one_hot
    
    prediction = model.predict(data_unseen[['age', 'bmi', 'children', 'smoker_code', 'sex_code']]) 
    #prediction = int(prediction.Label[0])
    return render_template('home.html',pred='Expected Bill will be {}'.format(prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    output = prediction.Label[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)

