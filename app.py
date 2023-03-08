import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np



app=Flask(__name__)

# loading our model
#sys.path.append(r'path/to/python module file')
#with open('regmodel.pkl','rb') as fs:
  #  regmodel=pickle.load(fs)

new_model=open('regmodel.pkl','rb')

regmodel=pickle.load(new_model)
#regmodel=load('reg.joblib')
scalar=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',method=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())[0]).reshape(1,-1))
    new_data=scalar.transform(np.array(list(data.value())).reshape(1,-1))
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

if __name__=="__main__":
    app.run(debug=True)