#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template
get_ipython().system('python -m  pipreqs.pipreqs . --force')
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # for rendering results on html GUI
    
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text='Employee Salary Should be $ {}'.format(output))

if __name__ == "__main__":
    app.run(debug=False)
    


# In[ ]:




