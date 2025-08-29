from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd


app = Flask(__name__)
model = joblib.load('used_car_price_model.pkl')
model_columns = joblib.load('model_columns.pkl')


@app.route('/') 
def home(): 
    return render_template('index.html')



@app.route('/predict', methods=['POST'])
def predict():

    input_data = request.form.to_dict() 
    
    input_df = pd.DataFrame([input_data])

    input_df['odometer'] = float(input_df['odometer'])
    input_df['vehicle_age'] = int(input_df['vehicle_age'])

    input_encoded = pd.get_dummies(input_df, columns=['manufacturer', 'condition', 'fuel', 'transmission', 'drive', 'paint_color', 'state'],
                        prefix='', prefix_sep='')

    columns_to_drop = ['nd', 'other', 'custom', 'other', 'other', 'rwd', 'salvage']
    input_encoded.drop(columns=[col for col in columns_to_drop if col in input_encoded.columns], inplace=True)

    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_encoded)[0]

    return render_template('index.html', prediction_text=f'Predicted Car Price: ${round(prediction, 2)}')



if __name__ == '__main__':
    app.run(debug=True)  