from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            station_name   = request.form.get('station_name'),
            destination    = request.form.get('destination'),
            train_category = request.form.get('train_category'),
            day_of_week    = request.form.get('day_of_week'),
            hour           = int(request.form.get('hour')),
            minute         = int(request.form.get('minute', 0)),
            is_construction= int(request.form.get('is_construction', 0))
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")

        results = predict_pipeline.predict(pred_df)
        print("After Prediction")

        return render_template('home.html',
                               prob_any   = results['prob_any'],
                               prob_6min  = results['prob_6min'],
                               prob_15min = results['prob_15min'],
                               q_median   = results['q_median'],
                               q_90th     = results['q_90th'],
                               risk_label = results['risk_label'])


if __name__ == "__main__":
    app.run(host="0.0.0.0")