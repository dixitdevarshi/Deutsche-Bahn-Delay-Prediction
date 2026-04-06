import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Load all artifacts
            preprocessor   = load_object(os.path.join('artifacts', 'preprocessor.pkl'))
            model          = load_object(os.path.join('artifacts', 'model.pkl'))
            threshold      = load_object(os.path.join('artifacts', 'threshold.pkl'))
            station_lookup = load_object(os.path.join('artifacts', 'station_lookup.pkl'))
            route_lookup   = load_object(os.path.join('artifacts', 'route_lookup.pkl'))
            station_stops  = load_object(os.path.join('artifacts', 'station_stops.pkl'))
            any_delay      = load_object(os.path.join('artifacts', 'any_delay_lookup.pkl'))
            quant_lookup   = load_object(os.path.join('artifacts', 'quantile_lookup.pkl'))
            logging.info("All artifacts loaded")

            # Extract raw inputs from dataframe
            station_name    = features['station_name'].values[0]
            destination     = features['destination'].values[0]
            train_category  = features['train_category'].values[0]
            day_of_week     = features['day_of_week'].values[0]
            hour_frac       = float(features['hour_frac'].values[0])
            is_construction = int(features['is_construction'].values[0])
            is_weekend      = int(features['is_weekend'].values[0])

            # Resolve station → lat, long, state
            GERMANY_CENTER = {'lat': 51.1657, 'long': 10.4515, 'state': 'Unknown'}
            if station_name in station_lookup:
                s              = station_lookup[station_name]
                lat, long, state = s['lat'], s['long'], s['state']
            else:
                lat   = GERMANY_CENTER['lat']
                long  = GERMANY_CENTER['long']
                state = GERMANY_CENTER['state']
                logging.info(f"Station '{station_name}' not found — using Germany centre")

            # Resolve route → num_stops
            route_key = (station_name, destination)
            if route_key in route_lookup:
                num_stops = route_lookup[route_key]
            elif station_name in station_stops:
                num_stops = station_stops[station_name]
            else:
                num_stops = 9  # overall median fallback
                logging.info(f"Route not found — using median fallback ({num_stops} stops)")

            # Build model input — same columns as training
            input_df = pd.DataFrame([{
                'state'          : state,
                'train_category' : train_category,
                'day_of_week'    : day_of_week,
                'hour'           : round(hour_frac),
                'num_stops'      : num_stops,
                'lat'            : lat,
                'long'           : long,
                'is_construction': is_construction,
                'is_disruption'  : 0,
                'has_info'       : 0,
                'is_weekend'     : is_weekend
            }])

            inp = preprocessor.transform(input_df)

            # Probabilities
            prob_6min = float(model.predict_proba(inp)[0, 1])

            # Any delay from historical lookup
            key      = (state, train_category)
            prob_any = float(any_delay.get(key, 0.35))

            # 15 min probability — use model if saved, else estimate
            try:
                model_15 = load_object(os.path.join('artifacts', 'classifier_15min.pkl'))
                thresh15 = load_object(os.path.join('artifacts', 'thresh_15min.pkl'))
                prob_15min = float(model_15.predict_proba(inp)[0, 1])
            except Exception:
                prob_15min = prob_6min * 0.3  # fallback estimate
            
            prob_15min = min(prob_15min, prob_6min)
            
            # Quantile range from lookup
            q_vals   = quant_lookup.get(key, {'q50': 9.0, 'q90': 21.0})
            q_median = q_vals['q50']
            q_90th   = q_vals['q90']

            # Risk label
            if prob_6min >= threshold:
                risk_label = 'HIGH RISK'
            elif prob_6min >= threshold * 0.6:
                risk_label = 'MODERATE RISK'
            else:
                risk_label = 'LOW RISK'

            logging.info(f"Prediction: {risk_label} | prob_6min={prob_6min:.1%}")

            return {
                'prob_any'  : round(prob_any   * 100, 1),
                'prob_6min' : round(prob_6min   * 100, 1),
                'prob_15min': round(prob_15min  * 100, 1),
                'q_median'  : round(q_median, 0),
                'q_90th'    : round(q_90th,   0),
                'risk_label': risk_label
            }

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 station_name: str,
                 destination: str,
                 train_category: str,
                 day_of_week: str,
                 hour: int,
                 minute: int = 0,
                 is_construction: int = 0):

        self.station_name    = station_name
        self.destination     = destination
        self.train_category  = train_category
        self.day_of_week     = day_of_week
        self.hour            = hour
        self.minute          = minute
        self.is_construction = is_construction

    def get_data_as_data_frame(self):
        try:
            return pd.DataFrame([{
                'station_name'   : self.station_name,
                'destination'    : self.destination,
                'train_category' : self.train_category,
                'day_of_week'    : self.day_of_week,
                'hour_frac'      : self.hour + self.minute / 60.0,
                'is_construction': self.is_construction,
                'is_weekend'     : 1 if self.day_of_week in ['Saturday', 'Sunday'] else 0
            }])
        except Exception as e:
            raise CustomException(e, sys)