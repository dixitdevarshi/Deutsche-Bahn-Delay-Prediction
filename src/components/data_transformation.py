import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            cat_features    = ['state', 'train_category', 'day_of_week']
            num_features    = ['hour', 'num_stops', 'lat', 'long']
            binary_features = ['is_construction', 'is_disruption', 'has_info', 'is_weekend']

            # Numerical pipeline — just scale, no imputer needed (already cleaned)
            num_pipeline = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])

            # Categorical pipeline — OHE only
            cat_pipeline = Pipeline(steps=[
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])

            preprocessor = ColumnTransformer([
                ('ohe',    cat_pipeline, cat_features),
                ('scaler', num_pipeline, num_features),
                ('pass',   'passthrough', binary_features)
            ])

            logging.info(f'Categorical features: {cat_features}')
            logging.info(f'Numerical features  : {num_features}')
            logging.info(f'Binary features     : {binary_features}')

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)
            logging.info('Read train and test data completed')

            preprocessing_obj = self.get_data_transformer_object()
            logging.info('Preprocessing object obtained')

            # Target and feature columns
            target_column    = 'dep_delayed'
            drop_columns     = [target_column, 'departure_delay_m', 'date', 'category', 'stop_bucket']

            # Drop only columns that exist
            drop_train = [c for c in drop_columns if c in train_df.columns]
            drop_test  = [c for c in drop_columns if c in test_df.columns]

            X_train = train_df.drop(columns=drop_train)
            y_train = train_df[target_column]

            X_test  = test_df.drop(columns=drop_test)
            y_test  = test_df[target_column]

            logging.info('Applying preprocessor on train and test data')

            X_train_arr = preprocessing_obj.fit_transform(X_train)
            X_test_arr  = preprocessing_obj.transform(X_test)

            # Combine features and target into single arrays
            train_arr = np.c_[X_train_arr, np.array(y_train)]
            test_arr  = np.c_[X_test_arr,  np.array(y_test)]

            logging.info('Saving preprocessor object')
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)