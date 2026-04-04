import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path:  str = os.path.join('artifacts', "test.csv")
    raw_data_path:   str = os.path.join('artifacts', "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            
            df = pd.read_csv(os.path.join('src', 'notebook', 'data', 'DBdata_processed.csv'))
            logging.info('Read the DB7 dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            
            # We train on earlier dates and test on later dates
            logging.info("Time-based train test split initiated")

            dates_sorted = sorted(df['date'].unique())
            split_idx    = int(len(dates_sorted) * 0.75)
            train_dates  = dates_sorted[:split_idx]
            test_dates   = dates_sorted[split_idx:]

            train_set = df[df['date'].isin(train_dates)]
            test_set  = df[df['date'].isin(test_dates)]

            logging.info(f"Train dates : {train_dates}")
            logging.info(f"Test dates  : {test_dates}")
            logging.info(f"Train size  : {len(train_set):,} rows")
            logging.info(f"Test size   : {len(test_set):,} rows")
            # ────────────────────────────────────────────────────────────

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,   index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)
    
    
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))