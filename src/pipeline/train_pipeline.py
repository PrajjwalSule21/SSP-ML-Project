import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.data_ingestion import DataIngestion



def training_pipeline():
    try:
        
        obj = DataIngestion()
        logging.info('Data Ingestion has started')
        train_data, test_data = obj.initiate_data_ingestion()
        logging.info('Get the train and test data')

        data_transformation = DataTransformation()
        logging.info('Data transformation has started')
        train_array, test_array, _ = data_transformation.initiate_data_transformation(train_data, test_data)
        logging.info('Get the train and test array for model building')



        modeltrainer = ModelTrainer()
        logging.info('Model Trainer has started')
        r2_score_model = modeltrainer.initiate_model_trainer(train_array=train_array, test_array=test_array)
        logging.info(f'Get the Accuray Score: {r2_score_model} of Best Model')
        

        

    except Exception as e:
            raise CustomException(e, sys)
    

if __name__ == "__main__":
     training_pipeline()


