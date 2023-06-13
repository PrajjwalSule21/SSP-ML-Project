import sys
import os
from datetime import datetime
from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.utils import save_object

@dataclass
class DataTransformationConfig: 
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()


    def column_transformer(self, train_df, test_df):
        try:
            logging.info('Tranforming the Age_Outlet column')
            train_df['Age_Outlet'] = datetime.now().year - train_df['Outlet_Establishment_Year']
            train_df.drop(columns=['Outlet_Establishment_Year', 'Outlet_Identifier', 'Item_Identifier'], inplace=True)

            test_df['Age_Outlet'] = datetime.now().year - test_df['Outlet_Establishment_Year']
            test_df.drop(columns=['Outlet_Establishment_Year', 'Outlet_Identifier', 'Item_Identifier'], inplace=True)

            logging.info('Tranforming the Item_Fat_Content column')
            train_df['Item_Fat_Content'] = train_df['Item_Fat_Content'].str.replace('LF','Low Fat')
            train_df['Item_Fat_Content'] = train_df['Item_Fat_Content'].str.replace('low fat','Low Fat')
            train_df['Item_Fat_Content'] = train_df['Item_Fat_Content'].str.replace('reg','Regular')

            test_df['Item_Fat_Content'] = test_df['Item_Fat_Content'].str.replace('LF','Low Fat')
            test_df['Item_Fat_Content'] = test_df['Item_Fat_Content'].str.replace('low fat','Low Fat')
            test_df['Item_Fat_Content'] = test_df['Item_Fat_Content'].str.replace('reg','Regular')

            return (train_df, test_df)
        
        except Exception as e:
            raise CustomException(e, sys)
        
        
    # def outlier_removal(self, train_df, test_df):
    #     try:
    #         logging.info('Remove the outlier from Item_Visibility column')
    #         train_percentile25 = train_df['Item_Visibility'].quantile(0.25)
    #         train_percentile75 = train_df['Item_Visibility'].quantile(0.75)
            
    #         train_IQR = train_percentile75 - train_percentile25
    #         train_upper_limit = train_percentile75 + 1.5 * train_IQR
    #         train_lower_limit = train_percentile25 - 1.5 * train_IQR
            

    #         train_df['Item_Visibility'] = np.where(
    #             train_df['Item_Visibility'] > train_upper_limit,
    #             train_upper_limit,
    #             np.where(
    #                 train_df['Item_Visibility'] < train_lower_limit,
    #                 train_lower_limit,
    #                 train_lower_limit['Item_Visibility']
    #             )
    #         )

    #         test_percentile25 = test_df['Item_Visibility'].quantile(0.25)
    #         test_percentile75 = test_df['Item_Visibility'].quantile(0.75)
            
    #         test_IQR = test_percentile75 - test_percentile25
    #         test_upper_limit = test_percentile75 + 1.5 * test_IQR
    #         test_lower_limit = test_percentile25 - 1.5 * test_IQR


    #         train_df['Item_Visibility'] = np.where(
    #             test_df['Item_Visibility'] > test_upper_limit,
    #             test_upper_limit,
    #             np.where(
    #                 test_df['Item_Visibility'] < test_lower_limit,
    #                 test_lower_limit,
    #                 test_lower_limit['Item_Visibility']
    #             )
    #         )
    
    #         return (
    #             train_df, 
    #             test_df ) 
    #     except Exception as e:
    #         raise CustomException(e, sys)



    def get_data_transformer_object(self):
        '''
        This function is responsible for data trnasformation
        
        '''
        try:
            categorical_columns = ['Item_Fat_Content', 'Item_Type', 'Outlet_Location_Type', 'Outlet_Type', 'Outlet_Size']
            numerical_columns = ['Item_Visibility', 'Item_MRP', 'Age_Outlet', 'Item_Weight']
            

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(
            
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("ordinal_encoder",OrdinalEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline, categorical_columns)

                ], remainder='passthrough'

            )

            logging.info('Pipeline for numerical and categorical completed')
            

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            logging.info('Apply column tranfromer on train and test data')
            train_df, test_df = self.column_transformer(train_df=train_df, test_df=test_df)

            # logging.info("Apply outlier removal on train and test data")
            # train_df, test_df = self.outlier_removal(train_df=train_df, test_df=test_df)


            preprocessing_obj=self.get_data_transformer_object()
            logging.info("Obtaining preprocessing object")

            target_column_name="Item_Outlet_Sales"

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            # print(len(input_feature_train_arr[0]))

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

            
        except Exception as e:
            raise CustomException(e,sys)