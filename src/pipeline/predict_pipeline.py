import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
        
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self,
        Item_Fat_Content: str,
        Item_Type: str,
        Outlet_Location_Type: str,
        Outlet_Type: str,
        Outlet_Size: str,
        Item_Visibility: int,
        Item_MRP: int,
        Age_Outlet: int,
        Item_Weight: int,
    ):

        self.Item_Fat_Content = Item_Fat_Content

        self.Item_Type = Item_Type

        self.Outlet_Location_Type = Outlet_Location_Type

        self.Outlet_Type = Outlet_Type

        self.Outlet_Size = Outlet_Size

        self.Item_Visibility = Item_Visibility

        self.Item_MRP = Item_MRP

        self.Age_Outlet = Age_Outlet

        self.Item_Weight = Item_Weight


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Item_Fat_Content": [self.Item_Fat_Content],
                "Item_Type": [self.Item_Type],
                "Outlet_Location_Type": [self.Outlet_Location_Type],
                "Outlet_Type": [self.Outlet_Type],
                "Outlet_Size": [self.Outlet_Size],
                "Item_Visibility": [self.Item_Visibility],
                "Item_MRP": [self.Item_MRP],
                "Age_Outlet": [self.Age_Outlet],
                "Item_Weight": [self.Item_Weight],
    
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
