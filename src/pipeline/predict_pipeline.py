import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            # model_path=os.path.join("artifacts","model.pkl")
            model_path=("artifacts/model.pkl")
            # preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            preprocessor_path=("artifacts/proprocessor.pkl")
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
    def __init__(  self,
        Pregnancies: int,
        Glucose: int,
        BloodPressure: int,
        SkinThickness: int,
        Insulin: int,
        BMI: float,
        DiabetesPedigreeFunction: float,
        Age: int):

        self.pregnancy = Pregnancies

        self.glucose = Glucose

        self.bloodpressure = BloodPressure

        self.skinthickness = SkinThickness

        self.insulin = Insulin

        self.bmi = BMI

        self.diabetespedigreefunction = DiabetesPedigreeFunction

        self.age = Age

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Pregnancies": [self.pregnancy],
                "Glucose": [self.glucose],
                "BloodPressure": [self.bloodpressure],
                "SkinThickness": [self.skinthickness],
                "Insulin": [self.insulin],
                "BMI": [self.bmi],
                "DiabetesPedigreeFunction": [self.diabetespedigreefunction],
                "Age": [self.age],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)