import os
import sys
from dataclasses import dataclass

from sklearn import svm
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            model = svm.SVC(kernel = 'poly')

            svm_model = model.fit(X_train, y_train)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=svm_model
            )

            predicted=svm_model.predict(X_test)

            accuracy = accuracy_score(y_test, predicted)
            return accuracy

            
        except Exception as e:
            raise CustomException(e,sys)