import numpy as np
import pandas as pd
import pickle
import json
# import os
import config

class Diabetes():
    def __init__(self,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age):
        self.Glucose = Glucose
        self.BloodPressure =BloodPressure
        self.SkinThickness = SkinThickness
        self.Insulin = Insulin
        self.BMI = BMI
        self.DiabetesPedigreeFunction = DiabetesPedigreeFunction
        self.Age = Age

    def load_model(self):
        with open(config.MODEL_FILE_PATH, "rb") as f:
            self.KNN_model1=pickle.load(f)
        with open(config.SCALING_FILE_PATH,'rb') as f:
            self.scalling_model = pickle.load(f)
        with open(config.JSON_FILE_PATH,"r") as f:
            self.json_data = json.load(f)

    def predict_diabetes(self):
        self.load_model()
         
        test_array=np.zeros(len(self.json_data["columns"]))
        test_array[0]=self.Glucose
        test_array[1]=self.BloodPressure
        test_array[2]=self.SkinThickness
        test_array[3]=self.Insulin
        test_array[4]=self.BMI
        test_array[5]=self.DiabetesPedigreeFunction
        test_array[6]=self.Age
        print("test_array is: ",test_array)
        # array([[Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        scale_test_array=self.scalling_model.transform([test_array])
        prediction=self.KNN_model1.predict(scale_test_array)[0]
        return prediction
if __name__=="__main__":
    Glucose= 85.000
    BloodPressure=66.000
    SkinThickness=29.000
    Insulin=0.000
    BMI=26.600
    DiabetesPedigreeFunction=0.351
    Age=31.000

    obj=Diabetes(Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age)
    obj.predict_diabetes()



        