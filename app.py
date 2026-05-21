from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pickle 
import pandas as pd
import json

app=FastAPI()
origins=["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

with open("best_model.pkl",'rb')as f:
    model=pickle.load(f)

with open("encoders.pkl",'rb')as f:
    encoders=pickle.load(f)
    
class model_input(BaseModel):
    A1_Score:int
    A2_Score:int
    A3_Score:int
    A4_Score:int 
    A5_Score:int
    A6_Score:int
    A7_Score:int
    A8_Score:int
    A9_Score:int
    A10_Score:int
    age:int
    gender:str
    ethnicity:str
    jaundice:str
    austim:str
    contry_of_res:str
    used_app_before:str
    relation:str
    result:int
    
@app.get("/")
def home():
    return {"message":"Autism Prediction Api Running"}

@app.post("/prediction")
def predict_autism(data:model_input):
    
    input_data=data.json()
    input_data=json.loads(input_data)
    input_data['ethnicity']=(
        "Others"
        if input_data['ethnicity'] in ["?","others"]
        else input_data['ethnicity']
    )
    input_data['relation']=(
        "Others"
        if input_data['relation'] in [
            "?",
            "Relative",
            "Health care professional",
            "Parent"]
        else input_data['relation']
    )
    catgorical_colmns=[
        
    "gender",
    "ethnicity",
    "jaundice",
    "austim",
    "contry_of_res",
    "used_app_before",
    "relation",
    ] 
    
    for col in catgorical_colmns:
        encoder=encoders[col]
        value=input_data[col]
        if value not in encoder.classes_:
            return {
                "error":f"unkowen Category {value} for coulmn {col}"
            }
        input_data[col]=encoder.transform([input_data[col]])[0]       
   
    feature_order = [
    'A1_Score',
    'A2_Score',
    'A3_Score',
    'A4_Score',
    'A5_Score',
    'A6_Score',
    'A7_Score',
    'A8_Score',
    'A9_Score',
    'A10_Score',
    'age',
    'gender',
    'ethnicity',
    'jaundice',
    'austim',
    'contry_of_res',
    'used_app_before',
    'result',
    'relation'
     ]
    
    
    input_df=pd.DataFrame([input_data])[feature_order]
    prediction=model.predict(input_df)    
      
    if(prediction[0]==1):
          return "Autism Detected"  
    else:
          return "No Autism detected"      