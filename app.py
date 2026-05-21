from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pickle 
import pandas as pd

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
    
    input_dist=data.json()
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
        value=input_dist[col]
        if value not in encoder.classes_:
            return {
                "error":f"unkowen Category {value} for coulmn {col}"
            }
        input_dist[col]=encoder.transform([input_dist[col]])[0]       
   
    input_df=pd.DataFrame([input_dist])
    prediction=model.predict(input_df)    
      
    if(prediction[0]==1):
          return "Autism Detected"  
    else:
          return "No Autism detected"      