

import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV,cross_val_score,train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import pickle

df=pd.read_csv('./train (1).csv')

print(df.shape)
print(df.head())
print(df.tail())
print(df.isnull().sum())
print(df.describe())

df['age']=df['age'].astype(int)

for col in df.columns:
    numerical_feture=['ID','age','result']
    if col not in numerical_feture:
     print(col,df[col].unique())
     print("*"*50)
     
df=df.drop(columns=['ID','age_desc'])
print(df.head())

print(df['contry_of_res'].unique())


mapping={
    "Viet Nam":"VietNam",
    "AmericanSamoa":"United States",
    "Hong Kong":"China"
}
df['contry_of_res']=df['contry_of_res'].replace(mapping)
print(df['contry_of_res'].unique())

print(df['Class/ASD'].value_counts())

print(df.describe())

sns.set_theme(style='whitegrid')


sns.histplot(df['age'],kde=True)
plt.title("destribution of age")
age_mean=df['age'].mean()
age_mediam=df['age'].median()

plt.axvline(age_mean,color='red',linestyle="--",label='Mean')

plt.axvline(age_mediam,color='green',linestyle="-",label='Mediann')
plt.legend()
plt.show()

sns.histplot(df['result'],kde=True)
plt.title("destribution of result")
age_mean=df['result'].mean()
age_mediam=df['result'].median()

plt.axvline(age_mean,color='red',linestyle="--",label='Mean')

plt.axvline(age_mediam,color='green',linestyle="-",label='Mediann')
plt.legend()
plt.show()
# %%
sns.boxplot(x=df['age'])
plt.title('box plot for age')
plt.xlabel('age')
plt.show()

q1=df['age'].quantile(0.25)
q3=df['age'].quantile(0.75)
IQR=q3-q1
lower_bound=q1-1.5*IQR
uper_bound=q3+1.5*IQR
age_outliar=df[(df['age']<lower_bound) | (df['age']>uper_bound)]
print(len(age_outliar))

q1=df['result'].quantile(0.25)
q3=df['result'].quantile(0.75)
IQR=q3-q1
lower_bound=q1-1.5*IQR
uper_bound=q3+1.5*IQR
result_outliar=df[(df['result']<lower_bound) | (df['result']>uper_bound)]
print(len(result_outliar))

# catagorical coulmns
print(df.columns)

categorical_coulmns=['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score',
       'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'age', 'gender',
       'ethnicity', 'jaundice', 'austim', 'contry_of_res', 'used_app_before',
        'relation'
]

for col in categorical_coulmns:
    sns.countplot(x=df[col])
    plt.title(f"Cpunt plot for {col}")
    plt.xlabel(col)
    plt.ylabel("count")
   
    plt.show();
    
    sns.countplot(x=df["Class/ASD"])
    plt.title(f"Cpunt plot for Class/ASD")
    plt.xlabel("Class/ASD")
    plt.ylabel("count")
    plt.show()
    
df['ethnicity']=df['ethnicity'].replace({"?":"Others","others":"Others"})


print(df['relation'].unique())
# %%
df['relation']=df['relation'].replace({
    "?":"Others",
    "Relative":"Others",
    "Health care professional":"Others",
    "Relative":"Others",
    "Parent":"Others",
    })


object_coulmns=df.select_dtypes(include=['object']).columns
print(object_coulmns)

#to store enocders
encoders={}

for coulmn in object_coulmns:
    label_encoder=LabelEncoder()
    df[coulmn]=label_encoder.fit_transform(df[coulmn])
    encoders[coulmn]=label_encoder
    
    with open("encoders.pkl","wb")as f:
        pickle.dump(encoders,f)
# %%
print(df.head())
# %%
plt.figure(figsize=(15,15))
sns.heatmap(df.corr(),annot=True,cmap="coolwarm",fmt=".2f")
plt.title("Correlation Matrix")
plt.show()
# %%
def replace_outliar_with_mediam(df,coulmn):
    q1=df[coulmn].quantile(0.25)
    q3=df[coulmn].quantile(0.75)
    IQR=q3-q1
    lower_bound=q1-1.5*IQR
    upper_bound=q3+1.5*IQR
    median=df[coulmn].median()
    
    df[coulmn]=df[coulmn].apply(lambda x: median if x<lower_bound or x>upper_bound else x)
    return df
    
df=replace_outliar_with_mediam(df,"age")
df=replace_outliar_with_mediam(df,"result")

print(df.head())
print(df.shape)

x=df.drop(columns=["Class/ASD"])
y=df['Class/ASD']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print(x_train.shape)
print(y_train.shape)

smote=SMOTE()
x_train_smote,y_train_smote=smote.fit_resample(x_train,y_train)
print(x_train_smote.shape)
print(y_train_smote.shape)
print(y_train_smote.value_counts())

models={
    "Decision Tree":DecisionTreeClassifier(random_state=42),
    "Random Forest":RandomForestClassifier(random_state=42),
    "XGBoost":XGBClassifier(random_state=42)
}

cv_scores={}
#s-fold cross validation for each model
for model_name,model in models.items():
    print(f"raining {model_name} with default parmeter..")
    scores=cross_val_score(model,x_train_smote,y_train_smote,cv=5,scoring="accuracy")
    cv_scores[model_name]=scores
    print(f"{model_name} cross-validation accuracy:{np.mean(scores):.2f}")
    print("*"*50)    
# %%
    Decision_tree=DecisionTreeClassifier(random_state=42)
    Random_forest=RandomForestClassifier(random_state=42)
    xgBoost=XGBClassifier(random_state=42)
    
# %%
param_grid_dt={
   'criterion':["gini","entropy"],
   "max_depth":[None,5,10,15],
   "min_samples_split":[2,5,10],
   "min_samples_leaf":[1,2,4]     
    }
param_grid_rf={
    "n_estimators":[50,100,200,500],
    "max_depth":[None,10,20,30],
    "min_samples_split":[2,5,10],
    "min_samples_leaf":[1,2,4],
    "bootstrap":[True,False]
    
}

param_grid_xgb={
    "n_estimators":[50,100,200,500],
    "max_depth":[3,5,7,10],
    "learning_rate":[0.01,0.1,0.2,0.3],
    "subsample":[0.5,0.7,1.0],
    "colsample_bytree":[0.5,0.7,1.0]
}

random_search_dt=RandomizedSearchCV(estimator=Decision_tree,param_distributions=param_grid_dt,n_iter=20,cv=5,scoring="accuracy",random_state=42)
random_search_rf=RandomizedSearchCV(estimator=Random_forest,param_distributions=param_grid_rf,n_iter=20,cv=5,scoring="accuracy",random_state=42)
random_search_xgb=RandomizedSearchCV(estimator=xgBoost,param_distributions=param_grid_xgb,n_iter=20,cv=5,scoring="accuracy",random_state=42)

random_search_dt.fit(x_train_smote,y_train_smote)
random_search_rf.fit(x_train_smote,y_train_smote)
random_search_xgb.fit(x_train_smote,y_train_smote)


best_model=None
best_score=0

if(random_search_dt.best_score_>best_score):
    best_model=random_search_dt.best_estimator_
    best_score=random_search_dt.best_score_
    

if(random_search_rf.best_score_>best_score):
    best_model=random_search_rf.best_estimator_
    best_score=random_search_rf.best_score_
    
if(random_search_xgb.best_score_>best_score):
    best_model=random_search_xgb.best_estimator_
    best_score=random_search_xgb.best_score_    
    
print(best_model)
print(best_score)

with open("best_model.pkl","wb") as f:
       pickle.dump(best_model,f)

y_test_prediction=best_model.predict(x_test)
print("accuracy score:\n",accuracy_score(y_test,y_test_prediction))
print("classification report:\n",classification_report(y_test,y_test_prediction))
print("Confusion matrix:\n",confusion_matrix(y_test,y_test_prediction))

