# Autism Prediction API using Machine Learning

A complete Machine Learning project for predicting Autism Spectrum Disorder (ASD) using multiple classification algorithms and a deployed FastAPI backend.

This project includes:
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature encoding
- Outlier handling
- SMOTE balancing
- Model training and tuning
- FastAPI deployment-ready backend

---

# Project Overview

The goal of this project is to predict whether a person shows signs of Autism Spectrum Disorder based on screening test scores and demographic information.

The project compares multiple machine learning models and selects the best-performing model for deployment.

---

# Features

- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Label Encoding for Categorical Features
- Outlier Detection & Treatment
- SMOTE for Imbalanced Dataset Handling
- Hyperparameter Tuning using RandomizedSearchCV
- Multiple ML Models:
  - Decision Tree
  - Random Forest
  - XGBoost
- FastAPI Backend
- Deployment Ready
- REST API for Predictions

---

# Tech Stack

## Machine Learning
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn

## Visualization
- Matplotlib
- Seaborn

## Backend
- FastAPI
- Uvicorn

---

# Project Structure

```bash
autism_ml_model/
│
├── app.py                # FastAPI backend
├── best_model.pkl        # Trained ML model
├── encoders.pkl          # Saved label encoders
├── requirements.txt      # Project dependencies
├── train.csv             # Dataset
├── README.md
└── .gitignore
