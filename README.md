A binary classification project that predicts whether it will rain on a given day using a Random Forest Classifier trained on weather data.

Dataset
The dataset (Rainfall.csv) contains daily weather measurements. After preprocessing, the following features are used:
pressure, dewpoint, humidity, cloud, sunshine, winddirection, windspeed
Target: rainfall — 1 (Yes) or 0 (No)

Workflow

Data Cleaning — handled missing values, dropped irrelevant and highly correlated columns
EDA — distribution plots, correlation heatmap, boxplots
Class Balancing — downsampled majority class to fix imbalance
Model Training — Random Forest with GridSearchCV hyperparameter tuning (5-fold CV)
Evaluation — accuracy, confusion matrix, classification report
Export — model saved as rainfall_prediction_model.pkl

Quick Prediction

import pickle, pandas as pd
with open("rainfall_prediction_model.pkl", "rb") as f:
    model_data = pickle.load(f)
model = model_data["model"]
feature_names = model_data["feature_names"]
input_data = (1015.9, 19.9, 95, 81, 0.0, 40.0, 13.7)
input_df = pd.DataFrame([input_data], columns=feature_names)
prediction = model.predict(input_df)
print("Rainfall" if prediction[0] == 1 else "No Rainfall")

Tech Stack
Python · pandas · scikit-learn · matplotlib · seaborn · Google Colab
