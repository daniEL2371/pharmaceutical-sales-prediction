# pharmaceutical-sales-prediction

**Table of content**

- [Overview](##abstract)
- [Requirements](#setup)
- [Install](#install)
- [Model Tracking]()

- [Features](#features)
  - [Data](#data.csv)
- [Notebooks](#notebooks)

  - [Multi variable sales prediction](notebooks/sales_prediction_model.ipynb)
  - [Time Series Sales Prediction](notebooks/LSTM_time_series_pred.ipynb)
  - [Data exploration](notebooks/data_exploration.ipynb)

- [Models](#Models)
- [Scripts](#scripts)
  - [Utility helper functions](scripts/helper.py)
  - [Logger](scripts/app_logger.py)
  - [Plotting graph](scripts/plots.py)
  - [Decision Tree Model](scripts/decisionTreesModel.py)
  - [Logistic Regression Model](scripts/logesticRegressionModel.py)
  - [Gradient Boosting Model](scripts/xGBClassifierModel.py)
  - [Decision Tree Model hyperparameter tuning](scripts/dt_tune_train.py)
  - [Logistic Regression Model hyperparameter tuning](scripts/logesticRegressionModel.py)
  - [Gradient Boosting Model hyperparameter tuning](scripts/xbg_tune_train.py)

## Overview

the finance team of Rossmann wants to forecast sales in all their stores across several cities six weeks ahead of time. Managers in individual stores rely on their years of experience as well as their personal judgement to forecast sales.
The data team identified factors such as promotions, competition, school and state holidays, seasonality, and locality as necessary for predicting the sales across the various stores.
The task is to build and serve an end-to-end product that delivers this prediction to analysts in the finance team.

## Requirements

Python 3.5 and above, Pip and MYSQL

## Install

```
git clone https://github.com/daniEL2371/pharmaceutical-sales-prediction
cd pharmaceutical-sales-prediction
pip install -r requirements.txt
```

## Model tracking

```
cd notebooks
mlflow ui
```

## Features

### Data

- train.csv: This is a dataset that holds data of sales at Rossman stores. It contains sale information from 2013 to 2015. There are 1017209 sales data in this dataset
- test.csv: This dataset holds test to check performance model
- store.csv: This dataset holds information about each stores. there are 1115 stores in this dataset

## Notebooks

### Data exploration

- Date Exploration is demonstrated notebooks/data_exploration.ipynb
- some insights about the data are described
- Daily, yearly, monthly sale trends are analysed
- Sesionality of sales are analyzed

### Multi variable sales prediction (notebooks/sales_prediction_model.ipynb)

- A machine learning approach to implement multi variable sales prediction implemented inside notebooks/sales_prediction_model.ipynb
- Date preprocessing from regression models.
- A random forest model and linear regression model is implmented using pythons sklearn pipeline to create our model.
- Feature importance calculation
- the resulting 2 models are serilzed and stored in Models folder

### Time Series Sales Prediction using deep learning

- Historical forecasting is a quick way to gather insights based on past sales performance. The idea is to look up sales from a past sale times series sales data and and predict future sales based on that.

- A deep learning approach to forcast future sales based on historical past data
- The notebook demonstrates how to apply time seris forcasting using LSTM networks
- The resulting model is then saved to the models folder

## Models

All models that are trained are saved inside models folder

## Scripts

- Utility helper functions is implemented in helper.py module
- Logger class for the project is implemented in app_logger.py module
- Plotting graphs like scatter plot, histogram, distribution graph, heat map, bar plot, and count plot is is implemented in scripts/plots.py module
- cleaner for Train data is implemented in cleanTrainData.py
- cleaner for Store data is implemented in cleanStoreDf.py
