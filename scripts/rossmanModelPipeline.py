import os
import sys
from IPython.display import Markdown, display, Image
import numpy as np
import pandas as pd
import random
import dvc.api
from app_logger import App_Logger
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from helper import Helper
from cleanTrainData import CleanTrainData
from cleanStoreDf import CleanStoreDf


def loss_function(actual, pred):
    mae = mean_absolute_error(actual, pred)
    return mae


class RossmanModelPipeline:

    def __init__(self, cleaned_rossman_data, model_name):

        self.X_train, self.X_test, self.y_train, self.y_test = self.prepare_data(
            cleaned_rossman_data)
        self.model_name = model_name

    def prepare_data(self, cleaned_rossman_data):
        feat_cols = ['DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday', 'Year', 'Open',
                     'Month', 'Day', 'Weekends', 'StoreType', 'Assortment', 'CompetitionDistance',
                     'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2',
                     'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval', "DayInMonth"]

        X = cleaned_rossman_data[feat_cols]
        y = cleaned_rossman_data["Sales"]
        return train_test_split(X, y, test_size=0.2)

    def Preproccessor(self):
        cols = self.X_train.columns
        numric_cols = ["CompetitionDistance", "Promo2SinceWeek", "Year"]
        categorical_cols = self.X_train.copy(deep=True).drop(
            columns=numric_cols, axis=1, inplace=False).columns.to_list()

        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler()),
                                              ('imputer', SimpleImputer(strategy='mean'))])
        categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                                  ('encoder', OrdinalEncoder())])

        preprocessor = ColumnTransformer(
            transformers=[('numric', numeric_transformer, numric_cols),
                          ('category', categorical_transformer, categorical_cols)])
        return preprocessor

    def train(self, regressor=RandomForestRegressor(n_jobs=-1, max_depth=15, n_estimators=15)):

        preprocessor = self.Preproccessor()
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('regressor', regressor)])

#         mlflow.set_experiment('Rossman-' + self.model_name)
#         mlflow.sklearn.autolog()
#         with mlflow.start_run(run_name="Baseline"):
        print("Training random forest model....")
        model = pipeline.fit(self.X_train, self.y_train)

        return pipeline, model

    def test(self, model):

        print("Testing random forest model....")

        predictions = model.predict(self.X_test)
        score_2 = r2_score(self.y_test, predictions)
        loss = loss_function(predictions, self.y_test)
        print(f" R2 score of model is: {score_2:.3f}")

        print(f"step Mean abs error of model is: {loss:.3f}")

        result_df = self.X_test.copy()
        result_df["Prediction Sales"] = predictions
        result_df["Actual Sales"] = self.y_test
        result_agg = result_df.groupby("Day").agg(
            {"Prediction Sales": "mean", "Actual Sales": "mean"})

        return score_2, loss, result_agg

    def pred_graph(self, res_dataframe):

        fig = plt.figure(figsize=(18, 5))
        sns.lineplot(x=res_dataframe.index,
                     y=res_dataframe["Actual Sales"], label='Actual')
        sns.lineplot(x=res_dataframe.index,
                     y=res_dataframe["Prediction Sales"], label='Prediction')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel(xlabel="Day", fontsize=16)
        plt.ylabel(ylabel="Sales", fontsize=16)
        plt.show()

        return fig

    def get_feature_importance(self, model):
        if (type(model.steps[1][1]) == type(LinearRegression())):
            model = model.steps[1][1]

            p_df = pd.DataFrame()
            p_df['features'] = self.X_train.columns.to_list()
            p_df['coff_importance'] = abs(model.coef_)

            return p_df

        importance = model.steps[1][1].feature_importances_
        f_df = pd.DataFrame(columns=["features", "importance"])
        f_df["features"] = self.X_train.columns.to_list()
        f_df["importance"] = importance
        return f_df
