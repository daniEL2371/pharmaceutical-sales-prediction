import numpy as np
import pandas as pd


class CleanStoreDf:
    """ This is a class to clean store df"""

    def __init__(self):
        pass

    def handle_missing_value(self, df):
        """We handled CompetitionDistance by replacing it with median"""

        df['CompetitionDistance'] = df['CompetitionDistance'].fillna(
            df['CompetitionDistance'].max())
        df['Promo2SinceWeek'] = df['Promo2SinceWeek'].fillna(
            df['Promo2SinceWeek'].max())
        df['Promo2SinceYear'] = df['Promo2SinceYear'].fillna(
            df['Promo2SinceWeek'].max())
        df['PromoInterval'] = df['PromoInterval'].fillna(
            df['PromoInterval'].mode()[0])
        df['CompetitionOpenSinceYear'] = df['CompetitionOpenSinceYear'].fillna(
            df['CompetitionOpenSinceYear'].mode()[0])
        df['CompetitionOpenSinceMonth'] = df['CompetitionOpenSinceMonth'].fillna(
            df['CompetitionOpenSinceMonth'].mode()[0])

        return df

    def to_numeric(self, df):

        df["CompetitionDistance"] = df["CompetitionDistance"].astype("float")
        df["Promo2SinceWeek"] = df["Promo2SinceWeek"].astype("int")
        return df

    def to_category(self, df):

        df["StoreType"] = df["StoreType"].astype("category")
        df["Assortment"] = df["Assortment"].astype("category")
        df["CompetitionOpenSinceMonth"] = df["CompetitionOpenSinceMonth"].astype(
            "category")
        df["CompetitionOpenSinceYear"] = df["CompetitionOpenSinceYear"].astype(
            "category")

        df["Promo2"] = df["Promo2"].astype("category")

        df["Promo2SinceYear"] = df["Promo2SinceYear"].astype("category")
        df["PromoInterval"] = df["PromoInterval"].astype("category")

        return df

    def get_cleaned(self, df):
        df = self.handle_missing_value(df)
        df = self.to_category(df)
        df = self.to_numeric(df)
        return df
