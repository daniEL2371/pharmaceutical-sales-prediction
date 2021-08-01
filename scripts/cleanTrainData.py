import numpy as np
import pandas as pd


class CleanTrainData:

    def __init__(self):
        pass

    def to_numeric(self, df):
        df["Customers"] = df["Customers"].astype("int")
        df["Sales"] = df["Sales"].astype("int")
        return df

    def to_category(self, df):

        df["Open"] = df["Open"].astype("category")
        df["DayOfWeek"] = df["DayOfWeek"].astype("category")
        df["Promo"] = df["Promo"].astype("category")
        df["StateHoliday"] = df["StateHoliday"].astype("category")
        df["SchoolHoliday"] = df["SchoolHoliday"].astype("category")
        df['StateHoliday'] = df['StateHoliday'].astype(
            "str").astype("category")
        return df

    def drop_closed_stores(self, df):

        try:
            cleaned = df.query("Open == 1")
            return cleaned
        except:
            pass

    def convert_to_datatime(self, df):
        try:
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        except:
            pass

    def sort_by_date(self, df):
        return df.sort_values(by=["Date"], ascending=False)

    def get_cleaned(self, df):
        df = self.to_category(df)
        df = self.to_numeric(df)
#         df = self.drop_closed_stores(df)
        df = self.convert_to_datatime(df)

        return df
