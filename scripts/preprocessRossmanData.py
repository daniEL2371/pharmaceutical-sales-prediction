import numpy as np
import pandas as pd


class PreprocessRossmanData:

    def __init__(self):
        pass

    def handle_outliers(self, df, col, method="lower_upper"):

        df = df.copy()
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)

        lower_bound = q1 - ((1.5) * (q3 - q1))
        upper_bound = q3 + ((1.5) * (q3 - q1))

        if method == "mean":
            df[col] = np.where(df[col] < lower_bound,
                               df[col].mean(), df[col])
            df[col] = np.where(df[col] > upper_bound, df[col].mean(), df[col])

        elif method == "mode":
            df[col] = np.where(df[col] < lower_bound,
                               df[col].mode()[0], df[col])
            df[col] = np.where(df[col] > upper_bound,
                               df[col].mode()[0], df[col])
        else:
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

        return df

    def transform_date(self, df):

        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = pd.DatetimeIndex(df['Date']).year
        df['Month'] = pd.DatetimeIndex(df['Date']).month
        df['Day'] = pd.DatetimeIndex(df['Date']).day

        df['Year'] = df['Year'].astype("int")
        df['Month'] = df['Month'].astype("category")
        df['Day'] = df['Day'].astype("category")
        df['DayInMonth'] = df['Day'].apply(lambda x: self.to_month_category(x))
        df['DayInMonth'] = df['DayInMonth'].astype("category")
        return df

    def to_month_category(self, value):
        try:
            if (value >= 1 and int(value) < 10):
                return "BegMonth"

            elif (value >= 10 and value < 20):
                return "MidMonth"
            else:
                return "EndMonth"
        except:
            pass

    def add_weekday_col(self, df):

        df["Weekends"] = df["DayOfWeek"].apply(lambda x: 1 if x > 5 else 0)
        df["Weekends"] = df["Weekends"].astype("category")
        return df

    def encode_train_data(self, df):

        StateHolidayEncoder = preprocessing.LabelEncoder()
        DayInMonthEncoder = preprocessing.LabelEncoder()

        df['StateHoliday'] = StateHolidayEncoder.fit_transform(
            df['StateHoliday'])
        df['DayInMonth'] = DayInMonthEncoder.fit_transform(df['DayInMonth'])
        return df

    def encode_store_data(self, df):
        StoreTypeEncoder = preprocessing.LabelEncoder()
        AssortmentEncoder = preprocessing.LabelEncoder()
        PromoIntervalEncoder = preprocessing.LabelEncoder()


#         PromoInterval
        df['StoreType'] = StoreTypeEncoder.fit_transform(df['StoreType'])
        df['Assortment'] = AssortmentEncoder.fit_transform(df['Assortment'])
        df['PromoInterval'] = PromoIntervalEncoder.fit_transform(
            df['PromoInterval'])

        return df

    def merge_encoded(self, train_enc, store_enc):
        return pd.merge(train_enc, store_enc, on="Store")

    def process(self, train_df, store_df, is_test_data=False):

        #         enc_train = self.encode_train_data(train_df)
        #         enc_store = self.encode_store_data(store_df)

        #         enc_train = enc_train.drop(columns=["Date"], axis=1)

        train_df = self.transform_date(train_df)
        train_df = self.add_weekday_col(train_df)

        if (not is_test_data):
            train_df = self.handle_outliers(train_df, "Sales")
            train_df = self.handle_outliers(train_df, "Customers")

        store_df = self.handle_outliers(store_df, "CompetitionDistance")

        merged = self.merge_encoded(train_df, store_df)
#         sclaed = helper.scaler(merged, merged.columns.to_list(), mode="standard")

        return merged
