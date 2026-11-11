import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

def noise_processing_func(df):
    df_copy = df.copy()
    df_copy["Fuel_station_manager"] = df_copy["Fuel_station_manager"].where(
    df_copy["Fuel_station_manager"].map(df_copy["Fuel_station_manager"].value_counts()) >= 20, "Other")
    df_copy["City"] = df_copy["City"].where(df_copy["City"].map(df_copy["City"].value_counts()) >= 20, "Other") 
    return df_copy

def datetime_processing_func(df):
    df_copy = df.copy()
    dt = pd.to_datetime(df_copy["Date"])
    df_copy["Date"] = dt.dt.dayofyear
    df_copy["Time"] = dt.dt.hour * 60 + dt.dt.minute
    return df_copy

def train_test_splitting(df):
    df = df.drop_duplicates().sort_values("Date").reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    df_learning, df_check = df.iloc[:split_idx], df.iloc[split_idx:]

    X_train, y_train = df_learning.drop(columns="Price"), df_learning["Price"]
    X_test,  y_test  = df_check.drop(columns="Price"),    df_check["Price"]
    return X_train, X_test, y_train, y_test

from sklearn.cluster import KMeans

def lag_feature_func(df, sort_feature):
    df_sorted = df.sort_values(sort_feature, kind="stable")
    df_sorted["price_lag"] = df_sorted.groupby(["Id", "isSelf"])["Price"].shift(1)
    return df_sorted.sort_index()


def region_generation_func(df):
    stations = df[["Id", "Latitude", "Longitudine"]].dropna().drop_duplicates("Id")
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    stations["region"] = kmeans.fit_predict(stations[["Latitude", "Longitudine"]])
    df = df.copy()
    df["region"] = df["Id"].map(stations.set_index("Id")["region"])
    return df