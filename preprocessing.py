import pandas as pd
import numpy as np
import pickle

import statsmodels.api as sm

from sklearn.preprocessing import OneHotEncoder
from project_functions.normalization import MaxAbsLogScaler

np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})
pd.options.display.float_format = '{:.2f}'.format

def make_page_name_features(df: pd.DataFrame) -> np.ndarray:

    """
    Extract features from page index and apply One Hot Encoding.
    Return matrix of One Hot Encoded features.
    """

    extract_groups = [
        r'(?P<page_name>^.*(?=_))'
        r'(?P<junk_1>_)',
        r'(?P<project>(?<=_)[^_]*(?=_))', 
        r'(?P<junk_2>_)',
        r'(?P<access>(?<=_)[^_]*(?=_))', 
        r'(?P<junk_3>_)', 
        r'(?P<agent>(?<=_).*$)'
        ]

    page_index = df.axes[0]

    page_features = page_index. \
        str.extract("".join(extract_groups)). \
        drop(columns=['page_name', 'junk_1', 'junk_2', 'junk_3']). \
        to_numpy()

    encoder = OneHotEncoder()
    
    arr_page_features = [encoder.fit_transform(feature.reshape(-1,1)).toarray() for feature in page_features.T] 

    return np.concatenate(arr_page_features, axis=1)

def make_page_stats_features(df: pd.DataFrame) -> np.ndarray:

    """
    Returns traffic statistics for each page:
    - visit median scaled with MaxAbsLogScaler (log(x+1) and min-max normalization)
    - autocorrelation lagged 7 days 
    """

    median_scaler = MaxAbsLogScaler()

    visits = df.to_numpy()
    
    page_visits_median = np.nanmedian(visits, axis=1).reshape(1,-1)
    page_visits_median_scaled = median_scaler.fit_transform(page_visits_median).reshape(-1,1)

    page_visits_autocorr_7 = np. \
        apply_along_axis(
            func1d=lambda arr, lag: sm.tsa.acf(x=arr, nlags=lag, missing='conservative')[-1], 
            axis=1,    
            arr=visits, 
            lag=7
        ). \
        reshape(-1,1)
    
    return np.concatenate([page_visits_median_scaled, page_visits_autocorr_7], axis=1)

def make_page_features(df: pd.DataFrame) -> np.ndarray:
    
    """
    Combines feature extraction from page index and calculation of visits median.
    """

    page_name_features = make_page_name_features(df)
    page_stats_features = make_page_stats_features(df)

    return np.concatenate([page_name_features, page_stats_features], axis=1)

def make_time_features(df: pd.DataFrame) -> np.ndarray:
    """
    Add circular time features - weekly and yearly cycles represented as sin and cos signals, to enable seasonality modeling.
    """

    visit_dates = df.axes[1]
    visit_dates = pd.DatetimeIndex(visit_dates)

    visit_dow = visit_dates.day_of_week.to_numpy()
    visit_doy = visit_dates.day_of_year.to_numpy()
    
    def sin_wave(s: pd.Series, n_steps: np.int64):
        return np.sin(s / n_steps * (2 * np.pi))

    def cos_wave(s: pd.Series, n_steps: np.int64):
        return np.cos(s / n_steps * (2 * np.pi))

    weekly_sin = sin_wave(visit_dow, 7).reshape(-1,1)
    weekly_cos = cos_wave(visit_dow, 7).reshape(-1,1)

    yearly_sin = sin_wave(visit_doy, 365).reshape(-1,1)
    yearly_cos = cos_wave(visit_doy, 365).reshape(-1,1)

    return np.concatenate([weekly_sin, weekly_cos, yearly_sin, yearly_cos], axis=1)

def run():
    """
    Run all preprocesing and persist results.
    """

    # Get data

    df_wiki = pd. \
        read_csv("input/train_2.csv"). \
        set_index("Page"). \
        astype('float')

    df_wiki.columns = pd.to_datetime(df_wiki.columns)
    
    # Exclude bad data (based on exploration)

    max_per_page = df_wiki.loc[:, "2016-07-01":"2016-09-01"].max(axis=1)
    outlier_pages = max_per_page[max_per_page >= 4e7].index

    df_wiki.loc[outlier_pages, "2016-07-18":"2016-08-18"] = pd.NA
    df_wiki = df_wiki.loc[:,:"2017-06-30"]

    # Include only pages that meet minimum days condition in both training and testing dataset

    TRAIN_TEST_RATIO = 0.8
    MIN_DAYS_VALID = 120

    train_test_split = round(TRAIN_TEST_RATIO * df_wiki.shape[1])

    print(f"Training days: {train_test_split}\nTesting days: {df_wiki.shape[1] - train_test_split}")

    has_min_days_train = df_wiki.iloc[:, :train_test_split].replace(0.0, pd.NA).notna().sum(axis=1) >= MIN_DAYS_VALID
    has_min_days_test = df_wiki.iloc[:, train_test_split:].replace(0.0, pd.NA).notna().sum(axis=1) >= MIN_DAYS_VALID

    df_wiki = df_wiki[has_min_days_train & has_min_days_test]

    print(f"\nPages used: {df_wiki.shape[0]}\nDays used: {df_wiki.shape[1]}")

    # Interpolate missing data (only at dates between existing data points)

    df_wiki = df_wiki.T. \
        asfreq(freq="D"). \
        interpolate(method='time', axis=0, limit_area="inside").T

    # Split datataset to train vs test+valid based on time (training based on history and predicting future)

    df_wiki_train = df_wiki.iloc[:, :train_test_split]
    df_wiki_test_valid = df_wiki.iloc[:, train_test_split:]

    # Split test and valid datasets based on pages

    df_wiki_test = df_wiki_test_valid.iloc[0::2, :]
    df_wiki_valid = df_wiki_test_valid.iloc[1::2, :]

    # Extract page features (page, site, access, agent) and visits stats

    page_features_train = make_page_features(df_wiki_train)
    page_features_test = make_page_features(df_wiki_test)
    page_features_valid = make_page_features(df_wiki_valid)
    
    # Make circular time features - weekly and yearly cycles represented by sin and cos signals

    time_features_train = make_time_features(df_wiki_train)
    time_features_test = make_time_features(df_wiki_test)
    time_features_valid = make_time_features(df_wiki_valid)

    # Data normalization
    # Each page normalized independently due to relevant differences in average visits

    visits_scaler_train = MaxAbsLogScaler()
    visits_scaler_test = MaxAbsLogScaler()
    visits_scaler_valid = MaxAbsLogScaler()

    visits_train = df_wiki_train.to_numpy()
    visits_test = df_wiki_test.to_numpy()
    visits_valid = df_wiki_valid.to_numpy()

    visits_scaled_train = visits_scaler_train.fit_transform(visits_train)
    visits_scaled_test = visits_scaler_test.fit_transform(visits_test)
    visits_scaled_valid = visits_scaler_valid.fit_transform(visits_valid)

    # Write prepared data

    np.savez(
        file="data/features_train.npz",
        visits = visits_scaled_train, 
        time = time_features_train, 
        page = page_features_train
        )

    np.savez(
        file="data/features_test.npz",
        visits = visits_scaled_test,
        time = time_features_test, 
        page = page_features_test
        )

    np.savez(
        file="data/features_valid.npz",
        visits = visits_scaled_valid,
        time = time_features_valid, 
        page = page_features_valid
        )

    # Write scalers for later (reverting scaling to obtain nominal values of predictions)

    with open("data/visits_scaler_train.pk", "wb") as file:
        pickle.dump(visits_scaler_train, file=file)

    with open("data/visits_scaler_test.pk", "wb") as file:
        pickle.dump(visits_scaler_test, file=file)

    with open("data/visits_scaler_valid.pk", "wb") as file:
        pickle.dump(visits_scaler_valid, file=file)
    
if __name__ == '__main__':
    run()