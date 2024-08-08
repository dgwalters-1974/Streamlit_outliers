from __future__ import division, print_function

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
import functions
import matplotlib.pyplot as plt

import pyod
import suod

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn import model_selection

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from numpy import percentile
import matplotlib.pyplot as plt
import matplotlib.font_manager

# Import all models
from pyod.models.abod import ABOD
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.iforest import IForest
# from pyod.models.dif import DIF
from pyod.models.pca import PCA
from pyod.models.ocsvm import OCSVM

st.set_page_config(layout='wide')

with st.sidebar:
    uploaded_file = st.file_uploader(label='.xlsx or .csv file containing swaps and fras only...',
                                     type=["xlsx", "csv"])


@st.cache_data
def import_and_run():
    # path = r'rates3.csv'
    df = pd.read_csv(uploaded_file)
    df.dropna(inplace=True)
    df.drop_duplicates('Date', inplace=True)

    df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')
    df = df.set_index('Date')
    df = df.diff().dropna().sort_index(ascending=False)

    # Save a copy of complete unscaled df in variable df_original (fras + swaps)
    df_original = df.copy()

    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(data_scaled, columns=df.columns, index=df.index)

    swaps_filter = [x for x in df.columns if 'Y' in x]
    fras_filter = [x for x in df.columns if 'Y' not in x]

    # Save swaps and fras to seperate scaled df
    swaps = df_scaled[swaps_filter]
    fras = df_scaled[fras_filter]

    #     Run models & write to dict
    outliers_fraction = 0.01
    random_state = 74

    # Define 7 outlier detection tools to be compared
    classifiers = {
        # 'Angle-based Outlier Detector (ABOD)': ABOD(contamination=outliers_fraction, n_neighbors=10), # Defaults to 'fast', n_neighbours=10
        'K Nearest Neighbors (KNN)': KNN(contamination=outliers_fraction),  # Defaults to 'largest', n_neighbours=5
        'Average KNN': KNN(method='mean', contamination=outliers_fraction),  # mean distance, n_neighbours=5
        'Median KNN': KNN(method='median', contamination=outliers_fraction),  # median distance, n_neighbours=5
        'Local Outlier Factor (LOF)': LOF(n_neighbors=35, contamination=outliers_fraction),
        # Defaults to n_neighbours=20, leaf_size=30
        'Isolation Forest': IForest(contamination=outliers_fraction, random_state=random_state),
        # 'Deep Isolation Forest (DIF)': DIF(contamination=outliers_fraction, random_state=random_state),
        'Principal Component Analysis (PCA)': PCA(contamination=outliers_fraction, random_state=random_state,
                                                  n_components=5),  # n_components default = all
        'One-class SVM (OCSVM)': OCSVM(contamination=outliers_fraction),
    }

    portfolio = {
        'swaps': swaps,
        'fras': fras
    }

    # https://pyod.readthedocs.io/en/latest/pyod.models.html
    #  https://www.dbs.ifi.lmu.de/~zimek/publications/KDD2008/KDD08-ABOD.pdf

    results = {}

    for name, dfs in portfolio.items():
        X = dfs.values
        results_df = pd.DataFrame()
        print()
        print('fitting', name, '......')

        for i, (clf_name, clf) in enumerate(classifiers.items()):
            print()
            print(i + 1, 'fitting', clf_name, 'for portfolio:', name)

            clf.fit(X)
            scores_pred = clf.decision_function(X) * (
                int(-1) if clf_name == 'Angle-based Outlier Detector (ABOD)' else int(-1))
            results_df[clf_name] = pd.Series(scores_pred)

        results_scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(results_df)
        results_scaled = pd.DataFrame(results_scaled, columns=results_df.columns, index=dfs.index)
        results[name] = results_scaled
        print()

    # Add Mahalanobis distance to outlier metrics
    def calculateMahalanobis(y=None, data=None, cov=None):
        data.sort_index(ascending=True)
        y_mu = y - np.mean(data)
        if not cov:
            cov = np.cov(data.values.T)
        inv_covmat = np.linalg.inv(cov)
        left = np.dot(y_mu, inv_covmat)
        mahal = np.dot(left, y_mu.T)
        return mahal.diagonal()

    results['swaps'].sort_index(ascending=True)
    results_mahala_s = calculateMahalanobis(y=swaps, data=swaps[list(swaps.columns)])
    results['swaps']['Mahalanobis'] = 1 - MinMaxScaler(feature_range=(0, 1)).fit_transform(
        results_mahala_s.reshape(-1, 1))
    results['fras'].sort_index(ascending=True)
    results_mahala_f = calculateMahalanobis(y=fras, data=fras[list(fras.columns)])
    results['fras']['Mahalanobis'] = 1 - MinMaxScaler(feature_range=(0, 1)).fit_transform(
        results_mahala_f.reshape(-1, 1))

    df_original.sort_index(ascending=True)
    results_swaps = pd.merge(df_original, results['swaps'], left_index=True, right_index=True)
    results_fras = pd.merge(df_original, results['fras'], left_index=True, right_index=True)
    return results_fras, results_swaps


# data_file = 'results_complete.csv'
# data = pd.read_csv(data_file)
# data = data.set_index(pd.DatetimeIndex(data['Date']))
# del data['Date']
# # data['Date'] = pd.to_datetime(data['Date'])
# data = data.dropna(axis=0).sort_index(ascending=True)

if uploaded_file is not None:
    results_fras, results_swaps = import_and_run()

    # if uploaded_file is not None:
    #     results_fras, results_swaps = import_and_run()


    with st.sidebar:
        data_picker = st.selectbox('SWAPS or FRAS', ['swaps', 'fras'])
        if data_picker == 'swaps':
            data = results_swaps
        elif data_picker == 'fras':
            data = results_fras

        cutoff = st.number_input('Quantile Cutoff', step=0.01)
        algo = st.selectbox('Choose algorithm', data.iloc[:, 26:].columns)

        chart_pick = st.selectbox('Choose maturity...', data.iloc[:, :26].columns)

    summary = data[data[algo] < np.quantile(data[algo], cutoff)]

    rates = data.iloc[:, :26]
    results = data.iloc[:, 26:]

    summary_rates = summary.iloc[:, :26]

    # rates[[x for x in rates.columns if 'Y' in x]]

    s = [x for x in rates.columns if 'Y' in x]
    f = [x for x in rates.columns if 'Y' not in x]

    if data_picker == 'swaps':
        flag = s
    elif data_picker == 'fras':
        flag = f

    st.title('Outlier detection for swaps / fras')
    st.subheader('An informal comparison of some traditional machine learning methods')

    try:
        is_outlier = data.index.sort_values()[-1] == summary.index.sort_values()[-1]
        st.write(f"Current value {'*IS*' if is_outlier == True else '*IS NOT*'} an outlier :grin:")
    except:
        pass

    functions.plot_results(rates.multiply(10000), summary_rates.multiply(10000), chart_pick)

    col1, col2 = st.columns(2, gap="small")
    with col1:
        functions.create_strip_plot(rates[flag].multiply(10000), summary_rates[flag].multiply(10000))
    with col2:
        functions.create_strip_plot(results, results.iloc[results.index.isin(summary.index)])


    # x=data.sort_values(algo).head(10)
    # y=data.sort_index(ascending=False).iloc[0]
    functions.plot_anomal(data, algo, flag)





