import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import math
from datetime import datetime


### Create strip plot of all data highlighting anomalies
def create_strip_plot(data, anomalies):

    stripdata_ii = pd.melt(data, ignore_index=True)
    stripdata = pd.melt(data, ignore_index=False)
    # stripdata_nodate = stripdata.loc[-stripdata.variable.isin(['Date'])]

    fig = px.strip(stripdata_ii, x="variable", y="value", color_discrete_sequence=['lightsteelblue'],
                   hover_name=stripdata.index.strftime("%B %d, %Y"))
    for index, row in anomalies.iterrows():
        fig.add_trace(go.Scatter(
            x=row.index,
            y=row,
            mode='markers',
            marker=dict(color="navy"),
            showlegend=False

        ))
    # add latest data point
    latest = data.iloc[-1, :18]
    fig.add_trace(go.Scatter(
        x=latest.index,
        y=latest,
        mode='markers',
        marker=dict(size=9,
                    color='magenta',
                    line=dict(width=1.2,
                              color='DarkSlateGray')),
        showlegend=False
    ))
    fig.update_layout(xaxis_title = '', yaxis_title = '')
    fig.update_layout(width=900, height=500)
    st.plotly_chart(fig)

### Create box plot of all data highlighting anomalies
def create_box_plot(data, anomalies):

    stripdata_ii = pd.melt(data, ignore_index=True)
    stripdata = pd.melt(data, ignore_index=False)
    # stripdata_nodate = stripdata.loc[-stripdata.variable.isin(['Date'])]

    fig = px.box(stripdata_ii, x="variable", y="value", color_discrete_sequence=['lightsteelblue'],
                   hover_name=stripdata.index.strftime("%B %d, %Y"))
    for index, row in anomalies.iterrows():
        fig.add_trace(go.Scatter(
            x=row.index,
            y=row,
            mode='markers',
            marker=dict(color="navy"),
            showlegend=False
        ))
    # add latest data point
    latest = data.sort_index(ascending=True).iloc[-1, :18]
    fig.add_trace(go.Scatter(
        x=latest.index,
        y=latest,
        mode='markers',
        marker=dict(size=9,
                    color='magenta',
                    line=dict(width=1.2,
                              color='DarkSlateGray')),
        showlegend=False
    ))
    fig.update_layout(xaxis_title = '', yaxis_title = '')
    fig.update_layout(width=800, height=500)
    st.plotly_chart(fig)

### Function to plot results. Shows trace of chosen maturity with latest data point and anomalies highlighted
def plot_results(df, anomalies, mat):
    df.sort_index(inplace=True, ascending=True)

    fig = px.line(df, x=df.index, y=df[mat])
    fig.update_traces(line=dict(color="dimgray", width=0.4))

    latest = df.iloc[-1:]

    fig.add_trace(go.Scatter(
        x=latest.index,
        y=latest[mat],
        mode='markers',
        marker=dict(size=9,
                    color='magenta',
                    line=dict(width=1.2,
                              color='DarkSlateGray')),
        showlegend=False
    ))

    anomalies.sort_index(ascending=True)
    fig.add_trace(go.Scatter(x=anomalies.index,
                             y=anomalies[mat],
                             mode='markers',
                             marker=dict(size=6,
                                         color='navy',
                                         line=dict(width=1,
                                                   color='DarkSlateGray'))
                             ))

    fig.update_layout(width=1300, height=500)
    fig.update_traces(line=dict(color="lightsteelblue", width=0.3), hovertemplate='Date: %{x} <br> Yield Change: %{y}')
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)

### function to plot bar charts of a selection of daily changes from anomaly list. Recent data point plotted first then biggest anomaly scores by chosen alg
def plot_anomal(data, algorithm, seperator):
    global flag
    df = data.sort_values(algorithm).head(10)

    # df.sort_index(inplace=True)
    latest = data.sort_index(ascending=True)[-1:]
    df = pd.concat([latest, df], axis=0).sort_index(ascending=False)
    df = df[seperator]

    columns = 2
    rows = math.ceil(df.shape[0] / columns)
    fr, axes = plt.subplots(rows, columns, figsize=(17, rows * 4))
    # fr.suptitle('Yield curve daily changes for selected anomalies ', fontsize=16)

    axes = axes.flatten()
    for c, i in enumerate(axes):
        if c < len(df.index):
            if c == 0:
                color_switch = 'magenta'
            else:
                color_switch = 'navy'
            axes[c].set_title(df.index[c].strftime("%B %d, %Y"), fontsize=10)
            axes[c].bar(df.columns, df.iloc[c, :].multiply(10000), color=color_switch)
            axes[c].set_ylim([-20, 20])

    st.pyplot(fr)

