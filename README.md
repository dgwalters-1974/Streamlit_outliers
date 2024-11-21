# Streamlit app for detecting outliers in interest rate swaps / fra data

Daily financial time series data investigated for outliers with a selection of unsupervised learning algorithms. The nature of the problem dictates that the number of anomalous points is governed by a contamination rate set by the user and determined by domain specific factors. Algorithms such as include K=nearest neighbours, Isolation Forest, Mahalanobis distance and Support Vector Machines were applied to the differenced data in order to categorize datapoints as normal / anomalous and the results were summarised in a Streamlit app.

<p align="center">
  <img src="/images//newplot_.png" width="700" title="hover text">
</p>

* Packages: Python , Matplotlib, Numpy, Pandas, Streamlit
* All models are trialled with default parameters for an overview of how they perform on the highly correlated data
* Streamlit app built to run models on new data sets / for other currencies 

<p align="center">
  <img src="/images//newplot.png" width="350" title="hover text">
</p>

<p float="left">
  <img src="/images/newplot (1).png" width="400" />
  <img src="/images//newplot.png" width="400"/> 
  
</p>

