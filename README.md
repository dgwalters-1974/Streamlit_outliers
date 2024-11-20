# Streamlit app for detecting outliers in interest rate swaps / fra data

Daily financial time series data investigated for outliers with a selection of unsupervised learning algorithms. The nature of the problem dictates that the number of anomalous points is governed by a contamination rate set by the user and determined by domain specific factors. Algorithms investigated include K=nearest neighbours, Isolation Forest, Mahalanobis distance and Support Vector Machines.
* Packages: Python , Matplotlib, Numpy, Pandas, Streamlit
* All models are trialled with default parameters for an overview of how they perform on the highly correlated data
* Streamlit app built to run models on new data sets / for other currencies 
![newplot](https://github.com/user-attachments/assets/8057c87a-51eb-471a-879c-146dc52fc084)
