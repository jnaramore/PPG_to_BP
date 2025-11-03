## PPG to Blood Pressure

The purpose of this code is to show the signal processing pipeline of predicting blood pressure from PPG signals.

The code should be run in the following order. NOTE: PPG_to_BP_prediction.ipynb requires a python version 3.10 environment for the pyPPG library, details on setup and included at the top of the notebook.

1. PPG_to_BP_preprocessing.ipynb - This script runs the necessary preprocessing on raw PPG signals from the PPGBP dataset. It saves preprocessed data into ppg_signals.mat and biomarkers_PPGBP.csv.

2. PPG_to_BP_prediction.ipynb - This scripts fits XGBoost models to cross-validation schemes with and without "leakage", or pulses from the same subject used both training and test sets.
