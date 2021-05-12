This folder is for "Module 5: Regime Prediction with Machine Learning". The necessary data is under ```data``` folder, and 3 notebook files and one python script are used in the analysis.

## code

```Part1.ipynb``` Introduces and describes the problem with academic references. Provides a brief overview of the dataset.

```Part2.ipynb``` Data cleaning and feature transformation

```Part3.ipynb``` Predictive models with machine learning algortihms

```regimeplot.py``` Custom plotting function with regimes

## data

This folder includes the dataset that are used in the analysis.

1. ```macro_raw.csv``` Raw macroeconomic features
2. ```macro_processed.csv``` Cleaned and processed macroeconomic features that we get after running ```Part2.ipynb```.
2. ```sp500.csv``` SP500 total returns closing prices from Yahoo finance
3. ```recession_dates.csv``` Recession dates during the sample period from NBER. Label 1 is used for recession and 0 for expansion periods.