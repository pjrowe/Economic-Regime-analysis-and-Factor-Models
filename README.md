## Summary

This repository contains my work related to the EDHEC Business School course *Python and Machine Learning for Asset Management* on Coursera (https://www.coursera.org/account/accomplishments/verify/R5NTFNMF67P4).  I have extended and adapted the base of code provided in the course for my own education and investing and trading activities at Crivano Capital. 

There are four main parts of this project:

1. Data Loading and Cleaning
2. Regime Classification / Prediction from Economic Indicators 
3. Scenario Simulations and MultiPeriod Optimization
4. Factor Model of Market Return

### Data Loading and Cleaning

The first step in building a prediction model is collecting and cleaning relevant economic data. The data we used came from the St. Louis Fed (see below).  The database of monthly macro indicators was downloaded on February 5, 2020, and the most recent entries were for the month of December 2019. 

The relevant Jupyter notebooks and code are as follows:

- Part 1 - Problem Description and Data Analysis.ipynb
- Data Preprocessing of updated data.ipynb
- Recession_Periods.csv
- data_clean_and_preprocess.py
- features.csv

**Source Database:**

- **M. McCracken and S. Ng** "__[FRED-MD: A Monthly Database for Macroeconomic Research](https://research.stlouisfed.org/econ/mccracken/fred-databases/)__", _Working Paper,_ 2015.  <a class="anchor" id="i"></a>
- https://s3.amazonaws.com/files.fred.stlouisfed.org/fred-md/monthly/current.csv


### Regime Classification / Prediction from Economic Indicators 

Based on the source database listed above we tested various binary economic regime predictors (normal regime or recession). 

Relevant files
- Forecasting_Regimes_with_updated_data.ipynb
- Random_Forest_Model_Tuning.ipynb

### Scenario Simulations and MultiPeriod Optimization

Relevant files
- Scenario_Simulations.ipynb
- Assets_7.csv

### Factor Model of Market Returns
