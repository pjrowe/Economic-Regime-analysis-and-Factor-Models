## Scenario_Simulations_and_MultiPeriod_Optimization.ipynb

This repository contains my work related to the EDHEC Business School course *Python and Machine Learning for Asset Management* on Coursera (https://www.coursera.org/account/accomplishments/verify/R5NTFNMF67P4).  I have extended the base of code provided in the course for my own personal use and edication.  There are four main parts of this particular project:

1. Data loading and cleaning
2. Regime Classification / Prediction from Economic Indicators 
3. Scenario Simulations and MultiPeriod Optimization
4. Factor Model of Market Returns

### Regime Classification / Prediction from Economic Indicators 

Based on the database below, which is updated regularly, we are going to build a recession predictor, testing various machine learning models in the process.  The database of monthly macro indicators was downloaded on February 5, 2020, and the most recent entries were for the month of December 2019.

Source Database: 

- **M. McCracken and S. Ng** "__[FRED-MD: A Monthly Database for Macroeconomic Research](https://research.stlouisfed.org/econ/mccracken/fred-databases/)__", _Working Paper,_ 2015.  <a class="anchor" id="i"></a>
- https://s3.amazonaws.com/files.fred.stlouisfed.org/fred-md/monthly/current.csv
 