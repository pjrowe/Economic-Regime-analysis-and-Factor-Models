This folder is for "Module 5: Regime Prediction with Machine Learning". The necessary data is under ```data``` folder, and 3 notebook files and one python script are used in the analysis.

## code

```Part1.ipynb``` Introduces and describes the problem with academic references. Provides a brief overview of the dataset.

```Part2.ipynb``` Data cleaning and feature transformation

```Part3.ipynb``` Predictive models with machine learning algortihms

```regimeplot.py``` Custom plotting function with regimes

```2021_03_Part1.ipynb``` Updated May 13, 2021 for most recent macro data (March 2021)

```2021_03_Part2.ipynb``` ditto

```2021_03_Part3.ipynb``` ditto

## data

This folder includes the dataset that are used in the analysis.  Dated files will have the data that is used in the corresponding dated notebooks above (e.g., 2021_03_Part1.ipynb will use macro data updated up to March 2021, with SP500 price for end of March 2021, and recession status for March 2021).  The last run date was May 13, 2021, but this was based on the most up-to-date macro data from the Fed, in March 2021.  Although the Random Forest and XGBoost models were considerably better than the linear regression models across all four out-of-sample error metrics, and even though RF and XGB models called for a recession (the LR models had very low probability for recession), the closing price of the S&P rose 6.5% from 3973 on March 31 to 4233 on May 7 (another record high).  There has been roughly a 3% pullback the last few days, to 4120 intraday as of May 13.  
 
1. ```macro_raw.csv``` Raw macroeconomic features
2. ```macro_processed.csv``` Cleaned and processed macroeconomic features that we get after running ```Part2.ipynb```.
2. ```sp500.csv``` SP500 total returns closing prices from Yahoo finance for the month, even though date of each row is first day of month. 
3. ```recession_dates.csv``` Recession dates during the sample period from NBER. Label 1 is used for recession and 0 for expansion periods.
4. ```2021_03_macro_raw.csv```  Raw macro features updated as of March 2021
5. ```2021_03_macro_processed.csv```  Cleaned and processed macroeconomic features that we get after running ```Part2.ipynb```.  Note that the last period of this file is February 2021, due to the cleaning / processing of the raw data.

etc.