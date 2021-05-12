## Summary

This repository contains my work related to the EDHEC Business School specialization *Python and Machine Learning for Asset Management* on Coursera (https://www.coursera.org/account/accomplishments/verify/R5NTFNMF67P4).  I have extended and adapted the base of code provided in the course for my own education and investing and trading activities at Crivano Capital. 

The Specialization is comprised of four courses. https://www.coursera.org/specializations/investment-management-python-machine-learning#courses

1. Introduction to Portfolio Construction and Analysis with Python
2. Advanced Portfolio Construction and Analysis with Python
3. Python and Machine Learning for Asset Management
4. Python and Machine-Learning for Asset Management with Alternative Data Sets

Course 2 was too high level and general for a one person investor/trader.  The theory is quite interesting, though, and I realized after this course that the practice of portfolio management has been largely automated (think ETFs) and things like portfolio rebalancing could be accomplished with the click of a button, after all the theory has been implemented.  Thus, Course 2 did not have much for me to try to use, unless I intended to work for a large asset management firm as part of a massive team.  That was my impression, at least.

Course 4 had some interesting applications of machine learning to investment process, but there were some applications that would be best described as 'smoke and mirrors', according to one headhunter.  In the former category, analyzing newly available sources of data (cell phone geolocation data, transactional data from web sites, etc.) could yield useful insights of retail activity during a Christmas shopping season.  In the less interesting category would be analyzing twitter feeds or 10-K for mentions of a certain company / competitor to analyze sentiment about a stock or company.  This application seems very susceptible to changing or gaming.  I can't imagine how a portfolio manager could have much conviction repeatably over such a soft analysis.  

That leaves Course 3, which was split into 5 modules / weeks.  This Course did have some promise for application of a small trader / investor.

1. Introducing the fundamentals of machine learning (background, nothing included in repo)
2. Machine learning techniques for robust estimation of factor models (Module 2 folder in repo)
3. Machine learning techniques for efficient portfolio diversification
4. Machine learning techniques for regime analysis (Module 4 folder in repo)
5. Identifying recessions, crash regimes and feature selection (Module 5 folder)

Week 2 and Week 3 were again, in my estimation, a bit too high level for me.  Rebalancing a large amount of money across asset classes or invidual stocks could be guided by such techniques, but this occurs at large funds / companies.  

The topics of Week 4 were as follows: Portfolio Decisions with Time-Varying Market Conditions, Trend filtering, A scenario-based portfolio model, A two regime portfolio example, A multi regime model for a University Endowment.  The two-regime portfolio was more my speed than a multi-regime model for an endowment.  Thus, I focused on Week 5, trying to use ML models to analyze and identify / predict a recession using publicly available economic data.  It would be easy to take a position in the SP500 emini futures (very liquid, tax advantageous, low commissions) as my 'portfolio' in anticipation of a major stock correction / recession.  

Then Covid hit, and a sudden recession without precedent.  So much for model building.  And then an unprecedented coordinated stimulus by Central Banks worldwide, leading to an all-time high in market multiples.  Again, so much for building models on the past data when the Fed was pumping $100 billion daily into the markets.  In addition, this all happened within 4-6 weeks top to bottom and back.  So, how useful could monthly economic indicators be?   

Anyway, I've updated this as a refresher, but as a practical method for a small fish to position in the markets, no, I don't think it is a useful exercise.  Is it any wonder so many short seller funds and value portfolio managers have closed or retired?  The market is a mania and has been one for a long time, with no end in sight.  Druckenmiller, Michael Burry, Chanos, Einhorn, Fleckenstein, and the list goes on and on.  

-------

### 1. Data Loading and Cleaning

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


### 2. Regime Classification / Prediction from Economic Indicators 

Based on the source database listed above we tested various binary economic regime predictors (normal regime or recession). 

Relevant files

- Forecasting_Regimes_with_updated_data.ipynb
- Random_Forest_Model_Tuning.ipynb

### 3. Scenario Simulations and MultiPeriod Optimization

Relevant files

- Scenario_Simulations.ipynb
- Assets_7.csv

### 4. Factor Model of Market Returns
 