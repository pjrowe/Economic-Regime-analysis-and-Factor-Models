"""Module 5 Jupyter Notebooks Part 1-3 and regimeplot.py.

 all rolled into one file."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import seaborn as sns

# Classification Functions and metrics from scikit-learn library
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')
import time

# to do :
# review transformations from appendix
# plot each column to understand xform


class RegimePlot:
    def __init__(self, df, regime_col):
        self.df = df
        self.regime = regime_col
        self.regime_dates = self.get_regime_dates()

    def get_regime_dates(self):
        regime_dates = []
        crash_regime = 1
        normal_regime = 0

        regime = normal_regime  # initial regime
        for i, j, k in zip(self.df[self.regime], self.df['Date'],
                           range(len(self.df))):
            if i == crash_regime and regime == normal_regime:
                # regime switch from normal to crash
                regime_span = []
                regime = crash_regime

                # start of the crash regime date
                regime_span.append(j)

            if i == normal_regime and regime == crash_regime:
                # end of crash regime
                regime = normal_regime
                # take end date from previous iteration
                regime_span.append(self.df['Date'].iloc[k - 1])

                # crash regime span [start_date,end_date]
                regime_dates.append(regime_span)
            if i == crash_regime and j == self.df['Date'].iloc[-1]:
                # if we are in crash regime at the end of dataset append
                # the last date
                regime_span.append(j)
                regime_dates.append(regime_span)
        return regime_dates

    def plt_regime(self, plt_series: list, series_label: list,
                   regime_label: str, log_scale=True, title=None,
                   orj_series=False):
        # Plot return or cumulative returns (of multiple series) over
        # single regime label

        plt.figure(figsize=(18, 6))
        plt.xlabel(' ')
        plt.ylabel(' ')
        # plt.ylim([1])
        if orj_series:
            # if True plt original series
            for i in range(len(plt_series)):
                plt.plot(self.df['Date'], self.df[plt_series[i]],
                         label=series_label[i])
        else:
            for i in range(len(plt_series)):
                plt.plot(self.df['Date'],
                         (1 + self.df[plt_series[i]]).cumprod(),
                         label=series_label[i])
        if log_scale:
            plt.yscale('log')

        for i in range(len(self.regime_dates)):
            if i != len(self.regime_dates) - 1:
                plt.axvspan(self.regime_dates[i][0],
                            self.regime_dates[i][1], alpha=0.30,
                            color='grey')
            else:
                plt.axvspan(self.regime_dates[i][0],
                            self.regime_dates[i][1],
                            alpha=0.30, color='grey',
                            label=regime_label)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16),
                   fontsize=12, frameon=False,
                   ncol=len(plt_series) + 1)
        if title:
            plt.title(title, fontsize=18)
        else:
            plt.title('Cumulative Performance Over time', fontsize=18)

        plt.show()


class MacroDataProcess:
    # Stationarity transofrmation
    # Add lag of the features

    def __init__(self, macro_data):
        self.data = macro_data
        self.transformation_codes = None

    def transform(self, df_col, code):
        """Transform each column of dataframe (df_col).

        Transformations for each code are shown in appendix

        Parameters
        ----------
        df_col: pandas dataframe column

        code: int or float
        """
        if code == 1:
            df_col.apply(lambda x: x)
            return df_col
        elif code == 2:
            df_col = df_col.diff()
            return df_col
        elif code == 3:
            df_col = df_col.diff(periods=2)
            return df_col
        elif code == 4:
            df_col = df_col.apply(np.log)
            return df_col
        elif code == 5:
            df_col = df_col.apply(np.log)
            df_col = df_col.diff(periods=2)
            return df_col
        elif code == 6:
            df_col = df_col.apply(np.log)
            df_col = df_col.diff(periods=2)
            return df_col
        elif code == 7:
            df_col = df_col.pct_change()
            df_col = df_col.diff()
            return df_col

    def stationarity(self):
        """Clean macro dataset and perform necessary changes."""
        # Keep transformation codes for each variable in a dictionary
        transformation_codes = {}
        df_tmp = pd.DataFrame(columns=self.data.columns)
        for col in self.data.columns:
            df_tmp[col] = self.data[col].iloc[1:]
            transformation_codes[col] = self.data[col].iloc[0]
        df_tmp['Date'] = pd.to_datetime(df_tmp['Date'])

        self.data = df_tmp
        self.tansformation_codes = transformation_codes
        # Make each feature stationary
        data_transformed:\
            pd.DataFrame = pd.DataFrame(columns=self.data.columns)
        for col in self.data.columns:
            if col == 'Date':
                data_transformed[col] = self.data[col]
            else:
                data_transformed[col] =\
                    self.transform(self.data[col],
                                   transformation_codes[col])
        self.data = data_transformed

    def add_lag(self, lag_values):
        for col in self.data.drop(['Date'], axis=1):
            for n in lag_values:
                self.data['{} {}M lag'.format(col, n)] =\
                    self.data[col].shift(n).ffill().values
        self.data.dropna(axis=0, inplace=True)
        return self.data


def remove_variables(df, n):
    # if a variable has more than 'n' NaN values remove it.
    dropped_cols = []
    for col in df.columns:
        if df[col].isna().sum() > n:
            dropped_cols.append(col)
            df.drop(col, axis=1, inplace=True)
    return df, dropped_cols


def dict_product(dicts):
    """Make all combinations of hyperparameters.

    >>> list(dict_product(dict(number=[1, 2], character='ab')))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def error_metrics(df, roc=False):
    """Calculate four classification error metrics.

    ACC: accuracy (0-1) loss
    QPS: quadratic probability score
    MCC: Matthew's Correlation Coefficient
    AUC: Area under the ROC curve
    """
    mcc = matthews_corrcoef(y_true=df.Regime, y_pred=df.crash_binary)
    mis_rate = df[df.Regime != df.crash_binary].shape[0] / df.shape[0]
    qps = sum((df.crash_prob - df.Regime) ** 2) / len(df)
    err_dict = {'ACC': 1 - mis_rate,
                'MCC': mcc,
                'QPS': qps}
    if roc is True:
        roc = roc_auc_score(y_true=df.Regime, y_score=df.crash_prob)
        err_dict['AUC'] = roc

    return err_dict


def clean_data(df_macro, missing_num=10):
    """Clean data.

    1. Remove last row
    2. Remove features that have more than 10 missing values (default).
    3. Forward fill missing values
    """
    print('Cleaning data...')
    print(f'Latest date of data available {df_macro.Date.iloc[-1]}')
    df_macro = df_macro[:-1]
    print(f'Latest data used for building model {df_macro.Date.iloc[-1]}')

    df_clean, dropped_cols = remove_variables(df_macro, missing_num)
    # forward fill last month and missing values in between
    df_clean.fillna(method='ffill', inplace=True)
    return df_clean


def process_data(df_clean):
    """Process data.

    1 Convert the features into stationary form by applying the necessary
    2 Transformations as stated in the appendix by authors (see notebook).
    3 Add 1, 3, 6, 9, 12 months lags of the features
    """
    print('Processing data...')
    df = MacroDataProcess(macro_data=df_clean)
    df.stationarity()
    lag_values = [1, 3, 6, 9, 12]  # each represents # months or rows
    df_process = df.add_lag(lag_values)
    df_process.to_csv('./data/macro_processed_dummy.csv', index=False)
    print('MacroFeatures shape:', df_process.shape)
    print('Start date: ' + str(df_process.Date.iloc[0]) + ' End date: '
          + str(df_process.Date.iloc[-1]))
    return df_process


def cross_valid(model_dict_cv, df_train, target_col, feature_col):
    scoring = 'ACC'
    model_dict = {}  # keep selected parameters for each model

    for model_tuple, param_grid in model_dict_cv.items():
        # produces all combinations of parameters for given model
        all_grid = list(dict_product(param_grid))
        cv_res = []
        for param in all_grid:
            # timeseries split for cross validation
            tscv = TimeSeriesSplit(n_splits=3)
            model = model_tuple[1](**param)
            score = []
            for train_index, test_index in tscv.split(df_train):
                X_train, X_test = df_train[feature_col].iloc[train_index],\
                    df_train[feature_col].iloc[test_index]
                y_train, y_test = df_train[target_col].iloc[train_index],\
                    df_train[target_col].iloc[test_index]
                date_train, date_test = df_train['Date'].iloc[train_index],\
                    df_train['Date'].iloc[test_index]
                model.fit(X_train, y_train)
                y_binary = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
                res_dict = {'Date': date_test,
                            'Regime': y_test,
                            'crash_prob': y_prob,
                            'crash_binary': y_binary}
                res_df = pd.DataFrame.from_dict(res_dict)
                score.append(error_metrics(res_df)[scoring])
            cv_res.append(np.mean(score))
        best_param = all_grid[np.argmax(cv_res)]
        model_dict[model_tuple] = best_param
    return model_dict


def test_out_of_sample(df, model_dict, feature_col, target_col,
                       split_date):
    # Prediction window and horizon values
    horizon_values = [0]
    roll_window = 150
    pred_window = 1
    threshold = 0.5  # threshold for binary classification
    k = 0
    err_df_rolling = pd.DataFrame([])
    for model_tuple, param in model_dict.items():
        t = df[df['Date'] == split_date].index.tolist()[0]
        # t is 154, almost 13 years into our time period which starts in 1960
        df_pred = df.iloc[t - roll_window:, :]
        # df_pred is 728 months long; df from split date to end is 578
        # thus we see df_pred starts 150 months before split date, since
        # our prediction window is 150 months prior to prediction month
        # model is fitted at every month in window of 150
        X = df_pred[feature_col]
        y = df_pred[target_col]
        date_range = df_pred['Date']
        y_prob = np.array([])
        date = np.array([], dtype='datetime64[s]')
        y_actual = np.array([])
        y_binary = np.array([])
        for i in np.arange(0, len(df_pred) - roll_window):
            model = model_tuple[1](**param)
            X_fit = X.iloc[i: i + roll_window, :]
            y_fit = y.iloc[i: i + roll_window]
            model = model.fit(X_fit, y_fit)
            X_predict = X.iloc[i + roll_window: i + roll_window + 1, :]
            y_pred = model.predict_proba(X_predict)[:, 1]
            # append class 1 probabilities
            y_prob = np.hstack((y_prob, y_pred))
            y_binary = np.hstack((y_binary, 1 if y_pred >= threshold else 0))
            date = np.hstack((date,
                              date_range.iloc[i + roll_window:
                                              i + roll_window + 1].values))
            y_actual = np.hstack((y_actual,
                                  y.iloc[i + roll_window:
                                         i + roll_window + 1].values))

        res_model_df = pd.DataFrame.from_dict({'Date': date,
                                               'Regime': y_actual,
                                               'crash_prob': y_prob,
                                               'crash_binary': y_binary})
        # 578 rows x 4 columns
        if k == 0:
            # initializes df with Date, Regime, and model abbrev as columns
            res_rolling_all = pd.DataFrame.from_dict({'Date': date,
                                                      'Regime': y_actual,
                                                      model_tuple[0]: y_prob})
            k += 1
        else:
            res_rolling_all[model_tuple[0]] = y_prob
            # y_prob is the y probabilities, length = 578
            # Date       Regime  LR      LR-l1 etc...
            # ========   ======  =======  ======
            # 1973-01-01  0      3.4e-9  etc.

        err_dict = error_metrics(res_model_df, roc=True)
        err_dict['Model'] = model_tuple[0]
        err_df_rolling = err_df_rolling.append(pd.DataFrame([err_dict]))
    return err_df_rolling, res_rolling_all


if __name__ == '__main__':
    # Plot regimes vs. SP500
    # regimes found at
    # https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions
    df_regime = pd.read_csv('data/recession_dates.csv', parse_dates=['Date'])
    df_sp500 = pd.read_csv('data/sp500.csv', parse_dates=['Date'],
                           usecols=['Date', 'Close'])
    print('Number of recession periods: ',
          df_regime[df_regime['Regime'] == 1].shape[0])
    print('Number of expansion periods: ',
          df_regime[df_regime['Regime'] == 0].shape[0])
    df_recession = df_regime.merge(df_sp500, on='Date', how='left')
    df = RegimePlot(df=df_recession, regime_col='Regime')
    df.plt_regime(plt_series=['Close'], series_label=['SP500'],
                  regime_label='Recession', orj_series=True,
                  log_scale=True)

    # =========================================================================
    # Clean and process data

    url = 'https://files.stlouisfed.org/files/htdocs/fred-md/monthly/current'
    url = url + '.csv'
    df_macro = pd.read_csv(url)
    df_macro.rename(columns={'sasdate': 'Date'}, inplace=True)
    while df_macro['Date'].iloc[-1:].isna().iloc[-1]:
        df_macro = df_macro.iloc[0:-1]

    # macro_raw.csv is file downloaded from coursera in May 2021
    # we check to see if Fed data has been updated, save the file if so,
    # and then clean and process the updated data
    df_macro2 = pd.read_csv('data/macro_raw.csv')
    df_macro2.rename(columns={'sasdate': 'Date'}, inplace=True)

    if df_macro2.Date.iloc[-1] < df_macro.Date.iloc[-1]:
        print(f'New date and data downloaded {df_macro.Date.iloc[-1]}')
        new_month = pd.to_datetime(df_macro.Date.iloc[-1])
        filename = './data/macro_raw_' + new_month.strftime('%Y_%m') + '.csv'
        df_macro.to_csv(filename, index=False)

    df_clean = clean_data(df_macro, missing_num=10)
    df_process = process_data(df_clean)
    print('Cleaned and Processed')

    # Data Set-up
    df = df_process.merge(df_regime, on='Date', how='left')
    split_date = '1973-01-01'  # train and test set split date
    df_train = df[df['Date'] < split_date]
    df_test = df[df['Date'] >= split_date]
    target_col = 'Regime'
    feature_col = df.columns.drop(['Regime', 'Date'])

    # %% Cross-Validation
    model_dict_cv =\
        {('LR', LogisticRegression): {'solver': ['saga'],
                                      'penalty': ['none'],
                                      'max_iter': [100]},
         ('LR_l1', LogisticRegression): {'solver': ['saga'],
                                         'max_iter': [100],
                                         'penalty': ['l1'],
                                         'C': [0.0001, 0.001, 0.01, 0.1, 1,
                                               10, 100]},
         ('LR_l2', LogisticRegression): {'solver': ['saga'],
                                         'max_iter': [100],
                                         'penalty': ['l2'],
                                         'C': [0.0001, 0.001, 0.01, 0.1, 1,
                                               10, 100]},
         ('DT', DecisionTreeClassifier): {'max_depth': [3, 5, 8, 10],
                                          'splitter': ['best', 'random'],
                                          'min_samples_split': [2, 3, 5]},
         ('RF', RandomForestClassifier): {'random_state': [42],
                                          'max_depth': [3, 5, 8, 10],
                                          'n_estimators': [100, 200, 400]},
         ('XGB', xgb.XGBClassifier): {'booster': ['gbtree'],
                                      'max_depth': [3, 5, 8, 10],
                                      'n_estimators': [100, 200, 400],
                                      'random_state': [42],
                                      'objective': ['binary:logistic']}}

    start = time.time()
    model_dict = cross_valid(model_dict_cv, df_train, target_col, feature_col)
    end = time.time()
    print(f'\nTime to execute cross validation: {round(end-start, 1)} seconds')

# =============================================================================
# as of 5/16/21 run
# {('LR', sklearn.linear_model._logistic.LogisticRegression):
#    {'solver': 'saga', 'penalty': 'none', 'max_iter': 100},

#  ('LR_l1', sklearn.linear_model._logistic.LogisticRegression):
#   {'solver': 'saga', 'max_iter': 100, 'penalty': 'l1', 'C': 0.01},

#  ('LR_l2', sklearn.linear_model._logistic.LogisticRegression):
#    {'solver': 'saga', 'max_iter': 100, 'penalty': 'l2', 'C': 0.0001},

#  ('DT', sklearn.tree._classes.DecisionTreeClassifier):
#    {'max_depth': 10, 'splitter': 'random', 'min_samples_split': 5},

#  ('RF', sklearn.ensemble._forest.RandomForestClassifier):
#    {'random_state': 42, 'max_depth': 3, 'n_estimators': 200},

#  ('XGB', xgboost.sklearn.XGBClassifier):
#    {'booster': 'gbtree', 'max_depth': 3, 'n_estimators': 100,
#   'random_state': 42, 'objective': 'binary:logistic'}}
# =============================================================================

    # %% Out-of-sample Prediction
    # the below takes 10 minutes or so
    # Out-of-sample predictions are performed on a rolling window basis.
    # Model performances are evaluated with error metrics and recession
    # prediction probabilities are visualized after that.
    start = time.time()
    err_df_rolling, res_rolling_all =\
        test_out_of_sample(df, model_dict, feature_col,
                           target_col, split_date)

    # =========================================================================
    # err_df_rolling data frame
    #   ACC  MCC   QPS   AUC   Model
    # 0.878  0.50  0.09  0.86   LR
    # 0.xx   0.xx  0.xx  0.xx   LR_l1
    # 0.xx   0.xx  0.xx  0.xx   LR_l2
    # 0.xx   0.xx  0.xx  0.xx   DT
    # 0.xx   0.xx  0.xx  0.xx   RF
    # 0.xx   0.xx  0.xx  0.xx   XGB
    # =========================================================================
    end = time.time()
    print(f'\nDuration of Out of sample predictions: '
          f'{round(end-start, 1)/60} min')

    fig, axs = plt.subplots(1, 4, figsize=(24, 4))
    for idx, name in enumerate(err_df_rolling.columns.drop('Model')):
        axs[idx].plot(err_df_rolling['Model'],
                      err_df_rolling[name], marker='o')
        axs[idx].set_title(name, fontsize=14)
    plt.suptitle('Out-of-sample Error Metrics for each Model', fontsize=16)
    plt.show()

    for model in list(model_dict.keys()):
        df = RegimePlot(df=res_rolling_all, regime_col='Regime')
        df.plt_regime(plt_series=[model[0]], series_label=[model[0]],
                      regime_label='Recession',
                      orj_series=True, log_scale=False,
                      title='Recession Prediction Probabilities with '
                      + model[0])
