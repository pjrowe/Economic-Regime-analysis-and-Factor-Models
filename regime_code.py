import numpy as np
from pandas.plotting import register_matplotlib_converters
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import cvxpy as cp
import seaborn as sns
# sns.set()
# register_matplotlib_converters()
from sklearn.linear_model import LinearRegression


def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements.

    ECDF = Empirical Cumulative Distribution Functions
    """
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n + 1) / n
    return x, y


def regime_return(asset_pd, column_number, regime_col):
    """Compute returns of a list of regime columns identified by number."""
    asset_name = asset_pd.columns[column_number]
    regime = asset_pd[regime_col].values[:-1]
    asset_return = (np.diff(asset_pd[asset_name], axis=0)
                    / asset_pd[asset_name].values[:-1:])
    ret_g, ret_c = asset_return[regime == 1], asset_return[regime == -1]
    return asset_return, ret_g, ret_c


def regime_hist(asset_pd, column_number, regime_col):
    """Plot the asset regimes with regime column identified by number."""
    asset_return, ret_g, ret_c = regime_return(asset_pd,
                                               column_number, regime_col)
    asset_name = asset_pd.columns[column_number]
    plt.hist(ret_g, bins=20, color='green',
             label='Growth Regime', alpha=0.3)
    plt.hist(ret_c, bins=15, color='red',
             label='Contraction Regime', alpha=0.3)

    plt.xlabel('Monthly Return')
    plt.ylabel('Frequency')
    plt.legend(loc='upper left')
    plt.title('Regime Histogram of Asset: ' + asset_name)
    return ret_g, ret_c


def Q_Q_plot(asset_data, column_num, ret):
    plt.figure(figsize=(12, 9))
    res = scipy.stats.probplot(ret[:, column_num], plot=plt)
    plt.title('Q-Q Plot of Asset: '
              + asset_data.columns[column_num], fontsize=24)
    plt.ylabel('Returns')
    plt.show()


def regime_plot(asset_data, column_num, regime_col):
    asset_return, ret_g, ret_c = regime_return(asset_data,
                                               [0, 1, 2, 3, 4, 5, 6, 7],
                                               regime_col)
    ret_g1 = ret_g[:, column_num]
    ret_c1 = ret_c[:, column_num]
    plt.figure(figsize=(12, 9))
    plt.plot(ecdf(ret_g1)[0], ecdf(ret_g1)[1], color='green',
             label='Normal Regime')
    plt.plot(ecdf(ret_c1)[0], ecdf(ret_c1)[1], color='red',
             label='Crash Regime')
    plt.xlabel('Monthly Return')
    plt.ylabel('Cumulative Probability')
    plt.legend(loc='upper left')
    plt.title('Cumulative Density of Asset: ' + asset_data.columns[column_num],
              fontsize=24)
    plt.show()


def trend_filtering(data, lambda_value):
    """Run trend-filtering algorithm to separate regimes.

    the betas are can be turned into square wave by testing against
    threshhold level with regime_switch() function
        data: numpy array of total returns.

    <Phil Rowe>:
    beta.value output is an array of length n that has values that
    minimize the problem; regime_switch function below
    detects at which indices of beta.value there is a change in
    regime by testing against a threshold
    """
    n = np.size(data)
    x_ret = data.reshape(n)

    Dfull = np.diag([1] * n) - np.diag([1] * (n - 1), 1)
    D = Dfull[0:(n - 1), ]

    beta = cp.Variable(n)
    lambd = cp.Parameter(nonneg=True)

    def tf_obj(x, beta, lambd):
        return (cp.norm(x - beta, 2)**2
                + lambd * cp.norm(cp.matmul(D, beta), 1))

    problem = cp.Problem(cp.Minimize(tf_obj(x_ret, beta, lambd)))
    lambd.value = lambda_value
    problem.solve()
    return beta.value


def filter_plot(data, lambda_value, regime_num=0, TR_num=1):
    """Plot original data vs. filtered data.

    in our particular application, the original data was monthly
    returns of SP500 and the filtered data
    (passed throuh trend_filtering() function was a set of beta values,
     or 'regimes', of the original data.
    The ‘fitted’ time series serves as the signal of the trend, and is
    an alternative approach to detecting the economic regime
    that tries to avoid the problems with timing / noise that result
    from trying to predict economic regime changes with macroeconomic
    data
    """
    ret_sp = data.iloc[:, regime_num]
    # monthly returns of sp500
    sp500TR = data.values[:, TR_num]
    # not used; not sure why we need; this is actual index level

    beta_value = trend_filtering(ret_sp.values, lambda_value)
    betas = pd.Series(beta_value, index=data.index)

    plt.figure(figsize=(12, 9))
    plt.plot(ret_sp, alpha=0.4, label='Original Series')
    plt.plot(betas, label='Fitted Series')
    plt.xlabel('Year')
    plt.ylabel('Monthly Return (%)')
    plt.legend(loc='upper left')
    plt.show()
    return betas


def regime_switch(betas, threshold=1e-5):
    """Return list of starting points of each regime."""
    n = len(betas)
    init_points = [0]
    curr_reg = (betas[0] > threshold)
    for i in range(n):
        if (betas[i] > threshold) == (not curr_reg):
            curr_reg = not curr_reg
            init_points.append(i)
    init_points.append(n)
    # regimelist[-1] will be the length of betas
    return init_points


def plot_regime_color(dataset, regime_num=0, TR_num=1,
                      lambda_value=16, log_TR=True):
    """Plot of return series versus regime."""
    returns = dataset.iloc[:, regime_num]
    TR = dataset.iloc[:, TR_num]
    betas = trend_filtering(returns.values, lambda_value)
    regimelist = regime_switch(betas)
    # regimelist[-1] will be the length of betas;
    # this is used to normalize x axis to 1
    curr_reg = np.sign(betas[0] - 1e-5)
    y_max = np.max(TR) + 500

    if log_TR:
        fig, ax = plt.subplots()
        for i in range(len(regimelist) - 1):
            if curr_reg == 1:
                ax.axhspan(0, y_max + 500, xmin=regimelist[i] / regimelist[-1],
                           xmax=regimelist[i + 1] / regimelist[-1],
                           facecolor='green', alpha=0.3)
            else:
                ax.axhspan(0, y_max + 500, xmin=regimelist[i] / regimelist[-1],
                           xmax=regimelist[i + 1] / regimelist[-1],
                           facecolor='red', alpha=0.5)
            curr_reg = -1 * curr_reg

        fig.set_size_inches(12, 9)
        plt.plot(TR, label='Total Return')
        # plot sp500 value, which is in TR
        plt.ylabel('SP500 Log-scale')
        plt.xlabel('Year')
        plt.yscale('log')
        plt.xlim([dataset.index[0], dataset.index[-1]])
        plt.ylim([80, 3500])
        plt.yticks([100, 500, 1000, 2000, 3000, 3500],
                   [100, 500, 1000, 2000, 3000, 3500])
        plt.title('Regime Plot of SP 500', fontsize=24)
        plt.show()


def geo_return(X, input_type='Return'):
    """Compute geometric return for each asset."""
    if input_type == 'Return':
        X_geo = 1 + X  # rows are months, columns are different assets
        y = np.cumprod(X_geo, axis=0)
        return (y[-1, :]) ** (1 / X.shape[0]) - 1
    # y will have same dimensions as X, but each successive month of y
    # will be cumulative return
    # hence, to get average per month return, we take cum return
    # (very last row of
    # y, andtake root of nth power where n = # of timesteps

    else:
        # this works if X entries are asset balances or values at each time
        # step, rather than # returns
        # it returns the monthly geometric mean (assuming the X.shape is
        # monthly as well)
        return (X[-1, :] / X[0, :])**(1 / (X.shape[0] - 1)) - 1


def portfolio_opt(mu, Q, r_bar):
    w = cp.Variable(mu.size)
    ret = mu.T * w
    risk = cp.quad_form(w, Q)
    prob = cp.Problem(cp.Minimize(risk),
                      [cp.sum(w) == 1, w >= 0, ret >= r_bar])
    prob.solve()
    return np.round_(w.value, decimals=3)


def efficient_frontier_traditional(r_annual, Q_all, r_bar, asset_names):

    n_asset = r_annual.size
    # this is 8 for original implementation, as there are 8 assets to choose
    weight_vec = np.zeros((len(r_bar), n_asset))
    # there are 16 different potential portfolio returns we want on frontier
    # there are 8 assets, so 16x8 is weights for each of 16 returns
    risk_port = np.zeros(len(r_bar))
    # 16x1 risk values for each of 16 potential portfolio returns (x,y)
    ret_port = np.zeros(len(r_bar))
    # 16 return values for portfolio

    for i, r_curr in enumerate(r_bar):
        w_opt = cp.Variable(r_annual.size)
        # 8x1, weight variables to solve below, minimizing risk with
        # constraints
        ret_opt = r_annual.T * w_opt
        # optimized return is the r_annual * optimized weights
        risk_opt = cp.quad_form(w_opt, Q_all)
        # executes w_opt.T * Q_all * w_opt
        # Total variance = weighting individual variances with w^2
        #  = sum(w^2*Var for each asset)
        # (8x1).T * 8x8  * 8x1 ==> 1x1 = portfolio variance
        prob = cp.Problem(cp.Minimize(risk_opt),
                          [cp.sum(w_opt) == 1, w_opt >= 0, ret_opt >= r_curr])
        # finds weights that minimizes risk and maximizes return
        prob.solve()

        weight_vec[i, :] = w_opt.value
        ret_port[i] = ret_opt.value
        risk_port[i] = np.sqrt(risk_opt.value)

    plt.figure(figsize=(12, 9))
    plt.plot(risk_port * 100, ret_port * 100, 'xb-')
    plt.xlabel("Risk (%)")
    plt.ylabel("Nominal Return (%)")
    plt.title("Efficient Frontier: Single-Period", fontsize=24)
#    print(pd.DataFrame(data=np.around(weight_vec, 3), columns=asset_names))


# =============================================================================
# def efficient_frontier_scenario(r_all_1, r_bar):
#     Q_1 = np.cov(r_all_1.T)
#     mu_1 = np.mean(r_all_1, axis=0)
#     efficient_frontier_traditional(r_annual, Q_all, r_bar)
#     plt.title("Efficient Frontier: Single-Period, Scenario-Equivalent",
#               fontsize=24)
#
# =============================================================================
def efficient_frontier_comparison(r_annual, Q_all, r_bar, n_scenarios=10000):
    n_asset = r_annual.size
    weight_vec = np.zeros((len(r_bar), n_asset))
    risk_port = np.zeros(len(r_bar))
    ret_port = np.zeros(len(r_bar))

    for i, r_curr in enumerate(r_bar):
        w_opt = cp.Variable(r_annual.size)
        ret_opt = r_annual.T * w_opt
        risk_opt = cp.quad_form(w_opt, Q_all)
        prob = cp.Problem(cp.Minimize(risk_opt),
                          [cp.sum(w_opt) == 1, w_opt >= 0, ret_opt >= r_curr])
        prob.solve()

        weight_vec[i, :] = w_opt.value
        ret_port[i] = ret_opt.value
        risk_port[i] = np.sqrt(risk_opt.value)
        # risk for chart is standard deviation

    plt.subplot(121)
    plt.plot(risk_port * 100, ret_port * 100, 'xb-')
    plt.xlabel("Risk (%)")
    plt.ylabel("Nominal Return (%)")
    plt.title("Traditional", fontsize=16)

    # the following generates n_scenarios of annual returns across all
    # assets, each return being a normal random variable with mean
    #  equivalent to geometric mean calculated from historical data,
    # and covariance = what we calculated from historical data
    # then we
    r_all_1 = np.random.multivariate_normal(r_annual.reshape(n_asset),
                                            Q_all, n_scenarios)

    Q_1 = np.cov(r_all_1.T)
    # np.cov takes needs each row to be a variable, and each column a
    # single observation
    # hence we need to transpose r_all_1
    mu_1 = np.mean(r_all_1, axis=0)
    # takes mean across 10,000 samples in 10,000x8 array and returns 8 means
    print('Mean returns from', n_scenarios, ' samples is ', np.around(mu_1, 4))
    print('Means calculated from historical ',
          np.around(r_annual.reshape(n_asset), 4))
    weight_vec1 = np.zeros((len(r_bar), n_asset))
    risk_port1 = np.zeros(len(r_bar))
    ret_port1 = np.zeros(len(r_bar))

    for i, r_curr in enumerate(r_bar):
        w_opt = cp.Variable(r_annual.size)
        ret_opt = mu_1.T * w_opt
        risk_opt = cp.quad_form(w_opt, Q_1)
        prob = cp.Problem(cp.Minimize(risk_opt),
                          [cp.sum(w_opt) == 1, w_opt >= 0, ret_opt >= r_curr])
        prob.solve()

        weight_vec1[i, :] = w_opt.value
        ret_port1[i] = ret_opt.value
        risk_port1[i] = np.sqrt(risk_opt.value)

    plt.subplot(122)
    plt.plot(risk_port1 * 100, ret_port1 * 100, 'xb-')
    plt.xlabel("Risk (%)")
    plt.title("Scenario-Equivalent", fontsize=16)


def efficient_frontier_twoRegime(ret, ret_g, ret_c, r_bar,
                                 asset_names, n_scenarios=10_000):
    Q_all = np.cov(ret.T) * 12
    r_annual = (1 + geo_return(ret))**12 - 1
    r_annual = r_annual.reshape(-1, 1)
    r_g = (1 + geo_return(ret_g))**12 - 1
    r_c = (1 + geo_return(ret_c))**12 - 1
    n_g = int(n_scenarios * ret_g.shape[0] / ret.shape[0])
    Q_g = np.cov(ret_g.T) * 12
    Q_c = np.cov(ret_c.T) * 12
    n_asset = r_annual.size

    # two regime model has some fraction of the 10,000 scenarios being
    # growth regime random variable
    # and remainder of 10,000 scenarios being a contraction regime
    # random variable
    s_1 = np.random.multivariate_normal(r_g, Q_g, n_g)
    s_2 = np.random.multivariate_normal(r_c, Q_c, n_scenarios - n_g)
    r_all_2 = np.vstack((s_1, s_2))

    Q_2 = np.cov(r_all_2.T)
    mu_2 = np.mean(r_all_2, axis=0)

    weight_vec2 = np.zeros((len(r_bar), n_asset))
    risk_port2 = np.zeros(len(r_bar))
    ret_port2 = np.zeros(len(r_bar))

    for i, r_curr in enumerate(r_bar):
        w_opt = cp.Variable(r_annual.size)
        ret_opt = mu_2.T * w_opt
        risk_opt = cp.quad_form(w_opt, Q_2)
        prob = cp.Problem(cp.Minimize(risk_opt),
                          [cp.sum(w_opt) == 1, w_opt >= 0,
                           ret_opt >= r_curr])
        prob.solve()

        weight_vec2[i, :] = w_opt.value
        ret_port2[i] = ret_opt.value
        risk_port2[i] = np.sqrt(risk_opt.value)

    efficient_frontier_traditional(r_annual, Q_all, r_bar, asset_names)
    # this plots traditional frontier
    plt.plot(risk_port2 * 100, ret_port2 * 100, 'xr-', label='Two-Regime')
    plt.legend(loc='best', ncol=2, shadow=True, fancybox=True, fontsize=16)
    plt.title("Efficient Frontier: Single-Period, Traditional vs Two-Regime Version")
    plt.show()
    return r_all_2


def regime_asset(n, mu1, mu2, Q1, Q2, p1, p2):
    # Endowment Simulation: still in progress
    s_1 = np.random.multivariate_normal(mu1, Q1, n).T
    # mu1 is average returns (geometric) for assets during growth regime
    # Q1 is covariance matrix during growth

    s_2 = np.random.multivariate_normal(mu2, Q2, n).T
    # mu2 is average return (geometric) for assets during contraction regime
    # Q2 is covariance matrix during contraction (historical)
    regime = np.ones(n)

    # p1 is assigned n1/(n1+n2), or Prob(stay in regime 1 when in
    # regime 1)
    # if random number generator generates a # > p1, then it falls into
    # 1-p1 territory, i.e., it's the case that we switch
    #    to regime 0 when currently in regime 1
    # p2 is assigned n3/(n3+n4), or Prob(switch to regime 1 when in 0)
    # in the same way, if random # is >p2, then it's calling for 1-p2,
    # or case where we stay in regime 0 in next step

    for i in range(n - 1):
        if regime[i] == 1:
            if np.random.rand() > p1:
                regime[i + 1] = 0
        else:
            if np.random.rand() > p2:
                regime[i + 1] = 0
    return (regime * s_1 + (1 - regime) * s_2).T
    # when regime[i]==1, then s_1 multivariate normal is generated for that
    #    scenario and month
    # when regime[i]==0, then s_0 multivariate normal is generated for that
    #     scenario and month


def transition_matrix(regime):
    """Compute the transition matrix given the regime vector.

    by summing the over the 'history' of the regime, we can calculate
    probabilities
    n1/(n1+n2) = prob of staying   in regime 1  in next step when
                 currently in regime=  1 ; this will be p1
    n2/(n1+n2) = prob of switching to regime -1 in next step when
                 currently in regime=  1 ; = 1-p1
    n3/(n3+n4) = prob of switching to regime 1  in next step when
                 currently in regime= -1 ; this will be p2
    n4/(n3+n4) = prob of staying   in regime -1 in next step when
                 currently in regime= -1 ; = 1-p2
    """
    n1, n2, n3, n4 = 0, 0, 0, 0
    for i in range(len(regime) - 1):
        if regime[i] == 1:
            if regime[i + 1] == 1:
                n1 += 1
            else:
                n2 += 1
        else:
            if regime[i + 1] == 1:
                n3 += 1
            else:
                n4 += 1
    return n1 / (n1 + n2), n2 / (n1 + n2), n3 / (n3 + n4), n4 / (n3 + n4)


def asset_simulation(assets_info, asset_num, regime_name,
                     random_seed=777, n_scenarios=10000, n_years=50):
    """Simulate regime-based monthly returns.

    assets_info is a pandas Dataframe containing asset total return
    indices; please refer to the dataset for format.
    asset_num is the number of assets we would like to use. By default,
    this should be the first few columns in dataset.
    regime_name is the column name of regime in the dataset.

    Returns a (n_year*12) * n_asset * n_scenario tensor for all asset
    information.
    """
    ret_all, ret_g, ret_c = regime_return(assets_info,
                                          np.arange(asset_num), 'Regime-5')
    regime = assets_info[regime_name].values[:-1]
    # lose 1 value from computing returns

    p1, _, p2, _ = transition_matrix(regime)
    # p1 is assigned n1/(n1+n2), or Prob(stay in regime 1 when in regime 1)
    # p2 is assigned n3/(n3+n4), or Prob(switch to regime 1 when in 0)

    mu1 = 1 + geo_return(ret_g)
    mu2 = 1 + geo_return(ret_c)
    Q1 = np.cov(ret_g.T)
    Q2 = np.cov(ret_c.T)
    r_all = np.zeros((n_years * 12, asset_num, n_scenarios))

    np.random.seed(random_seed)
    for i in range(n_scenarios):
        r_all[:, :, i] = regime_asset(n_years * 12, mu1, mu2, Q1, Q2, p1, p2)
    return r_all


def fund_simulation(holdings, asset_return, hold_type='fixed',
                    spending_rate=0.03):
    """Simulate monthly data of a fund for a certain number of years.

    asset_return should be total return, i.e. 1 plus the percentage
    return.
    if hold_type is "fixed" (by default), holdings is fixed mix
    if hold_type is a number, this is rebalance frequency (in months)
    if hold_type is "dynamic", dynamic portfolio optimization will be
    conducted (to be implemented...)

    The simulation returns a full path of wealth at the end of each
    year, so it is a n_scenarios*n_years matrix.
    """
    n_months, n_assets, n_scenarios = asset_return.shape
    wealth_path = np.zeros((n_scenarios, int(n_months / 12)))

    if hold_type == 'fixed':
        for i in range(n_scenarios):
            holdings_each = holdings
            for j in range(n_months):
                holdings_each = holdings_each * asset_return[j, :, i]
                if j % 12 == 0:
                    holdings_each = holdings_each * (1 - spending_rate)
                    wealth_path[i, int(j / 12)] = np.sum(holdings_each)
        return wealth_path

    elif type(hold_type) == int:
        for i in range(n_scenarios):
            holdings_each = holdings
            for j in range(n_months):
                holdings_each = holdings_each * asset_return[j, :, i]
                if j % hold_type == 0:  # Rebalance
                    asset_temp = np.sum(holdings_each)
                    holdings_each = asset_temp * holdings
                if j % 12 == 0:
                    holdings_each = holdings_each * (1 - spending_rate)
                    wealth_path[i, int(j / 12)] = np.sum(holdings_each)
        return wealth_path

    else:  # "Dynamic" -- to be implemented
        return 0
