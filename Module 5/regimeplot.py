import matplotlib.pyplot as plt


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
        for i, j, k in zip(self.df[self.regime], self.df['Date'], range(len(self.df))):
            if i == crash_regime and regime == normal_regime:  # regime switch from normal to crash
                regime_span = []
                regime = crash_regime
                regime_span.append(j)  # start of the crash regime date
            if i == normal_regime and regime == crash_regime:  # end of crash regime
                regime = normal_regime
                regime_span.append(self.df['Date'].iloc[k - 1])  # take end date from previous iteration
                regime_dates.append(regime_span)  # crash regime span [start_date,end_date]
            if i == crash_regime and j == self.df['Date'].iloc[-1]:
                # if we are in crash regime at the end of dataset append the last date
                regime_span.append(j)
                regime_dates.append(regime_span)
        return regime_dates

    def plt_regime(self, plt_series: list, series_label: list, regime_label: str,log_scale=True, title=None, orj_series=False):
        # Plot return or cumulative returns (of multiple series) over single regime label

        plt.figure(figsize=(18, 6))
        plt.xlabel(' ')
        plt.ylabel(' ')
        #plt.ylim([1])
        if orj_series:
            # if True plt original series
            for i in range(len(plt_series)):
                plt.plot(self.df['Date'], self.df[plt_series[i]], label=series_label[i])
        else:
            for i in range(len(plt_series)):
                plt.plot(self.df['Date'], (1 + self.df[plt_series[i]]).cumprod(), label=series_label[i])
        if log_scale: plt.yscale('log')

        for i in range(len(self.regime_dates)):
            if i != len(self.regime_dates) - 1:
                plt.axvspan(self.regime_dates[i][0], self.regime_dates[i][1], alpha=0.30, color='grey')
            else:
                plt.axvspan(self.regime_dates[i][0], self.regime_dates[i][1], alpha=0.30, color='grey',
                            label=regime_label)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), fontsize=12, frameon=False,
                   ncol=len(plt_series) + 1)
        if title:
            plt.title(title, fontsize=18)
        else:
            plt.title('Cumulative Performance Over time', fontsize=18)

        plt.show()


