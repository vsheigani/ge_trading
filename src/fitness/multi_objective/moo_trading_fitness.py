import numpy as np
import pandas as pd
import math
from fitness.base_ff_classes.base_ff import base_ff
from pathlib import Path
from utilities.trading.indicators import make_indicator
from utilities.trading.metrics import calculate_cagr, calculate_sharpe_ratio
from utilities.trading.strategy import cross, cross_num, count_negative, count_positive, shrink, widen


bars_file = 'datasets/bars/aapl_dollar_bars.h5'



class dummyfit(base_ff):
    maximise = True
    default_fitness = 0.
    label = "dummy"
    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()


class moo_trading_fitness:
    maximise = True
    multi_objective = True

    def __init__(self):
        new_inds = ['KAMA', 'MAMA', 'FAMA',  'MIDPOINT', 'MIDPRICE', 'SAR', 'T3', 'SMA',
              'TEMA', 'TRIMA', 'WMA', 'ROCR100', 'ULTOSC', 'AD', 'ADOSC', 'OBV', 'ATR', 'NATR',
              'TRANGE', 'AVGPRICE', 'MEDPRICE', 'TYPPRICE', 'WCLPRICE', 'HT_DCPERIOD', 'HT_DCPHASE',
              'IN_PHASOR', 'QUAD_PHASOR', 'SINE', 'LEAD_SINE',  'HT_TRENDMODE', 'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE',
              'CDL3LINESTRIKE', 'CDL3OUTSIDE', 'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS', 'CDLABANDONEDBABY',
              'CDLADVANCEBLOCK', 'CDLBELTHOLD', 'CDLBREAKAWAY', 'CDLCLOSINGMARUBOZU', 'CDLCONCEALBABYSWALL',
              'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER', 'CDLDOJI',  'CDLDOJISTAR', 'CDLDRAGONFLYDOJI', 'CDLENGULFING', 'CDLEVENINGDOJISTAR', 
              'CDLEVENINGSTAR', 'CDLGAPSIDESIDEWHITE', 'CDLGRAVESTONEDOJI', 'CDLHAMMER', 
              'CDLHANGINGMAN', 'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE', 'CDLHIKKAKE', 
              'CDLHIKKAKEMOD', 'CDLHOMINGPIGEON',  'CDLIDENTICAL3CROWS',  'CDLINNECK', 
              'CDLINVERTEDHAMMER', 'CDLKICKING', 'CDLKICKINGBYLENGTH', 'CDLLADDERBOTTOM', 
              'CDLLONGLEGGEDDOJI', 'CDLLONGLINE', 'CDLMARUBOZU', 'CDLMATCHINGLOW', 'CDLMATHOLD', 'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR',
              'CDLONNECK', 'CDLPIERCING', 'CDLRICKSHAWMAN', 'CDLRISEFALL3METHODS', 
              'CDLSEPARATINGLINES',  'CDLSHOOTINGSTAR', 'CDLSHORTLINE', 'CDLSPINNINGTOP',
              'CDLSTALLEDPATTERN', 'CDLSTICKSANDWICH', 'CDLTAKURI', 'CDLTASUKIGAP',
              'CDLTHRUSTING', 'CDLTRISTAR', 'CDLUNIQUE3RIVER', 'CDLUPSIDEGAP2CROWS', 'CDLXSIDEGAP3METHODS']

        ind_ls = ['MA', 'DEMA', 'EMA', 'HT_TRENDLINE', 'ADX', 'RSI', 'MACD', 'BBANDS_U', 'BBANDS_L', 'STOCH_D', 'STOCH_K',
                  'APO', 'ADXR', 'AROON_U', 'AROON_D', 'AROONOSC', 'CMO', 'CCI', 'DX', 'MFI', 'MINUS_DI', 'MINUS_DM', 'MOM',
                  'PLUS_DI', 'PPO', 'PLUS_DM', 'ROC', 'ROCP', 'ROCR', 'STOCHRSI_D', 'STOCHRSI_K', 'TRIX', 'WILLR', 'BOP', 'BETA', 'CORREL',
                  'LINEARREG', 'LINEARREG_ANGLE', 'LINEARREG_INTERCEPT', 'LINEARREG_SLOPE', "STDDEV", 'TSF', 'VAR',
                  'MIN', 'MAX', 'HIGH_SKEW', 'LOW_SKEW', 'HIGH_KURT', 'LOW_KURT', 'KURT', 'SKEW', 'MININDEX','MAXINDEX'] + \
                 ['Open', 'High', 'Low', 'Close', 'Volume'] + \
                 ['day', 'dayofweek', 'hour', 'minute', 'minute_of_day', 'minute_of_week', 'minute_of_month'] + new_inds
        # last_train_date = '2022-07-08'
        last_train_date = '2016-01-01'

        aapl_dollar_bars_path = Path().resolve().parent / bars_file
        train_bars = pd.read_hdf(aapl_dollar_bars_path, key='key')
        
        if train_bars.index.name != 'date_time':
            train_bars = train_bars.set_index('date_time')
        train_bars = train_bars[['open', 'high', 'low', 'close', 'volume']]

        cagr_fitness = dummyfit()
        sharpe_fitness = dummyfit()
        cagr_fitness.default_fitness = 1e-9
        sharpe_fitness.default_fitness = 1e-9
        cagr_fitness.label = "cagr"
        sharpe_fitness.label = "sharpe_ratio"

        self.fitness_functions = [sharpe_fitness, cagr_fitness]
        self.num_obj = len(self.fitness_functions)

        self.bars = train_bars[train_bars.index < last_train_date]
        self.ind_ls = ind_ls
        self.cross = cross
        self.cross_num = cross_num
        self.widen = widen
        self.shrink = shrink
        self.count_negative = count_negative
        self.count_positive = count_positive
        self.make_indicator = make_indicator

        self.default_fitness = []
        for f in self.fitness_functions:
            self.default_fitness.append(f.default_fitness)


    def _backtest(self, trades, exit_bar, bars):
        counter = 0
        purchase_bar = 0
        data_ls = []
        active_pos = []
        buy_price = 0
        sell_price = 0
        init_cash = 100000
        cash = init_cash
        trade_cost = 2.
        slippage = 0.0005
        trades.iloc[0] = 0
        trades.iloc[1] = 0
        num_trades = 0
        time_ls = []
        trades_log = []

        for row in bars.itertuples():
            if trades.iloc[counter] == 1:
                if len(active_pos)  == 0:
                    buy_price = (1 + slippage) * row[4]
                    cash = cash - trade_cost
                    if cash <= 0:
                        return [np.nan, np.nan]
                    size = math.floor(cash/buy_price)
                    # if size == 0:
                    #     data_ls.append(cash)
                    #     continue
                    cash = cash - size * buy_price
                    active_pos.append((buy_price, size, counter))
                    self.long_start = row[0]
                elif len(active_pos) == 1:
                    if active_pos[-1][1] < 0:
                        sell_price, size, _ = active_pos.pop(0)
                        buyback_price = (1 + slippage) * row[4]
                        cash = cash - abs(size) * buyback_price
                        num_trades += 1
                        self.short_end = row[0]
                        time_ls.append(self.short_end-self.short_start)
                        trades_log.append(sell_price-buyback_price)

            elif trades.iloc[counter] == -1:
                if len(active_pos) == 0:
                    sell_price = (1 - slippage) * row[4]
                    cash = cash - trade_cost
                    if cash <= 0:
                        return [np.nan, np.nan]
                    size = math.floor(cash/sell_price)
                    # if size == 0:
                    #     continue
                    cash = cash + size * sell_price
                    active_pos.append((sell_price, -size, counter))
                    self.short_start = row[0]
                elif len(active_pos) == 1:
                    if active_pos[-1][1] > 0:
                        sell_price = (1 - slippage) * row[4]
                        buy_price, size, _ = active_pos.pop(0)
                        cash = cash + size * sell_price
                        num_trades += 1
                        self.long_end = row[0]
                        time_ls.append(self.long_end-self.long_start)
                        trades_log.append(sell_price-buy_price)

            for i, pos in enumerate(active_pos):
                curr_price = row[4]
                _, size, purchase_bar = pos
                if counter >= purchase_bar + exit_bar:
                    if size > 0:
                        buy_price, size, _ = active_pos.pop(i)
                        sell_price = (1 - slippage) * curr_price
                        cash = cash + size * sell_price
                        num_trades += 1
                        self.long_end = row[0]
                        time_ls.append(self.long_end-self.long_start)
                        trades_log.append(sell_price-buy_price)
                    elif size < 0:
                        sell_price, size, _ = active_pos.pop(i)
                        buyback_price = (1 + slippage) * curr_price
                        cash = cash - abs(size) * buyback_price
                        num_trades += 1
                        self.short_end = row[0]
                        time_ls.append(self.short_end-self.short_start)
                        trades_log.append(sell_price-buyback_price)

            counter += 1
            portfolio_value = 0
            for pos in active_pos:
                _, size, _ = pos
                curr_price = row[4]
                portfolio_value += size * curr_price
            data_ls.append(cash + portfolio_value)

        portfolio = pd.Series(data=data_ls, index=bars.index, name='close')
        cagr = calculate_cagr(portfolio)
        sharpe_ratio = calculate_sharpe_ratio(portfolio)

        # if num_trades < 50:
        #     return [np.nan, np.nan]
        return [sharpe_ratio, cagr] 


    def __call__(self, ind):
        """
        Note that math functions used in the solutions are imported from either
        utilities.fitness.math_functions or called from numpy.
        :param ind: An individual to be evaluated.
        :return: The fitness of the evaluated individual.
        """

        # the multi-objective fitness is defined as a list of values, each one
        # representing the output of one objective function. The computation is
        # made by the function multi_objc_eval, implemented by a subclass,
        # according to the problem.
        # fitness = [ff(ind) for ff in self.fitness_functions]
        p, d = ind.phenotype, {'bars': self.bars, 'ind_ls': self.ind_ls,
         'cross': self.cross, 'cross_num': self.cross_num, 'widen':self.widen,
          'shrink':self.shrink, 'count_positive': self.count_positive, 'count_negative': self.count_negative,
          'make_indicator': self.make_indicator, 'pd': pd, 'np': np}
        # fitness = [np.nan, np.nan]
        try:
            # Evaluate the fitness using the evaluate() function. This function
            # can be over-written by classes which inherit from this base
            # class.

            exec(p, d)
            trades = d['_results_']
            exit_bar = d['_exit_bar_']
            fitness = self._backtest(trades, exit_bar, self.bars)

        except (FloatingPointError, ZeroDivisionError, OverflowError,
                MemoryError):
            # FP err can happen through eg overflow (lots of pow/exp calls)
            # ZeroDiv can happen when using unprotected operators
            fitness = [np.nan, np.nan]
            # These individuals are valid (i.e. not invalids), but they have
            # produced a runtime error.
            ind.runtime_error = True
        
        except Exception as err:
            # Other errors should not usually happen (unless we have
            # an unprotected operator) so user would prefer to see them.
            print(f"Error in running {p}\n error:{err}")
            fitness = [np.nan, np.nan]
            # raise 
        if any([np.isnan(i) for i in fitness]):
            # Check if any objective fitness value is NaN, if so set default
            # fitness.
            fitness = [1e-9, 1e-9]

        return fitness

    @staticmethod
    def value(fitness_vector, objective_index):
        """
        This is a static method required by NSGA-II for sorting populations
        based on a given fitness function, or for returning a given index of a
        population based on a given fitness function.

        :param fitness_vector: A vector/list of fitnesses.
        :param objective_index: The index of the desired fitness.
        :return: The fitness at the objective index of the fitness vector.
        """

        if not isinstance(fitness_vector, list):
            return float("inf")

        return fitness_vector[objective_index]