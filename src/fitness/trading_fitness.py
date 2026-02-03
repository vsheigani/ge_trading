import numpy as np
import pandas as pd
import math
from fitness.base_ff_classes.base_ff import base_ff
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


def scale(ind: pd.Series) -> pd.Series:
    scaler = MinMaxScaler(feature_range=(0, 100))
    scaled_ind = scaler.fit_transform(ind.values.reshape(-1, 1))
    return pd.Series(scaled_ind.flatten(), name=ind.name)


def calculate_percent_return(series: pd.Series) -> float:
    return series.iloc[-1] / series.iloc[0] - 1


def calculate_volatility(series: pd.Series) -> float:
    start_date = series.index[0]
    end_date = series.index[-1]
    years_past = (end_date - start_date).days / 365.25

    shifted_series = series.shift(1, axis=0)
    return_series = series / shifted_series - 1
    entries_per_year = return_series.shape[0] / years_past
    volatility = return_series.std() * np.sqrt(entries_per_year)
    return volatility


def calculate_cagr(series: pd.Series) -> float:
    start_date = series.index[0]
    end_date = series.index[-1]
    years_past = (end_date - start_date).days / 365.25
    start_price = series.iloc[0]
    end_price = series.iloc[-1]
    value_factor = end_price / start_price
    cagr = (value_factor ** (1 / years_past)) - 1
    return cagr


def calculate_sharpe_ratio(series: pd.Series, benchmark_rate: float=0):
    start_date = series.index[0]
    end_date = series.index[-1]
    years_past = (end_date - start_date).days / 365.25
    start_price = series.iloc[0]
    end_price = series.iloc[-1]
    value_factor = end_price / start_price
    cagr = (value_factor ** (1 / years_past)) - 1

    shifted_series = series.shift(1, axis=0)
    return_series = series / shifted_series - 1
    entries_per_year = return_series.shape[0] / years_past

    volatility = return_series.std() * np.sqrt(entries_per_year)
    return (cagr - benchmark_rate) / volatility


def cross(ind1, ind2, is_above=True):
    current = ind1 > ind2
    previous = ind1.shift(1) < ind2.shift(1)
    cross = current & previous if is_above else ~current & ~previous
    return cross


def cross_num(ind1, num, is_above=True):
    current = ind1 > num
    previous = ind1.shift(1) < num
    cross = current & previous if is_above else ~current & ~previous
    exact = (ind1 == num)
    return (cross | exact)


def widen(ind1, ind2, period):
    res = (ind1.shift(1) - ind2.shift(1)) < (ind1 - ind2)
    for i in range(1, period):
        res = res & ((ind1.shift(i+1) - ind2.shift(i+1)) < (ind1.shift(i) - ind2.shift(i)))
    return res


def shrink(ind1, ind2, period):
    res = (ind1.shift(1) - ind2.shift(1)) > (ind1 - ind2)
    for i in range(1, period):
        res = res & ((ind1.shift(i+1) - ind2.shift(i+1)) > (ind1.shift(i) - ind2.shift(i)))
    return res


def count_positive(ind, period):
    return ((ind.rolling(window=period).agg(lambda x: (x > 0).sum())) > period-1)


def count_negative(ind, period):
    return ((ind.rolling(window=period).agg(lambda x: (x < 0).sum())) > period-1)


def make_ind(bars, ind_name, f, s, p):
    from talib import MA, HT_TRENDLINE, ADX, STOCH, RSI, MACD, BBANDS, DEMA, EMA, APO, ADXR,\
            AROON, AROONOSC, CMO, CCI, DX, MFI, MINUS_DI, MINUS_DM, MOM, PLUS_DI, PPO,\
            PLUS_DM, ROC, ROCP, ROCR, STOCHRSI, TRIX, WILLR, BOP, BETA, CORREL,\
            LINEARREG, LINEARREG_ANGLE, LINEARREG_INTERCEPT, LINEARREG_SLOPE, STDDEV, TSF, VAR,\
            MAX, MIN, MININDEX, MAXINDEX, KAMA, MAMA, MIDPOINT, MIDPRICE, SAR, T3, SMA, TEMA, TRIMA, \
            WMA, ROCR100, ULTOSC, AD, ADOSC, OBV, ATR, NATR, TRANGE, AVGPRICE, MEDPRICE, TYPPRICE, \
            WCLPRICE, HT_DCPERIOD, HT_DCPHASE, HT_PHASOR, HT_SINE,  HT_TRENDMODE,\
            CDL2CROWS, CDL3BLACKCROWS, CDL3INSIDE, CDL3LINESTRIKE, CDL3OUTSIDE, CDL3STARSINSOUTH,\
            CDL3WHITESOLDIERS,  CDLABANDONEDBABY,  CDLADVANCEBLOCK, CDLBELTHOLD, CDLBREAKAWAY,\
            CDLCLOSINGMARUBOZU, CDLCONCEALBABYSWALL, CDLCOUNTERATTACK, CDLDARKCLOUDCOVER,\
            CDLDOJI,  CDLDOJISTAR, CDLDRAGONFLYDOJI, CDLENGULFING, CDLEVENINGDOJISTAR, \
            CDLEVENINGSTAR, CDLGAPSIDESIDEWHITE, CDLGRAVESTONEDOJI, CDLHAMMER, \
            CDLHANGINGMAN, CDLHARAMI, CDLHARAMICROSS, CDLHIGHWAVE, CDLHIKKAKE, \
            CDLHIKKAKEMOD, CDLHOMINGPIGEON,  CDLIDENTICAL3CROWS,  CDLINNECK, \
            CDLINVERTEDHAMMER, CDLKICKING, CDLKICKINGBYLENGTH, CDLLADDERBOTTOM, \
            CDLLONGLEGGEDDOJI, CDLLONGLINE, CDLMARUBOZU, CDLMATCHINGLOW, CDLMATHOLD, CDLMORNINGDOJISTAR, CDLMORNINGSTAR,\
            CDLONNECK, CDLPIERCING, CDLRICKSHAWMAN, CDLRISEFALL3METHODS, \
            CDLSEPARATINGLINES,  CDLSHOOTINGSTAR, CDLSHORTLINE, CDLSPINNINGTOP,\
            CDLSTALLEDPATTERN, CDLSTICKSANDWICH, CDLTAKURI, CDLTASUKIGAP,\
            CDLTHRUSTING, CDLTRISTAR, CDLUNIQUE3RIVER, CDLUPSIDEGAP2CROWS, CDLXSIDEGAP3METHODS
    period = max(p + 1, 5)
    fast = min(f, s) + 2
    slow = max(f, s) + 2
    open = bars['open']
    high = bars['high']
    low = bars['low']
    close = bars['close']
    volume = bars['volume']

    single_arg = {'HT_TRENDLINE': HT_TRENDLINE}
    periodic = {'MA': MA, 'DEMA': DEMA, 'EMA': EMA, 'RSI': RSI, 'CMO': CMO,
                'MOM': MOM, 'ROC': ROC, 'ROCR': ROCR, 'ROCP': ROCP, 'ROCR100': ROCR100,
                'TRIX': TRIX, 'LINEARREG': LINEARREG, 'LINEARREG_ANGLE': LINEARREG_ANGLE,
                'LINEARREG_INTERCEPT': LINEARREG_INTERCEPT, 'LINEARREG_SLOPE': LINEARREG_SLOPE,
                'STDDEV': STDDEV, 'TSF': TSF, 'VAR': VAR, 'KAMA': KAMA, 'MIDPOINT': MIDPOINT,
                'T3': T3, 'SMA': SMA, 'TEMA': TEMA, 'TRIMA': TRIMA, 'WMA': WMA}
    two_highlow = {'BETA': BETA, 'CORREL': CORREL, 'MIDPRICE': MIDPRICE,}
    three_args = {'ADX': ADX, 'ADXR': ADXR, 'CCI': CCI, 'DX': DX,
                'MINUS_DI': MINUS_DI, 'PLUS_DI': PLUS_DI, 'ATR': ATR, 'NATR': NATR}

    close_only = {'HT_DCPERIOD': HT_DCPERIOD, 'HT_DCPHASE': HT_DCPHASE, 'HT_TRENDMODE': HT_TRENDMODE}
    all_four = {'CDL2CROWS': CDL2CROWS, 'CDL3BLACKCROWS': CDL3BLACKCROWS, 'CDL3INSIDE': CDL3INSIDE,
                'CDL3LINESTRIKE': CDL3LINESTRIKE, 'CDL3OUTSIDE': CDL3OUTSIDE, 'CDL3STARSINSOUTH': CDL3STARSINSOUTH,
                'CDL3WHITESOLDIERS': CDL3WHITESOLDIERS, 'CDLABANDONEDBABY': CDLABANDONEDBABY, 'CDLADVANCEBLOCK': CDLADVANCEBLOCK,
                'CDLBELTHOLD': CDLBELTHOLD, 'CDLBREAKAWAY': CDLBREAKAWAY, 'CDLCLOSINGMARUBOZU': CDLCLOSINGMARUBOZU, 
                'CDLCONCEALBABYSWALL': CDLCONCEALBABYSWALL, 'CDLCOUNTERATTACK': CDLCOUNTERATTACK, 'CDLDARKCLOUDCOVER': CDLDARKCLOUDCOVER,
                'CDLDOJI': CDLDOJI, 'CDLDOJISTAR': CDLDOJISTAR, 'CDLDRAGONFLYDOJI': CDLDRAGONFLYDOJI,
                'CDLENGULFING': CDLENGULFING, 'CDLEVENINGDOJISTAR': CDLEVENINGDOJISTAR, 'CDLEVENINGSTAR': CDLEVENINGSTAR,
                'CDLGAPSIDESIDEWHITE': CDLGAPSIDESIDEWHITE, 'CDLGRAVESTONEDOJI': CDLGRAVESTONEDOJI, 'CDLHAMMER': CDLHAMMER, 
                'CDLHANGINGMAN': CDLHANGINGMAN, 'CDLHARAMI': CDLHARAMI, 'CDLHARAMICROSS': CDLHARAMICROSS, 'CDLHIGHWAVE': CDLHIGHWAVE,
                'CDLHIKKAKE': CDLHIKKAKE, 'CDLHIKKAKEMOD': CDLHIKKAKEMOD, 'CDLHOMINGPIGEON': CDLHOMINGPIGEON, 'CDLIDENTICAL3CROWS': CDLIDENTICAL3CROWS, 
                'CDLINNECK': CDLINNECK, 'CDLINVERTEDHAMMER': CDLINVERTEDHAMMER, 'CDLKICKING': CDLKICKING, 'CDLKICKINGBYLENGTH': CDLKICKINGBYLENGTH,
                'CDLLADDERBOTTOM': CDLLADDERBOTTOM, 'CDLLONGLEGGEDDOJI': CDLLONGLEGGEDDOJI, 'CDLLONGLINE': CDLLONGLINE, 'CDLMARUBOZU': CDLMARUBOZU,
                'CDLMATCHINGLOW': CDLMATCHINGLOW, 'CDLMATHOLD': CDLMATHOLD, 'CDLMORNINGDOJISTAR': CDLMORNINGDOJISTAR, 'CDLMORNINGSTAR': CDLMORNINGSTAR,
                'CDLONNECK': CDLONNECK, 'CDLPIERCING': CDLPIERCING, 'CDLRICKSHAWMAN': CDLRICKSHAWMAN, 'CDLRISEFALL3METHODS': CDLRISEFALL3METHODS, 
                'CDLSEPARATINGLINES': CDLSEPARATINGLINES, 'CDLSHOOTINGSTAR': CDLSHOOTINGSTAR, 'CDLSHORTLINE': CDLSHORTLINE, 'CDLSPINNINGTOP': CDLSPINNINGTOP,
                'CDLSTALLEDPATTERN': CDLSTALLEDPATTERN, 'CDLSTICKSANDWICH':CDLSTICKSANDWICH, 'CDLTAKURI': CDLTAKURI, 'CDLTASUKIGAP': CDLTASUKIGAP,
                'CDLTHRUSTING': CDLTHRUSTING, 'CDLTRISTAR': CDLTRISTAR, 'CDLUNIQUE3RIVER': CDLUNIQUE3RIVER, 'CDLUPSIDEGAP2CROWS': CDLUPSIDEGAP2CROWS,
                'CDLXSIDEGAP3METHODS': CDLXSIDEGAP3METHODS}

    if ind_name in all_four.keys():
        return all_four[ind_name](open, high, low, close)
    if ind_name in single_arg.keys():
        return single_arg[ind_name](close)
    elif ind_name in periodic.keys():
        return periodic[ind_name](close, period)
    elif ind_name in two_highlow.keys():
        return two_highlow[ind_name](high, low, period)
    elif ind_name in three_args.keys():
        return three_args[ind_name](high, low, close, period)
    elif ind_name in close_only.keys():
        return close_only[ind_name](close)
    elif ind_name == 'Open':
        return open
    elif ind_name == 'Close':
        return close
    elif ind_name == 'High':
        return high
    elif ind_name == 'Low':
        return low
    elif ind_name == 'Volume':
        return volume
    elif ind_name == 'week':
        return bars.index.isocalendar().week
    elif ind_name == 'isoday':
        return bars.index.isocalendar().day
    elif ind_name == 'day':
        return pd.Series(bars.index.day, index=bars.index)
    elif ind_name == 'dayofweek':
        return pd.Series(bars.index.dayofweek, index=bars.index)
    elif ind_name == 'hour':
        return pd.Series(bars.index.hour, index=bars.index)
    elif ind_name == 'minute':
        return pd.Series(bars.index.minute, index=bars.index)
    elif ind_name == 'minute_of_day':
        return pd.Series(bars.index.hour*60 + bars.index.minute , index=bars.index)
    elif ind_name == 'minute_of_week':
        return pd.Series(bars.index.dayofweek*1440 + bars.index.hour*60 + bars.index.minute , index=bars.index)
    elif ind_name == 'minute_of_month':
        return pd.Series((bars.index.day-1)*1440 + bars.index.hour*60 + bars.index.minute , index=bars.index)
    elif ind_name == 'SAR':
        return SAR(high, low)
    elif ind_name == 'ULTOSC':
        return ULTOSC(high, low, close)
    elif ind_name == 'AD':
        return AD(high, low, close, volume)
    elif ind_name == 'ADOSC':
        return ADOSC(high, low, close, volume, fastperiod=fast, slowperiod=slow)
    elif ind_name == 'OBV':
        return OBV(close, volume)
    elif ind_name == 'TRANGE':
        return TRANGE(high, low, close)
    elif ind_name == 'AVGPRICE':
        return AVGPRICE(open, high, low, close)
    elif ind_name == 'MEDPRICE':
        return MEDPRICE(high, low)
    elif ind_name == 'TYPPRICE':
        return TYPPRICE(high, low, close)
    elif ind_name == 'WCLPRICE':
        return WCLPRICE(high, low, close)
    elif ind_name == 'STOCH_K':
        k, _ = STOCH(high, low, close, fastk_period=fast, slowk_period=slow, slowk_matype=0, slowd_period=slow, slowd_matype=0)
        return k
    elif ind_name == 'STOCH_D':
        _, d = STOCH(high, low, close, fastk_period=fast, slowk_period=slow, slowk_matype=0, slowd_period=slow, slowd_matype=0)
        return d
    elif ind_name == 'BBANDS_U':
        upperband, _, _ = BBANDS(close, timeperiod=period, nbdevup=slow, nbdevdn=fast, matype=0)
        return upperband
    elif ind_name == 'BBANDS_L':
        _, _, lowerband = BBANDS(close, timeperiod=period, nbdevup=slow, nbdevdn=fast, matype=0)
        return lowerband           
    elif ind_name == 'MAMA':
        mama, _ = MAMA(close)
        return mama        
    elif ind_name == 'FAMA':
        _, fama = MAMA(close)
        return fama
    elif ind_name == 'IN_PHASOR':
        inphase, _ = HT_PHASOR(close)
        return inphase
    elif ind_name == 'QUAD_PHASOR':
        _, quad = HT_PHASOR(close)
        return quad
    elif ind_name == 'SINE':
        sine, _ = HT_SINE(close)
        return sine
    elif ind_name == 'LEAD_SINE':
        _, lead = HT_SINE(close)
        return lead
    elif ind_name == 'MACD':
        macd, _, _ = MACD(close, fastperiod=fast, slowperiod=slow, signalperiod=(period%10)+1)
        return macd  
    elif ind_name == 'APO':
        apo = APO(close, fastperiod=fast, slowperiod=slow)
        return apo
    elif ind_name == 'PPO':
        ppo = PPO(close, fastperiod=fast, slowperiod=slow)
        return ppo
    elif ind_name == 'AROON_U':
        _, aroonup = AROON(high, low, timeperiod=period)
        return aroonup
    elif ind_name == 'AROON_D':
        aroondown, _ = AROON(high, low, timeperiod=period)
        return aroondown
    elif ind_name == 'AROONOSC':
        aroon = AROONOSC(high, low, timeperiod=period)
        return aroon
    elif ind_name == 'BOP':
        aroon = BOP(open, high, low, close)
        return aroon
    elif ind_name == 'MINUS_DM':
        aroon = MINUS_DM(high, low, period)
        return aroon
    elif ind_name == 'MFI':
        aroon = MFI(high, low, close, volume, period)
        return aroon
    elif ind_name == 'PLUS_DM':
        aroon = PLUS_DM(high, low, period)
        return aroon
    elif ind_name == 'STOCHRSI_K':
        k, _ = STOCHRSI(close, timeperiod=period, fastk_period=fast, fastd_period=slow)
        return k
    elif ind_name == 'STOCHRSI_D':
        _, d = STOCHRSI(close, timeperiod=period, fastk_period=fast, fastd_period=slow)
        return d
    elif ind_name == 'WILLR':
        willr = WILLR(high, low, close, timeperiod=period)
        return willr
    elif ind_name == 'MIN':
        aroon = MIN(low, period)
        return aroon
    elif ind_name == 'MAX':
        aroon = MAX(high, period)
        return aroon
    elif ind_name == 'HIGH_KURT':
        aroon = high.rolling(period).kurt()
        return aroon
    elif ind_name == 'LOW_KURT':
        aroon = low.rolling(period).kurt()
        return aroon
    elif ind_name == 'HIGH_SKEW':
        aroon = high.rolling(period).skew()
        return aroon
    elif ind_name == 'LOW_SKEW':
        aroon = low.rolling(period).skew()
        return aroon
    elif ind_name == 'KURT':
        aroon = close.rolling(period).kurt()
        return aroon
    elif ind_name == 'SKEW':
        aroon = close.rolling(period).skew()
        return aroon
    elif ind_name == 'MININDEX':
        aroon = np.arange(len(bars.index)) - MININDEX(bars['low'], period)
        return aroon
    elif ind_name == 'MAXINDEX':
        aroon = np.arange(len(bars.index)) - MAXINDEX(bars['high'], period)
        return aroon   
    else:
        raise ValueError("wrong indicator name")

def make_indicator(bars, ind_name, f, s, p):
    # return scale(make_ind(bars, ind_name, f, s, p))
    return make_ind(bars, ind_name, f, s, p)



class trading_fitness(base_ff):
    maximise = True


    def __init__(self):
        super().__init__()

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

        aapl_dollar_bars_path = Path().resolve().parent/'datasets'/'bars'/'aapl_dollar_bars.h5'
        train_bars = pd.read_hdf(aapl_dollar_bars_path, key='key')
        
        if train_bars.index.name != 'date_time':
            train_bars = train_bars.set_index('date_time')
        train_bars = train_bars[['open', 'high', 'low', 'close', 'volume']]



        self.fitness_functions = calculate_sharpe_ratio

        self.bars = train_bars[train_bars.index < last_train_date]
        self.ind_ls = ind_ls
        self.cross = cross
        self.cross_num = cross_num
        self.widen = widen
        self.shrink = shrink
        self.count_negative = count_negative
        self.count_positive = count_positive
        self.make_indicator = make_indicator

        self.default_fitness = 0.


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
                        return np.nan
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
                        return np.nan
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
        return portfolio


    def _backtest_years(self, ind, years):
        sharpe_ratios_ls = []
        
        # print(self.bars)
        for year in years:
            new_idx = self.bars.index[self.bars.index >= pd.to_datetime(f'{year}-01-01')]
            new_idx = new_idx[new_idx < pd.to_datetime(f'{year+1}-01-01')]

            p, d = ind.phenotype, {'bars': self.bars.loc[new_idx, :], 'ind_ls': self.ind_ls,
            'cross': self.cross, 'cross_num': self.cross_num, 'widen':self.widen,
            'shrink':self.shrink, 'count_positive': self.count_positive, 'count_negative': self.count_negative,
            'make_indicator': self.make_indicator, 'pd': pd, 'np': np}

            exec(p, d)
            trades = d['_results_']
            exit_bar= d['_exit_bar_']
            portfolio = self._backtest(trades, exit_bar, self.bars.loc[new_idx, :])
            sharpe_ratio = calculate_sharpe_ratio(portfolio)
            # if sharpe is np.nan:
            #     sharpe = 0.
            sharpe_ratios_ls.append(sharpe_ratio)
        

        return np.min(sharpe_ratios_ls)

    def evaluate(self, ind, **kwargs):
        years = [2009, 2011, 2013]
        min_sharpe_ratio = self._backtest_years(ind, years)
        return min_sharpe_ratio

