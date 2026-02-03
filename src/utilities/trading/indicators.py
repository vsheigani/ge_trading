import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def scale(ind: pd.Series) -> pd.Series:
    scaler = MinMaxScaler(feature_range=(0, 100))
    scaled_ind = scaler.fit_transform(ind.values.reshape(-1, 1))
    return pd.Series(scaled_ind.flatten(), name=ind.name)


def make_ind(bars: pd.DataFrame, ind_name: str, f:int, s:int, p:int)-> pd.Series:
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

def make_indicator(bars: pd.DataFrame, ind_name: str, f:int, s:int, p:int) -> pd.Series:
    return scale(make_ind(bars, ind_name, f, s, p))