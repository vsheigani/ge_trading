import numpy as np
import pandas as pd

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


def calculate_sharpe_ratio(series: pd.Series, benchmark_rate: float=0) -> float:
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


