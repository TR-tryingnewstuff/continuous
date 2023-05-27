import pandas as pd 
import numpy as np

WINDOW = 60

SEASONALS = [-1, 0, 1, 1, 0, -1, 1, 0, 0, 1, 1, 1]
DAILY_INDICATORS = ['bull_ob', 'bear_ob', 'bull_bb', 'bear_bb', 'bull_low_liquidity', 'bear_high_liquidity']

data = '/Users/thomasrigou/Downloads/es-15m.csv'


def get_15min_data(start, stop=-1, indicator=False, economic_calendar = True):
    """Returns the S&P 500 historical 15min data + some custom indicators if indicators = True"""
    
    df = pd.read_csv(data, delimiter=';', names=['date', 'time', 'open', 'high', 'low', 'close', 'volume'])
    df = df.iloc[start:stop].reset_index()

    df['index'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%d/%m/%Y %H:%M:%S', exact=True, infer_datetime_format=True)
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', exact=True, infer_datetime_format=True)

    df['hour'] = df['index'].dt.hour
    df['minute'] = df['index'].dt.minute
    df['weekday'] = df['index'].dt.weekday
    
    
    df = df.dropna()
    
   # df = df.drop(['time'] , axis=1)
    

    return df

def get_1h_data(df):
    df_1 = df.groupby(['date', 'hour']).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})
    
    return df_1

def get_daily_data(df):
    """Returns the S&P 500 historical daily data + some custom indicators if indicators = True"""
    df_daily = df.groupby('date').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})
    df_daily['adr'] = abs(df_daily.open - df_daily.close).rolling(5).mean()

    
    df_daily = fib(df_daily)
    return df_daily

def orderblock(df): 
    cond_price_bull = (df['close'] < df['close'].shift(-1)) & (df['close'] < df['close'].shift(-2))
    cond_fvg_bull = df['high'].shift(-1) < df['low'].shift(-3) 
    cond_large_candle = df['close'] * 1.002 < df['close'].shift(-1)
    candle_is_down = df['close'] < df['open']
    
    df['ob_bull'] = (cond_price_bull | cond_fvg_bull | cond_large_candle) & candle_is_down
    
    return df
    

def fib(df):
    """Returns the dataframe with fibonacci measures indicated
    to be used on higher dataframe"""

    
    price_range = df['high'] - df['low']
    df['fib_1'] = df['high']
    df['fib_0.75'] = df['low'] + price_range * 0.75
    df['fib_0.5'] = df['low'] + price_range * 0.5
    df['fib_0.25'] = df['low'] + price_range * 0.25
    df['fib_0'] = df['low']
    
    return df

def data_main(start, stop=-1):
    df = get_15min_data(start, stop)
    df_daily = get_daily_data(df)#.dropna()
    df_1 = get_1h_data(df)#.dropna()
   # df_1[['open', 'high', 'low', 'close']] = df_1[['open', 'high', 'low', 'close']].shift(1)
    df_daily[['open', 'high', 'low', 'close', 'fib_1', 'fib_0.75', 'fib_0.5', 'fib_0.25', 'fib_0']] = df_daily[['open', 'high', 'low', 'close', 'fib_1', 'fib_0.75', 'fib_0.5', 'fib_0.25', 'fib_0']].shift(1)
    df_1[['open', 'high', 'low', 'close']] = df_1[['open', 'high', 'low', 'close']].shift(1)

   # df = df*1 # converts bool to int
    
    return df, df_1, df_daily

def data_multitimeframe(stop=-1):
    df, df1, df_daily = data_main(0, stop)
    
    df = pd.merge(df, df_daily, 'left', left_on='date', right_on='date', suffixes=('','_d'))
    df = pd.merge(df, df1, 'left', left_on=['date', 'hour'], right_on=['date', 'hour'], suffixes=('', '_1'))
    
    return df