#%%
from cProfile import run
from email.mime import image
from get_data_stock_trading_mbfa import *
from helpers_mbfa import *
import pandas
import numpy as np

from multiprocessing import Process

from matplotlib.lines import Line2D
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib as mpl
import PIL
from joblib import Parallel, delayed

#mpl.rcParams['savefig.pad_inches'] = 0

toml = get_toml_data('config.toml')
image_path = toml['file']['image']

WINDOW = toml['config_data']['window']
df, _, _ = data_main(0, 80000)

df['minute'] = df['index'].dt.minute
#print(df.head(30), df.tail(30), df.columns, df.info())


start = 1000
loops_per_worker = 17000

def download_image_as_array(df1, i):
  """downloads an annotated graph image mimicking what a trader would look at"""

  for i in range(i, i+loops_per_worker):
      
        df = df1.copy()
        df = df.iloc[0:i]


        df = df.iloc[-df.hour.values[-1]*4 - 7*4  - int(df.minute.values[-1]/15) :].reset_index()

        fig = Figure(figsize=(6, 6))
        fig.tight_layout(pad=0)
        ax = fig.gca()


        # Plotting Candles 

        ax.set_ylim(df.low.min(),  df.high.max())
        ax.set_xlim(right=max(60, len(df)))
        date_index = np.array(df.index)

        bars = np.array(df.close)-np.array(df.open)
        wicks = np.array(df.high)-np.array(df.low)
        ax.bar(date_index, bars, width=0.6, bottom=df.open, color='blue')#, color=color_index)
        ax.bar(date_index, wicks, width=0.2, bottom=df.low, color='blue')#, color=color_index)



        ax.axis('off')
        fig.set_size_inches((10,10))

        fig.savefig(f"/Users/thomasrigou/stock_trading_rl/cnn_3d/image/test_ob{i}.png",dpi=55, bbox_inches='tight')
        print(i)

  return 
  

if __name__ == '__main__':
    Parallel(n_jobs=3, max_nbytes=None)(delayed(download_image_as_array)(df, i) for df, i in zip([df, df, df], [start, start+loops_per_worker, start+2*loops_per_worker]))




print('done')
