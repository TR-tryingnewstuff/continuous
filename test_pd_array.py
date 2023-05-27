#%%
from datetime import timedelta
import io
from xml.etree.ElementPath import xpath_tokenizer

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
#from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image

from get_data_stock_trading_mbfa import *
import matplotlib
matplotlib.use('Agg')

df, df1, daily_df = data_main(0, 70000)

print(df.head())




df4 = df.iloc[-df.hour.values[-1]*4 - 4*4  - int(df.minute.values[-1]/15) - int(df.weekday.values[-1]) * 92:]
df4['hour'] = df4['hour'] // 4

df4 = df4.groupby(['date', 'hour']).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).reset_index()


df = df.iloc[-df.hour.values[-1]*4 - 7*4  - int(df.minute.values[-1]/15) :].reset_index()

fig = Figure(figsize=(6, 6))
fig.tight_layout(pad=0)
ax = fig.gca()


# Plotting Candles 

ax.set_ylim(df.low.min(),  df.high.max())
ax.set_xlim(right=+80)
date_index = np.array(df.index)

bars = np.array(df.close)-np.array(df.open)
wicks = np.array(df.high)-np.array(df.low)
ax.bar(date_index, bars, width=0.6, bottom=df.open, color='blue')#, color=color_index)
ax.bar(date_index, wicks, width=0.2, bottom=df.low, color='blue')#, color=color_index)



ax.axis('off')
fig.set_size_inches((10,10))

fig.savefig(f"/Users/thomasrigou/stock_trading_rl/cnn_3d/test_ob.png",dpi=50, bbox_inches='tight')

image_15 = Image.open(f'test_ob.png').convert('L')
image_15 = np.asarray(image_15)     
plt.imshow(image_15)
plt.show()  

fig.clear()
#_________________________ 4 HOUR
#fig = plt.figure(figsize=(6, 6))
#fig.tight_layout(pad=0)
ax = fig.gca()


# Plotting Candles 

ax.set_ylim(df4.low.min(),  df4.high.max())
ax.set_xlim(right=60)
date_index = np.array(df4.index)

bars = np.array(df4.close)-np.array(df4.open)
wicks = np.array(df4.high)-np.array(df4.low)
ax.bar(date_index, bars, width=0.6, bottom=df4.open, color='blue')#, color=color_index)
ax.bar(date_index, wicks, width=0.2, bottom=df4.low, color='blue')#, color=color_index)


ax.axis('off')
fig.set_size_inches((10,10))

fig.savefig(f"/Users/thomasrigou/stock_trading_rl/cnn_3d/test_ob_4h.png",dpi=50, bbox_inches='tight')

plt.close('all')
