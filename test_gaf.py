from pyts.image.gaf import GramianAngularField
import pandas as pd
import numpy as np
from get_data_stock_trading_mbfa import data_main
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from pyts.image import GramianAngularField

df, _, df_daily = data_main(0, 10000)

df_daily = df_daily.set_index('date')
print(df_daily.loc[:'2007-07-04'].iloc[-60:])

df = pd.merge(df, df_daily, 'left', left_on='date', right_on='date', suffixes=('','_d'))

df['vol_pct'] = abs(df['close'].pct_change())


print(df)
WINDOW = 50

def rescale_and_gaf(i, df, column):
    min_ = min(df['fib_0.5'].iloc[i+WINDOW] - df['adr'].iloc[i+WINDOW], min(df.low.values[i:i+WINDOW]))
    max_ = max(df['fib_0.5'].iloc[i+WINDOW] + df['adr'].iloc[i+WINDOW], max(df.high.values[i:i+WINDOW]))
    
    scaled_serie = np.array((2*df[column].iloc[i:i+WINDOW] - max_ - min_) / (max_ - min_)).reshape(1, -1)

    gaf = GramianAngularField(50, None, 'd')
    
    img = gaf.fit_transform(scaled_serie)

    return img

def no_scale_gaf(i, df, column):
    
    serie = np.array(df[column].iloc[i:i+WINDOW]*100).reshape(1,-1)
    gaf = GramianAngularField(50, None, 'd')
    
    img = gaf.fit_transform(serie)

    return img




#img = rescale_and_gaf(3000, df)
#im, im1 = map(rescale_and_gaf, (100, 100, 100), (df, df, df))

im, im1, im2, im3 = map(rescale_and_gaf, (1000, 1000, 1000, 1000), (df, df, df, df), ('open', 'high', 'low', 'close'))


vol = no_scale_gaf(1000, df, 'vol_pct')
image = np.stack((im, im1, im2, im3, vol), axis=3)
#print(vol)

#plt.imshow(image, cmap='hot', interpolation='nearest')
#plt.show()

print(im.shape, im1.shape, image.shape, image.squeeze().shape)

    

