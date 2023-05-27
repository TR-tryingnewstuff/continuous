import gymnasium as gym
from gymnasium.spaces import Discrete, Box, Dict
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from get_data_stock_trading_mbfa import *
import matplotlib



WINDOW = 100
start_steps = 1000
stop = 80000
VISUALIZE = False
SEE_PROGRESS = True
TRADING_HOURS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

df, df1, daily_df = data_main(0, stop)
df = pd.merge(df, daily_df, 'left', left_on='date', right_index=True, suffixes=('','_d'))
df = df.drop(['open_d', 'low_d', 'high_d'], axis=1) # need close_d for rescaling 15min GAF

df['minute'] = df['index'].dt.minute
df['vol_pct'] = abs(df['close'].pct_change()) / 10
daily_df['vol_pct'] = abs(daily_df['close'].pct_change()/10)

class Market(gym.Env):
    """A class to represent a trading environment"""
    def __init__(self, env_config):
        
        # Initializing the attributes
        self.list_capital = []
        self.position = np.array([0.0])
        self.commission = 0.75 # per contract 
        self.capital = 1000
       
        self.done = False 
        self.n_step = start_steps

        self.df = df.iloc[self.n_step:self.n_step+WINDOW]
        self.time = np.array([self.df.iloc[-1,8]])   
        self.weekday = np.array([self.df.weekday.values[-1]])     
        self.date = str(df.date.values[-1])[0:10]
        

        self.df1 = df1.loc[:(self.date, self.df['hour'].values[-1])].iloc[-60:]
        self.daily_df = daily_df.loc[:self.date].iloc[-60:]
        
        
        self.observation = {'image': self.get_image(), 'vol': np.array([self.daily_df.adr.values[-1] / self.df['close'].values[-1] * 100])}
       
        self.observation_space = Dict({
            'image': Box(0, 255, shape=self.get_image().shape),  
            'vol': Box(0, 15)})

        
        self.action_space = Dict({
            'enter': Discrete(3, start=-1), 
            'stop': Discrete(3, start=1),
            'profit': Discrete(3, start=1)
            })
        
    
    def reset(self):
        """Resets the environment"""
        
        self.list_capital = []
        self.position = np.array([0.0])
        self.capital = 1000
        self.done = False
        self.n_step = start_steps
        
        
        self.df = df.iloc[self.n_step:self.n_step+WINDOW]
        self.time = np.array([self.df.iloc[-1,8]])  
        
        self.observation = {'image': self.get_image(), 'vol': np.array([self.daily_df.adr.values[-1] / self.df['close'].values[-1] * 100])}
        
        return self.observation
         
        
    def step(self, action):
        """Performs one iteration given the action choosen by the agent \\
        returns the following observation, the reward, whether the episode is done and additional info"""
        
        self.n_step += 1  
        self.df = df.iloc[self.n_step:self.n_step+WINDOW] 
         
        if action['enter'] != 0:
            reward = self.get_reward(action)
            
        else:
            reward = 0
        
        self.capital = self.capital + reward
        
        #reward = self.adjust_reward(reward)
        
        # Filter for taking trades only during certain hours
            
        while not self.df['hour'].values[-1] in TRADING_HOURS:
                self.n_step += 1
                self.df = df.iloc[self.n_step:self.n_step+WINDOW]       
                                  
            
        if SEE_PROGRESS and (self.n_step % 10 == 0):
            self.list_capital.append(round(float(self.capital)))
            
            
        if (self.n_step > len(df) - 300) or (self.capital < 500): 
            self.done = True
            print(self.list_capital)
            
        
        self.observation = {'image': self.get_image(), 'vol': np.array([self.daily_df.adr.values[-1] / self.df['close'].values[-1] * 100])}
        
        info = {}  
        return self.observation, float(reward), self.done, info
       
    
    def get_reward(self, action):
        """Compute reward from the chosen action, it returns the price change multiplied by the size of the position"""
        reward = 0
        
        # you can disable the following line if you want 
        #action = self.rule_of_thumbs_check(action)
        
        curr_price = self.df['close'].values[-1]
        sl = action['stop']/1000 + 0.0005
        tp = (action['stop']/1000 + 0.0005) * action['profit']
        
        # If stop loss and take profit given --------------
        if action['profit'] >= 2:
            if action['enter'] == 1:
                stop_price = curr_price * (1-sl)
                tp_price = curr_price * (1+tp)
            
                while (self.df['low'].values[-1] > stop_price) and (self.df['high'].values[-1] < tp_price):
                    self.n_step += 1
                    self.df = df.iloc[self.n_step:self.n_step+WINDOW]
                
                if self.df['low'].values[-1] < stop_price:
                    reward -= sl * self.capital
                
                if self.df['high'].values[-1] > tp_price:
                    reward += tp *self.capital
                    
            else: 
                stop_price = curr_price * (1+sl)
                tp_price = curr_price * (1-tp)
            
                while (self.df['high'].values[-1] < stop_price) and (self.df['low'].values[-1] > tp_price):
                    self.n_step += 1
                    self.df = df.iloc[self.n_step:self.n_step+WINDOW]
                
                if self.df['high'].values[-1] > stop_price:
                    reward -= sl * self.capital
                
                if self.df['low'].values[-1] < tp_price:
                    reward += tp *self.capital
        
        # time based profit ------------------
        elif action['profit'] == 1:
            if action['enter'] == 1:
                stop_price = curr_price *(1-0.0025)
                
                while self.df['hour'].values[-1] < 16 and self.df['low'].values[-1] > stop_price:
                        self.n_step += 1
                        self.df = df.iloc[self.n_step:self.n_step+WINDOW]
                    
                reward += ((self.df['close'].values[-1] - curr_price) / curr_price) * self.capital 

            elif action['enter'] == -1:
                stop_price = curr_price *(1+0.0025)
                
                while self.df['hour'].values[-1] < 16 and self.df['high'].values[-1] < stop_price:
                        self.n_step += 1
                        self.df = df.iloc[self.n_step:self.n_step+WINDOW]
                    
                reward -= ((self.df['close'].values[-1] - curr_price) / curr_price) * self.capital
        
        num_of_contracts = abs(self.capital / self.df['close'].values[-1]) 
        reward -= self.commission * num_of_contracts
        
            
        return reward
    
    def rule_of_thumbs_check(self, action):
        """Basic rule of thumbs to prevent the bot from taking low probability actions \\
            Returns the action with 'enter' set to 1 if a rule was broken"""
            
                
        return action

    def adjust_reward(self, reward):
        """reward is adjusted to risk based on personal preferences \n
           In our case, we divide the reward by a downside risk \\ 
           measure that is adjusted to risk-aversion preferences
        """
          
        downside_risk = (self.df['close'].values[:] - self.df['open'].values[:]).std()
        adjusted_dev = downside_risk * np.exp(-0.2)

            
        adjusted_reward = reward / adjusted_dev
        
        return adjusted_reward
  
    
    def get_image(self):
        data = self.df.iloc[-self.df.hour.values[-1]*4 - 7*4  - int(self.df.minute.values[-1]/15) :].reset_index()

        fig = plt.figure(figsize=(6, 6))
        fig.tight_layout(pad=0)
        ax = fig.gca()

        # Plotting Candles 

        ax.set_ylim(data.low.min(),  data.high.max())
        ax.set_xlim(right=60)
        date_index = np.array(data.index)

        bars = np.array(data.close)-np.array(data.open)
        wicks = np.array(data.high)-np.array(data.low)
        ax.bar(date_index, bars, width=0.6, bottom=data.open, color='blue')
        ax.bar(date_index, wicks, width=0.2, bottom=data.low, color='blue')

        ax.axis('off')
        fig.set_size_inches((10,10))

        fig.savefig(f"test_ob.png",dpi=45, bbox_inches='tight')

        image = Image.open(f'test_ob.png').convert('L')
        image = np.asarray(image) 
        
        plt.close(fig)
        
        fig = plt.figure(figsize=(6, 6))
        fig.tight_layout(pad=0)
        ax = fig.gca()
        
        df4_start = self.df['date'].values[-1] - pd.Timedelta(days=self.df['weekday'].values[-1])
        df4_index = self.df.loc[df['date'] == df4_start].reset_index().index.values[0]
        df4 = self.df.iloc[df4_index:]
        df4['hour'] = df4['hour'] // 4

        df4 = df4.groupby(['date', 'hour']).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).reset_index()  
        
        ax.set_ylim(df4.low.min(),  df4.high.max())
        ax.set_xlim(right=+60)
        date_index = np.array(df4.index)

        bars = np.array(df4.close)-np.array(df4.open)
        wicks = np.array(df4.high)-np.array(df4.low)
        ax.bar(date_index, bars, width=0.6, bottom=df4.open, color='blue')#, color=color_index)
        ax.bar(date_index, wicks, width=0.2, bottom=df4.low, color='blue')#, color=color_index)
        

        ax.axis('off')
        fig.set_size_inches((10,10))

        fig.savefig(f"test_ob_4h.png",dpi=45, bbox_inches='tight')

        image_4 = Image.open(f'test_ob_4h.png').convert('L')
        image_4 = np.asarray(image_4)     
        
        plt.close(fig)
        
        image = np.stack((image, image_4), axis=2)
                
        return image.squeeze()

class Market_continuous(Market):
    def __init__(self, env_config):
        super().__init__(env_config)
        self.list_capital = []
        self.position = np.array([0.0])
        self.commission = 0.00025 # percentage 
        self.capital = 1000
       
        self.done = False 
        self.n_step = start_steps

        self.df = df.iloc[self.n_step-WINDOW:self.n_step]
        self.time = np.array([self.df.iloc[-1,8]])   
        self.weekday = np.array([self.df.weekday.values[-1]])     
        self.date = str(df.date.values[-1])[0:10]
        

        self.daily_df = daily_df.loc[:self.date].iloc[-60:]
        
        
        self.observation = {'image': self.get_image(), 'position': self.position, 'vol': np.array([self.daily_df.adr.values[-1] / self.df['close'].values[-1] * 100])}
       
        self.observation_space = Dict({
            'image': Box(0, 255, shape=self.get_image().shape),
            'position': Box(-1, 1),  
            'vol': Box(0, 15)})

        
        self.action_space = Dict({
            'enter': Discrete(3, start=-1), 
            'size': Box(0, 1),
            })
    
    
    def reset(self):
        """Resets the environment"""
        
        self.list_capital = []
        self.position = np.array([0.0])
        self.capital = 1000
        self.done = False
        self.n_step = start_steps
        
        
        self.df = df.iloc[self.n_step-WINDOW:self.n_step]
        self.time = np.array([self.df.iloc[-1,8]])
        self.date = str(df.date.values[-1])[0:10]  
        self.daily_df = daily_df.loc[:self.date].iloc[-60:]
        
        self.observation = {'image': self.get_image(), 'position': self.position,'vol': np.array([self.daily_df.adr.values[-1] / self.df['close'].values[-1] * 100])}
        
        return self.observation
    
    def step(self, action):
        print(action)
        """Performs one iteration given the action choosen by the agent \\
        returns the following observation, the reward, whether the episode is done and additional info"""
        
        self.n_step += 1  
        self.df = df.iloc[self.n_step-WINDOW:self.n_step] 
         
        if (action['enter'] != 0) or (self.position != 0):
            reward = self.get_reward(action)
            
        else:
            reward = 0
        
        self.capital = self.capital + reward
        
        #reward = self.adjust_reward(reward)
        
        # Filter for taking trades only during certain hours
            
        while not (self.df['hour'].values[-1] in TRADING_HOURS) and (self.position != 0):
                self.n_step += 1
                self.df = df.iloc[self.n_step-WINDOW:self.n_step]       
                                  
            
        if SEE_PROGRESS and (self.n_step % 2 == 0):
            self.list_capital.append(round(float(self.capital)))
            if self.n_step % 400 == 0: 
                if len(self.list_capital) > 400:
                    self.list_capital = self.list_capital[-400:]
                print(self.list_capital)
                
            
            
        if (self.n_step > len(df) - 300) or (self.capital < 500): 
            self.done = True
            print(self.list_capital)
            
        self.date = str(df.date.values[-1])[0:10]
        self.daily_df = daily_df.loc[:self.date].iloc[-60:]
            
        
        self.observation = {'image': self.get_image(), 'position': self.position, 'vol': np.array([self.daily_df.adr.values[-1] / self.df['close'].values[-1] * 100])}
        
        info = {}  
        return self.observation, float(reward), self.done, info
        
    def get_reward(self, action):
        
        reward = 0
        prev_pos = self.position
        
        if action['enter'] == 1:
            self.position = min(self.position + action['size'], np.array([1.0]))       
        
        elif action['enter'] == -1:
            self.position = max(self.position - action['size'], np.array([-1.0]))

        reward += self.position * ((self.df.close.values[-1] - self.df.close.values[-1]) * self.capital/ self.df.open.values[-1])
        reward -= abs(self.position - prev_pos) * self.capital / self.df.open.values[-1] * self.commission
        
        if self.df.hour.values[-1] in [17, 18, 19, 20, 21, 22, 23]:
            self.n_step += 1
            self.df = df.iloc[self.n_step-WINDOW:self.n_step]
        
                    
        return reward
    
    def get_image(self):
        
       # image = Image.open(f'/Users/thomasrigou/stock_trading_rl/cnn_3d/image/test_ob{self.n_step}.png').convert('L')
        #image = np.asarray(image) 
        data = self.df.iloc[-self.df.hour.values[-1]*4 - 7*4  - int(self.df.minute.values[-1]/15) :].reset_index()

        fig = Figure(figsize=(6, 6))
        fig.tight_layout(pad=0)
        ax = fig.gca()

        # Plotting Candles 

        ax.set_ylim(data.low.min(),  data.high.max())
        ax.set_xlim(left=-1, right=max(60, len(data)))
        date_index = np.array(data.index)

        bars = np.array(data.close)-np.array(data.open)
        wicks = np.array(data.high)-np.array(data.low)
        ax.bar(date_index, bars, width=0.6, bottom=data.open, color='blue')
        ax.bar(date_index, wicks, width=0.2, bottom=data.low, color='blue')

        ax.axis('off')
        fig.set_size_inches((10,10))

        fig.savefig(f"test_ob.png",dpi=45, bbox_inches='tight')

        image = Image.open(f'test_ob.png').convert('L')
        image = np.asarray(image) 
                
        return image.squeeze()


class Market_continuous_15_4h(Market):
    def __init__(self, env_config):
        super().__init__(env_config)
        # Initializing the attributes
        self.list_capital = []
        self.position = np.array([0.0])
        self.commission = 0.00025 # percentage 
        self.capital = 1000
       
        self.done = False 
        self.n_step = start_steps

        self.df = df.iloc[self.n_step:self.n_step+WINDOW]
        self.time = np.array([self.df.iloc[-1,8]])   
        self.weekday = np.array([self.df.weekday.values[-1]])     
        self.date = str(df.date.values[-1])[0:10]
        

        self.df1 = df1.loc[:(self.date, self.df['hour'].values[-1])].iloc[-60:]
        self.daily_df = daily_df.loc[:self.date].iloc[-60:]
        
        
        self.observation = {'image': self.get_image(), 'position': self.position, 'vol': np.array([self.daily_df.adr.values[-1] / self.df['close'].values[-1] * 100])}
       
        self.observation_space = Dict({
            'image': Box(0, 255, shape=self.get_image().shape),
            'position': Box(-1, 1),  
            'vol': Box(0, 15)})

        
        self.action_space = Dict({
            'enter': Discrete(3, start=-1), 
            'size': Box(0, 1),
            })
    
    
    def reset(self):
        """Resets the environment"""
        
        self.list_capital = []
        self.position = np.array([0.0])
        self.capital = 1000
        self.done = False
        self.n_step = start_steps
        
        
        self.df = df.iloc[self.n_step:self.n_step+WINDOW]
        self.time = np.array([self.df.iloc[-1,8]])  
        
        self.observation = {'image': self.get_image(), 'position': self.position,'vol': np.array([self.daily_df.adr.values[-1] / self.df['close'].values[-1] * 100])}
        
        return self.observation
    
    def step(self, action):
        """Performs one iteration given the action choosen by the agent \\
        returns the following observation, the reward, whether the episode is done and additional info"""
        
        self.n_step += 1  
        self.df = df.iloc[self.n_step:self.n_step+WINDOW] 
         
        if action['enter'] != 0:
            reward = self.get_reward(action)
            
        else:
            reward = 0
        
        self.capital = self.capital + reward
        
        #reward = self.adjust_reward(reward)
        
        # Filter for taking trades only during certain hours
            
        while not (self.df['hour'].values[-1] in TRADING_HOURS) and (self.position != 0):
                self.n_step += 1
                self.df = df.iloc[self.n_step:self.n_step+WINDOW]       
                                  
            
        if SEE_PROGRESS and (self.n_step % 2 == 0):
            self.list_capital.append(round(float(self.capital)))
            if self.n_step % 400 == 0: 
                if len(self.list_capital) > 400:
                    self.list_capital = self.list_capital[-400:]
                print(self.list_capital)
                
            
            
        if (self.n_step > len(df) - 300) or (self.capital < 500): 
            self.done = True
            print(self.list_capital)
            
        
        self.observation = {'image': self.get_image(), 'position': self.position, 'vol': np.array([self.daily_df.adr.values[-1] / self.df['close'].values[-1] * 100])}
        
        info = {}  
        return self.observation, float(reward), self.done, info
        
    def get_reward(self, action):
        
        reward = 0
        prev_pos = self.position
        
        if action['enter'] == 1:
            self.position = min(self.position + action['size'], np.array([1.0]))       
        
        elif action['enter'] == -1:
            self.position = max(self.position - action['size'], np.array([-1.0]))

        reward += self.position * ((self.df.close.values[-1] - self.df.close.values[-1]) * self.capital/ self.df.open.values[-1])
        reward -= abs(self.position - prev_pos) * self.capital / self.df.open.values[-1] * self.commission
        
        if self.df.hour.values[-1] in [17, 18, 19, 20, 21, 22, 23]:
            self.n_step += 1
            self.df = df.iloc[self.n_step:self.n_step+WINDOW]
        
                    
        return reward
    
    def get_image(self):
        
        image = cv2.imread(f'/Users/thomasrigou/stock_trading_rl/cnn_3d/image/test_ob{self.n_step}.png', 0)#.convert('L')
        image = np.asarray(image) 
        
        image_4 = cv2.imread(f'/Users/thomasrigou/stock_trading_rl/cnn_3d/image_4h/test_ob{self.n_step}.png', 0)#.convert('L')
        image_4 = np.asarray(image_4)   
        
        
        image = np.stack((image, image_4), axis=2)
                
        return image.squeeze()
