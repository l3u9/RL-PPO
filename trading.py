import pandas as pd
import numpy as np
import random
from collections import deque
from utils import TradingGraph, Write_to_file
from model import *

class CustomEnv:
    def __init__(self, df, initial_balance=1000, lookback_window_size=50, Render_range=100):
        self.Render_range = Render_range
        # print(df)
        self.df = df.dropna().reset_index()
        # print(df)
        self.df_total_steps = len(self.df) - 1
        # print(self.df_total_steps)
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size
        
        self.action_space = np.array([0, 1, 2])
        
        self.orders_history = deque(maxlen=self.lookback_window_size)
        
        self.market_history = deque(maxlen=self.lookback_window_size)
        
        self.state_size = (self.lookback_window_size, 10)
        
    #     self.lr = 0.0001
    #     self.epochs = 1
    #     self.normalize_value = 100000
    #     self.optimizer = Adam
        
    #     self.Actor = Actor_Model(input_shape=self.state_size, action_space = self.action_space.shape[0], lr=self.lr, optimizer = self.optimizer)
    #     self.Critic = Critic_Model(input_shape=self.state_size, action_space = self.action_space.shape[0], lr=self.lr, optimizer = self.optimizer)
        
    # def create_writer(self):
    #     self.replay_count = 0
    #     self.writer = SummaryWriter(comment="Crypto_trader")

    def reset(self, env_steps_size = 0):
        self.visualization = TradingGraph(Render_range=self.Render_range)
        self.trades = deque(maxlen=self.Render_range)
        
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.crypto_held = 0
        self.crypto_sold = 0
        self.crypto_bought = 0
        
        if env_steps_size > 0:
            self.start_step = random.randint(self.lookback_window_size, self.df_total_steps - env_steps_size)
            self.end_step = self.start_step + env_steps_size
        else:
            self.start_step = self.lookback_window_size
            self.end_step = self.df_total_steps
        
        self.current_step = self.start_step
        
        for i in reversed(range(self.lookback_window_size)):
            current_step = self.current_step - i
            self.orders_history.append([self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])
            self.market_history.append([self.df.loc[current_step, 'Open'],
                                       self.df.loc[current_step, 'High'],
                                       self.df.loc[current_step, 'Low'],
                                       self.df.loc[current_step, 'Close'],
                                       self.df.loc[current_step, 'Volume']
                                       ])
        state = np.concatenate((self.market_history, self.orders_history), axis=1)
        return state
        
    def step(self, action):
        self.crypto_bought = 0
        self.crypto_sold = 0
        self.current_step += 1
        
        current_price = random.uniform(
            self.df.loc[self.current_step, 'Open'],
            self.df.loc[self.current_step, 'Close']
        )


        Date = self.df.loc[self.current_step, 'Date']
        High = self.df.loc[self.current_step, 'High']
        Low = self.df.loc[self.current_step, 'Low']
        
        
        if action == 0:
            pass
        
        elif action == 1 and self.balance > self.initial_balance / 100:
            self.crypto_bought = self.balance / current_price
            self.balance -= self.crypto_bought * current_price
            self.crypto_held += self.crypto_bought
            self.trades.append({'Date' : Date, 'High' : High, 'Low' : Low, 'total': self.crypto_bought, 'type': "buy"})
        
        elif action == 2 and self.crypto_held > 0:
            self.crypto_sold = self.crypto_held
            self.balance += self.crypto_sold * current_price
            self.crypto_held -= self.crypto_sold
            self.trades.append({'Date' : Date, 'High' : High, 'Low' : Low, 'total': self.crypto_sold, 'type': "sell"})
        
        Write_to_file(Date, self.orders_history[-1])
        
        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.crypto_held * current_price
        
        self.orders_history.append([self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])
        
        reward = self.net_worth - self.prev_net_worth
        
        if self.net_worth <= self.initial_balance/2:
            done = True
        else:
            done = False
        
        obs = self._next_observation()
        
        return obs, reward, done

    def _next_observation(self):
        self.market_history.append([self.df.loc[self.current_step, 'Open'],
                                self.df.loc[self.current_step, 'High'],
                                self.df.loc[self.current_step, 'Low'],
                                self.df.loc[self.current_step, 'Close'],
                                self.df.loc[self.current_step, 'Volume']
                                ])
#         print("=" * 0x20 + "market_history" + "=" * 0x20)
#         print(*self.market_history)
        
        obs = np.concatenate((self.market_history, self.orders_history), axis=1)
#         print(obs.shape)
        return obs
    
    def render(self, visualize=False):
        if visualize:
            Date = self.df.loc[self.current_step, 'Date']
            Open = self.df.loc[self.current_step, 'Open']
            Close = self.df.loc[self.current_step, 'Close']
            High = self.df.loc[self.current_step, 'High']
            Low = self.df.loc[self.current_step, 'Low']
            Volume = self.df.loc[self.current_step, 'Volume']

            # Render the environment to the screen
            self.visualization.render(Date, Open, High, Low, Close, Volume, self.net_worth, self.trades)
        
#         print(f'Step: {self.current_step}, Net Worth: {self.net_worth}')
        
def Random_games(env, visualize, train_episodes = 50, training_batch_size = 500):
    average_net_worth = 0
    for episode in range(train_episodes):
        state = env.reset(env_steps_size=training_batch_size)

        while True:
            env.render(visualize)
            action = np.random.randint(3, size=1)[0]

            state, reward, done = env.step(action)

            if env.current_step == env.end_step:
                average_net_worth += env.net_worth
                print("net_worth:",env.net_worth)
                break

    print("average_net_worth:", average_net_worth/train_episodes)




df = pd.read_csv('./pricedata.csv')
df = df.sort_values('Date')

lookback_window_size = 50
train_df = df[:-720-lookback_window_size]
test_df = df[-720-lookback_window_size:] # 30 days

train_env = CustomEnv(train_df, lookback_window_size=lookback_window_size)
test_env = CustomEnv(test_df, lookback_window_size=lookback_window_size)

Random_games(test_env, visualize=False, train_episodes = 1000)

