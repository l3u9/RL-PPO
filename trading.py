import pandas as pd
import numpy as np
import random
import copy
from collections import deque
from utils import TradingGraph, Write_to_file
from model import *

# from tensorboardX import SummaryWriter
from torch.optim import Adam, RMSprop
from model import Actor_Model, Critic_Model

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
        
        self.state_size = self.lookback_window_size * 10

        self.lr = 0.0001
        self.epochs = 1
        self.normalize_value = 100000
        self.optimizer = RMSprop

        self.Actor = Actor_Model(input_shape=self.state_size, action_space = self.action_space.shape[0], lr=self.lr, optimizer = self.optimizer)
        self.Critic = Critic_Model(input_shape=self.state_size, action_space = self.action_space.shape[0], lr=self.lr, optimizer = self.optimizer)

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
        self.episode_orders = 0 # test
        self.env_steps_size = env_steps_size


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


    def get_gaes(self, rewards, dones, values, next_values, gamma = 0.99, lamda = 0.95, normalize=True):
        deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)

    def replay(self, states, actions, rewards, predictions, dones, next_states):
        # reshape memory to appropriate shape for training
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)

        # Compute discounted rewards
        #discounted_r = np.vstack(self.discount_rewards(rewards))

        # Get Critic network predictions 
        values = self.Critic.forward(states)
        next_values = self.Critic.forward(next_states)
        # Compute advantages
        #advantages = discounted_r - values
        # print("values: ", type(values))
        # print("next_value: ", type(next_values))
        # advantages, target = self.get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))
        advantages, target = self.get_gaes(rewards, dones, values.detach().numpy(), next_values.detach().numpy())
        '''
        pylab.plot(target,'-')
        pylab.plot(advantages,'.')
        ax=pylab.gca()
        ax.grid(True)
        pylab.show()
        '''
        # stack everything to numpy array
        y_true = np.hstack([advantages, predictions, actions])
        
        # training Actor and Critic networks
        a_loss = self.Actor.fit(states, y_true, epochs=self.epochs )
        c_loss = self.Critic.fit(states, target, epochs=self.epochs )

        # self.writer.add_scalar('Data/actor_loss_per_replay', np.sum(a_loss.history['loss']), self.replay_count)
        # self.writer.add_scalar('Data/critic_loss_per_replay', np.sum(c_loss.history['loss']), self.replay_count)
        # self.replay_count += 1


    def act(self, state):
        # state = torch.tensor(state.flatten(), dtype=torch.float32).unsqueeze(0)

        # Use the network to predict the next action to take, using the model
        # prediction = self.Actor.forward(np.expand_dims(state, axis=0))[0]
        prediction = self.Actor.forward(state)
        # print("Prediction:", prediction)
        # print("Prediction.item():", prediction.item())
        # print("Type of prediction.item():", type(prediction.item()))
        prediction = prediction.detach().numpy().flatten()
        # print("Numpy prediction:", prediction)
        action = np.random.choice(self.action_space, p=prediction)
        return action, prediction

    def save(self, name="Crypto_trader"):
        # save keras model weights
        self.Actor.save_weights(f"{name}_Actor.pth")
        self.Critic.save_weights(f"{name}_Critic.pth")

    def load(self, name="Crypto_trader"):
        # load keras model weights
        self.Actor.load_weights(f"{name}_Actor.pth")
        self.Critic.load_weights(f"{name}_Critic.pth")

        
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

def train_agent(env, visualize=False, train_episodes=50, training_batch_size=500):
    # env.create_writer()
    total_average = deque(maxlen=100)
    best_average = 0

    for episode in range(train_episodes):
        state = env.reset(env_steps_size=training_batch_size)

        states, actions, rewards, predictions, dones, next_states = [], [], [], [], [], []
        for t in range(training_batch_size):
            env.render(visualize)
            # print(state.shape)
            action, prediction = env.act(state)
            next_state, reward, done = env.step(action)
            states.append(np.expand_dims(state, axis=0))
            next_states.append(np.expand_dims(next_state, axis=0))
            action_onehot = np.zeros(3)
            action_onehot[action] = 1
            actions.append(action_onehot)
            rewards.append(reward)
            dones.append(done)
            predictions.append(prediction)
            state = next_state

        env.replay(states, actions, rewards, predictions, dones, next_states)
        total_average.append(env.net_worth)
        average = np.average(total_average)

        # env.writer.add_scalar('Data/average net_worth', average, episode)
        # env.writer.add_scalar('Data/episode_orders', env.episode_orders, episode)

        print("net worth {} {:.2f} {:.2f} {}".format(episode, env.net_worth, average, env.episode_orders))
        if episode > len(total_average):
            if best_average < average:
                best_average = average
                print("Saving model")
                env.save()

def test_agent(env, visualize=True, test_episodes=10):
    env.load()
    average_net_worth = 0
    for episode in range(test_episodes):
        state = env.reset()
        while True:
            env.render(visualize)
            action, prediction = env.act(state)
            state, reward, done = env.step(action)
            if env.current_step == env.end_step:
                average_net_worth += env.net_worth
                print("net_worth:", episode, env.net_worth, env.episode_orders)
                break
            
    print("average {} episodes agent net_worth: {}".format(test_episodes, average_net_worth/test_episodes))


    


df = pd.read_csv('./pricedata.csv')
df = df.sort_values('Date')
lookback_window_size = 50
train_df = df[:-720-lookback_window_size]
test_df = df[-720-lookback_window_size:] # 30 days


train_env = CustomEnv(train_df,lookback_window_size=lookback_window_size)
test_env = CustomEnv(test_df,lookback_window_size=lookback_window_size)

# train_agent(train_env, visualize=False, train_episodes=20000, training_batch_size=500)


test_agent(test_env, visualize=False, test_episodes=20)

Random_games(test_env, visualize=False, train_episodes = 20)