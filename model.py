import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

print(torch.cuda.is_available())

class Actor_Model(nn.Module):
    def __init__(self, input_shape, action_space, lr, optimizer):
        super(Actor_Model, self).__init__()
        self.action_space = action_space
        # print("self.action_space: ", self.action_space)

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_shape, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64, action_space),
            nn.Softmax(dim=-1),
        )
        if optimizer == optim.Adam:
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        elif optimizer == optim.RMSprop:
            self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        else:
            raise NotImplementedError

        self.loss = self.ppo_loss

    
    def ppo_loss(self, y_true, y_pred):
        advantages, prediction_piks, actions = y_true[:, :1], y_true[:, 1:1+self.action_space], y_true[:, 1+self.action_space:]
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 0.001

        prob = actions * y_pred
        old_prob = actions * prediction_piks

        prob = torch.clamp(prob, 1e-10, 1.0)
        old_prob = torch.clamp(old_prob, 1e-10, 1.0)

        ratio = torch.exp(torch.log(prob) - torch.log(old_prob))

        p1 = ratio * advantages
        p2 = torch.clamp(ratio, min=1 - LOSS_CLIPPING, max=1 + LOSS_CLIPPING) * advantages

        actor_loss = -torch.mean(torch.min(p1, p2))

        entropy = -(y_pred * torch.log(y_pred + 1e-10))
        entropy = ENTROPY_LOSS * torch.mean(entropy)

        total_loss = actor_loss - entropy

        return total_loss
    
    def forward(self, x):
        x = torch.tensor(x.flatten(), dtype=torch.float32).unsqueeze(0)
        # x = torch.tensor(x, dtype=torch.float32)
        return self.model(x)
    
    def save_weights(self, name="Crypto_trader_Actor.pth"):
        torch.save(self.state_dict(), name)

    def load_weights(self, name="Crypto_trader_Actor.pth"):
        self.load_state_dict(torch.load(name))
        self.eval()

    def fit(self, states, y_true, epochs, batch_size=50):
        states = torch.tensor(states, dtype=torch.float32)
        y_true = torch.tensor(y_true, dtype=torch.float32)

        dataset = TensorDataset(states, y_true)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()

        for epoch in range(epochs):
            for batch_states, batch_y_true in data_loader:
                self.optimizer.zero_grad()

                y_pred = self.model(batch_states)
                loss = self.ppo_loss(batch_y_true, y_pred)
                loss.backward()
                self.optimizer.step()


class Critic_Model(nn.Module):
    def __init__(self, input_shape, action_space, lr, optimizer):
        super(Critic_Model, self).__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_shape, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64,1),
        )

        if optimizer == optim.Adam:
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        elif optimizer == optim.RMSprop:
            self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        else:
            raise NotImplementedError
        
    def ppo2_loss(self, y_true, y_pred):
        value_loss = torch.mean((y_true - y_pred) ** 2)
        return value_loss

    def forward(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        return self.model(state)
    
    def save_weights(self, name="Crypto_trader_Actor.pth"):
        torch.save(self.state_dict(), name)

    def load_weights(self, name="Crypto_trader_Actor.pth"):
        self.load_state_dict(torch.load(name))
        self.eval()

    def fit(self, states, target, epochs, batch_size=50):
        states = torch.tensor(states, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        dataset = TensorDataset(states, target)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()

        for epoch in range(epochs):
            for batch_states, batch_target in data_loader:
                self.optimizer.zero_grad()

                y_pred = self.model(batch_states)
                loss = self.ppo2_loss(batch_target, y_pred)
                loss.backward()
                self.optimizer.step()