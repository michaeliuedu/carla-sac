import parl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# clamp bounds for Std of action_log
LOG_SIG_MAX = 20.0
LOG_SIG_MIN = -20.0

__all__ = ['TorchModel']


class TorchModel(parl.Model):
    def __init__(self, obs_dim, action_dim):
        super(TorchModel, self).__init__()
        self.actor_model = Actor(obs_dim, action_dim)
        self.critic_model = Critic(obs_dim, action_dim)

    def policy(self, obs):
        return self.actor_model(obs)

    def value(self, obs, action):
        return self.critic_model(obs, action)


class Actor(parl.Model):
    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()

        hidden_sizes = (1024, 512, 256, 128)
        initial_mean_factor = 0.1
        initial_std = 0.4

        IMITATION = True
        print('loading offline model: ', IMITATION)

        #Task 3 - Offline CNN Implementation
        dict_state = torch.load("../imitation/model_imitation_latent_1action21_steer_throttle_optim.pt")
        self.offline = {}
        for k, v in dict_state.items():
            if k.startswith('imitation'):
                print(k)
                self.offline[k[13:]] = v
       
        self.actor0 = nn.Sequential(
                nn.Linear(*obs_dim, 1024), nn.ReLU(),
                nn.Linear(1024, 512), nn.ReLU(),
                nn.Linear(512, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, action_dim),nn.Tanh()
                # nn.Softmax(dim=-1)
        )

        self.actor1 = nn.Sequential(
                nn.Linear(*obs_dim, 1024), nn.ReLU(),
                nn.Linear(1024, 512), nn.ReLU(),
                nn.Linear(512, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, action_dim),nn.Tanh()
                # nn.Softmax(dim=-1)
        )
        
        self.action_logstd = nn.Parameter(torch.full((action_dim,), np.log(initial_std), dtype=torch.float32), requires_grad=True)

        print(self.actor0)

        if IMITATION:
            self.actor0.load_state_dict(self.offline)
            self.actor1.load_state_dict(self.offline)


    def forward(self, obs):
        state = obs
        act_mean = self.actor0(state)
        act_std = self.actor1(state)

        act_log_std = torch.clamp(act_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        # act_log_std = act_std
        return act_mean, act_log_std


class Critic(parl.Model):
    def __init__(self, obs_dim, action_dim):
        super(Critic, self).__init__()

        fc1_dims = 32
        fc2_dims = 16

        # Q1 network
        self.critic1 = nn.Sequential(
                nn.Linear(*obs_dim + action_dim, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1),
                nn.Sigmoid()  ##ali
        )

        # Q2 network
        self.critic2 = nn.Sequential(
                nn.Linear(*obs_dim + action_dim, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1),
                nn.Sigmoid()  ##ali
        )

        print(self.critic1)
        

    def forward(self, obs, action):
        x = torch.cat([obs, action], 1)
        state = x
        q1 = self.critic1(state)
        q2 = self.critic2(state)
        return q1, q2



__all__ = ['TorchAgent']


class TorchAgent(parl.Agent):
    def __init__(self, algorithm):
        super(TorchAgent, self).__init__(algorithm)

        self.device = torch.device("cuda" if torch.cuda.
                                   is_available() else "cpu")

        self.alg.sync_target(decay=0)

    def predict(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)  # Remove the reshape operation
        action = self.alg.predict(obs)
        action_numpy = action.cpu().detach().numpy().flatten()
        return action_numpy

    def sample(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)  # Remove the reshape operation
        action, _ = self.alg.sample(obs)
        action_numpy = action.cpu().detach().numpy().flatten()
        return action_numpy


    def learn(self, obs, action, reward, next_obs, terminal):
        terminal = np.expand_dims(terminal, -1)
        reward = np.expand_dims(reward, -1)

        obs = torch.FloatTensor(obs).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        terminal = torch.FloatTensor(terminal).to(self.device)
        critic_loss, actor_loss = self.alg.learn(obs, action, reward, next_obs,
                                                 terminal)
        return critic_loss, actor_loss

