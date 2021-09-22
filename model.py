import datetime, gym, os, time, ray, math, torch, random
import scipy.signal
import numpy as np

class DataBuffer:
    def __init__(self):
        self.obs_buf = []
        self.act_buf = []
        self.rew_buf = []
        self.done = []
    def save_data(self, obs, act, rew, done):
        self.obs_buf.append(obs)
        self.act_buf.append(act)
        self.rew_buf.append(rew)
        self.done.append(done)
    def clear_data(self):
        self.obs_buf = []
        self.act_buf = []
        self.rew_buf = []
        self.done = []
    def get_data(self):
        return self.obs_buf, self.act_buf, self.rew_buf, self.done

class CreateModel(torch.nn.Module):
    def __init__(self, obs_size, act_size, lr, v_layer_size=[64, 64], pi_layer_size=[64, 64]):
        super(CreateModel, self).__init__()
        self.v_layer = torch.nn.Sequential(
            torch.nn.Linear(obs_size, v_layer_size[0]),
            torch.nn.Tanh(),
            torch.nn.Linear(v_layer_size[0], v_layer_size[1]),
            torch.nn.Tanh(),
            torch.nn.Linear(v_layer_size[1], 1)
        )
        self.pi_layer = torch.nn.Sequential(
            torch.nn.Linear(obs_size, pi_layer_size[0]),
            torch.nn.Tanh(),
            torch.nn.Linear(pi_layer_size[0], pi_layer_size[1]),
            torch.nn.Tanh(),
            torch.nn.Linear(pi_layer_size[1], act_size),
            torch.nn.Tanh()
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

class PPO:
    def __init__(self, obs_size, act_size, log_std, lr=3e-5):
        self.model = CreateModel(obs_size=obs_size, act_size=act_size, lr=lr)
        if os.path.isfile('model_data_save.pth'):
            self.model.load_state_dict(torch.load('model_data_save.pth'))
        self.old_model = CreateModel(obs_size=obs_size, act_size=act_size, lr=lr)
        self.log_std = log_std

    def discount_cumsum(self, x, discount):
        out = 0.
        out_list = []
        for a in x[::-1]:
            out = discount * out + a
            out_list.append(out)
        out_list.reverse()
        return out_list

    def log_gaussian(self, x, mu):
        log_std_list = self.log_std * torch.ones(mu.shape, dtype=torch.float)
        return -log_std_list - 1 / 2 * np.log(2 * np.pi) - 1 / 2 * ((x - mu) / np.exp(-self.log_std)) ** 2

    def loss(self, state_list, act_list, reward_list, lam=0.95, gam=0.99, epsilon=0.1, c=10, last_val=0.):
        state_list = torch.tensor(state_list, dtype=torch.float)
        act_list = torch.tensor(act_list, dtype=torch.float)
        vals = self.model.v_layer(state_list)
        reward_list = torch.tensor(reward_list)
        last_val = torch.tensor(last_val).reshape(1,1)
        vals = torch.cat((vals, last_val))
        reward_list = torch.cat((reward_list, last_val))

        target = self.discount_cumsum(reward_list.detach().numpy(), gam)
        target = torch.tensor(target, dtype=float)

        deltas = reward_list[:-1] + gam * vals[1:] - vals[:-1]
        advs = self.discount_cumsum(deltas.detach().numpy(), lam*gam)
        advs = torch.tensor(advs, dtype=torch.float)
        mu = self.model.pi_layer(state_list)
        mu_old = self.old_model.pi_layer(state_list)
        logp = self.log_gaussian(act_list, mu)
        logp_old = self.log_gaussian(act_list, mu_old)
        ratio = torch.exp((logp - logp_old.detach())*500)
        loss = -torch.min(ratio * advs, torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advs) + c * ((target[:-1]-vals[:-1]) ** 2)
        random.shuffle(loss)
        return loss.mean()

    def update(self, state_list, act_list, reward_list, epoch=5):
        for i in range(epoch):
            if len(act_list) < 2:
                break
            loss = self.loss(state_list, act_list, reward_list)
            self.model.optimizer.zero_grad()
            loss.backward()
            self.model.optimizer.step()
        #self.old_model = self.model
        torch.save(self.model.state_dict(), 'model_data.pth')
        self.old_model.load_state_dict(torch.load('model_data.pth'))