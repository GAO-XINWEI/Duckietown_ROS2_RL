import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np
from rl_control.Parameters import *


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class ACNet(nn.Module):
    def __init__(self, action_dim, device):
        super(ACNet, self).__init__()
        self.device = device
        # conv
        self.conv = nn.Sequential(
            # nn.MaxPool2d(2),
            nn.Conv2d(CHANNAL, NET_SIZE // 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(NET_SIZE // 8, NET_SIZE // 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(NET_SIZE // 8, NET_SIZE // 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(NET_SIZE // 8, NET_SIZE // 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(NET_SIZE // 4, NET_SIZE // 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(NET_SIZE // 4, NET_SIZE // 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(NET_SIZE // 4, NET_SIZE // 2, kernel_size=2, stride=1))

        # shared full
        self.shared_full = nn.Sequential(
            nn.Dropout(DROP_PROB),
            nn.Linear(6912, NET_SIZE),
            nn.Sigmoid())

        # LSTM layer
        # self.lstm = nn.LSTM(NET_SIZE, NET_SIZE, num_layers=1, batch_first=True)

        # actor
        self.actor_full = nn.Sequential(
            nn.Linear(NET_SIZE, NET_SIZE))
        self.actor_mu = nn.Sequential(
            nn.Linear(NET_SIZE, NET_SIZE),
            nn.Linear(NET_SIZE, action_dim))
        self.actor_sigma = nn.Sequential(
            nn.Linear(NET_SIZE, NET_SIZE),
            nn.Linear(NET_SIZE, action_dim))

        # critic
        self.critic_full = nn.Sequential(
            nn.Linear(NET_SIZE, NET_SIZE),
            nn.Linear(NET_SIZE, NET_SIZE),
            nn.Linear(NET_SIZE, 1))

        self.apply(weights_init)
        # self.optimizer = optim.AdamW(self.parameters(), lr=LEARNING_RATE)

    def forward(self, state, lstm_state=None, old_action=None):
        # share full
        conv = self.conv(state)
        shared_full = self.shared_full(conv.view(conv.size(0), -1))
        # shared_lstm, lstm_state = self.lstm(shared_full, lstm_state)
        shared_lstm = shared_full
        # actor
        action_mean = ACTOR_MEAN_FACTOR * torch.tanh(self.actor_mu(self.actor_full(shared_lstm))/ACTOR_MEAN_FACTOR)
        cov_mat = torch.diag_embed(torch.clamp(
            ACTOR_SIGMA_FACTOR * torch.sigmoid(self.actor_sigma(self.actor_full(shared_lstm))),     # todo
            VARIANCE_BOUNDARY))
        # critic
        state_value = CITIC_NET_FACTOR * self.critic_full(shared_lstm)
        """'''action dim'''
        # print('actor_full', actor_full)
        # print('action_mean', action_mean)  # action_mean tensor([[-0.0975]], device='cuda:0', grad_fn=<TanhBackward>)
        # print(self.actor_sigma(self.actor_full(self.shared_full(state))))
        # tensor([[0.7744]], device='cuda:0', grad_fn=<SoftplusBackward>)
        # print(torch.diag(self.actor_sigma(self.actor_full(self.shared_full(state)))))
        # If input is a vector (1-D tensor),
        # then returns a 2-D square tensor with the elements of input as the diagonal.
        # If input is a matrix (2-D tensor),
        # then returns a 1-D tensor with the diagonal elements of input.
        # cov_mat = torch.diag(self.actor_sigma(self.actor_full(self.shared_full(state))))
        # print(torch.diag_embed(self.actor_sigma(self.actor_full(self.shared_full(state)))))
        '''parameter passing check(preposition print)'''
        # conv_full = self.conv(state)
        # print('conv_full', conv_full)
        # print('conv_full', conv_full.view(conv_full.size(0), -1))
        ''''''
        # print('shared_full', shared_full)
        # print('actor_full', actor_full)
        # print('action_mean', action_mean)
        # print('cov_mat', cov_mat)"""

        try:
            dist = MultivariateNormal(action_mean, cov_mat)
        except:
            # error when gradiant blow out
            print('state', state, 'state_value', state_value, 'action_mean', action_mean, 'cov_mat', cov_mat)
            print(dist)
            raise ValueError
        if old_action is None:
            action = torch.clamp(dist.sample(), -1, 1)
            # print(action)
            if BOOL_TRAINING:
                if np.random.uniform(0, 1) < EGREEDY:
                    action = torch.tensor(np.expand_dims(np.random.uniform(-1, 1, 2), axis=0), dtype=torch.float32).to(self.device)
            # print(action)
        else:
            action = old_action
        action_logprob = dist.log_prob(action)
        dist_entropy = dist.entropy()
        # if old_action is not None:
        #     print('old_action', old_action)
        #     print('dist', dist)
        #     print('dist.log_prob(old_action)', dist.log_prob(action))
        #     print('dist.prob(old_action)', torch.exp(dist.log_prob(action)))
        # else:
        #     print('action', action)
        #     print('dist', dist)
        #     print('dist.log_prob(action)', dist.log_prob(action))

        return action.detach(), torch.squeeze(state_value), lstm_state, action_logprob, dist_entropy


if __name__ == '__main__':
    import time
    device = torch.device("cuda:0")
    torch.set_default_tensor_type(torch.DoubleTensor)
    ac = [ACNet(2, device).to(device) for _ in range(1)]
    rnn_state = None
    state0 = None
    print(ac[0])
    test_tensor = 255 * np.random.random(size=(1, 3, 30, 40))
    print('test_tensor', test_tensor)
    print(ac[0](torch.tensor(test_tensor).to(device)))
    time.sleep(60)
    
