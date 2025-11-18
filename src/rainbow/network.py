import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=args.state_dim[2], out_channels=32, kernel_size=(8, 8), stride=(4, 4))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2))
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(p=0.5)

        convw = self._conv2d_size_out(self._conv2d_size_out(args.state_dim[0], 8, 4), 4, 2)
        convh = self._conv2d_size_out(self._conv2d_size_out(args.state_dim[1], 8, 4), 4, 2)
        linear_input_size = convw * convh * 64

        if args.use_noisy:
            self.fc1 = NoisyLinear(linear_input_size, args.action_dim)
        else:
            self.fc1 = nn.Linear(linear_input_size, args.action_dim)

    def forward(self, s):
        s = s / 255.0  # normalize pixel values
        s = s.permute(0, 3, 1, 2)  # change from NHWC to NCHW
        s = torch.relu(self.conv1(s))
        s = self.bn1(s)
        s = torch.relu(self.conv2(s))
        s = self.bn2(s)
        s = self.dropout(s)
        s = s.flatten(start_dim=1)
        Q = self.fc1(s)
        return Q

    def _conv2d_size_out(self, size, kernel_size, stride):
        return (size - (kernel_size - 1) - 1) // stride + 1


class DuelingNet(nn.Module):
    def __init__(self, args):
        super(DuelingNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=args.state_dim[2], out_channels=32, kernel_size=(8, 8), stride=(4, 4))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2))
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(p=0.5)
        # self.fc1 = nn.Linear(args.state_dim[0] * args.state_dim[1] * args.state_dim[2], args.hidden_dim)
        # self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)

        convw = self._conv2d_size_out(self._conv2d_size_out(args.state_dim[0], 8, 4), 4, 2)
        convh = self._conv2d_size_out(self._conv2d_size_out(args.state_dim[1], 8, 4), 4, 2)
        linear_input_size = convw * convh * 64

        if args.use_noisy:
            self.V = NoisyLinear(linear_input_size, 1)
            self.A = NoisyLinear(linear_input_size, args.action_dim)
        else:
            self.V = nn.Linear(linear_input_size, 1)
            self.A = nn.Linear(linear_input_size, args.action_dim)

    def forward(self, s):
        s = s / 255.0  # normalize pixel values
        s = s.permute(0, 3, 1, 2)  # change from NHWC to NCHW
        s = torch.relu(self.conv1(s))
        s = self.bn1(s)
        s = torch.relu(self.conv2(s))
        s = self.bn2(s)
        s = self.dropout(s)
        s = s.flatten(start_dim=1)
        V = self.V(s)  # batch_size X 1
        A = self.A(s)  # batch_size X action_dim
        Q = V + (A - torch.mean(A, dim=-1, keepdim=True))  # Q(s,a)=V(s)+A(s,a)-mean(A(s,a))
        return Q

    def _conv2d_size_out(self, size, kernel_size, stride):
        return (size - (kernel_size - 1) - 1) // stride + 1


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            self.reset_noise()
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)  # mul是对应元素相乘
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)

        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))  # 这里要除以out_features

    def reset_noise(self):
        epsilon_i = self.scale_noise(self.in_features)
        epsilon_j = self.scale_noise(self.out_features)
        self.weight_epsilon.copy_(torch.ger(epsilon_j, epsilon_i))
        self.bias_epsilon.copy_(epsilon_j)

    def scale_noise(self, size):
        x = torch.randn(size)  # torch.randn产生标准高斯分布
        x = x.sign().mul(x.abs().sqrt())
        return x
