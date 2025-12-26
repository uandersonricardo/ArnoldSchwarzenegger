import copy

import torch
import numpy as np

from src.rainbow.network import DuelingNet, Net


class DQN(object):
    def __init__(self, args):
        self.action_dim = args.action_dim
        self.batch_size = args.batch_size  # batch size
        self.max_train_steps = args.max_train_steps
        self.lr = args.lr  # learning rate
        self.gamma = args.gamma  # discount factor
        self.tau = args.tau  # Soft update
        self.use_soft_update = args.use_soft_update
        self.target_update_freq = args.target_update_freq  # hard update
        self.update_count = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        self.grad_clip = args.grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_double = args.use_double
        self.use_dueling = args.use_dueling
        self.use_per = args.use_per
        self.use_n_steps = args.use_n_steps
        if self.use_n_steps:
            self.gamma = self.gamma ** args.n_steps

        if self.use_dueling:  # Whether to use the 'dueling network'
            self.net = DuelingNet(args)
        else:
            self.net = Net(args)

        self.target_net = copy.deepcopy(self.net)  # Copy the online_net to the target_net

        # move networks to device before creating optimizer
        self.net.to(self.device)
        self.target_net.to(self.device)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def choose_action(self, state, epsilon):
        with torch.no_grad():
            state = torch.unsqueeze(torch.tensor(state, dtype=torch.float, device=self.device), 0)
            q = self.net(state)
            if np.random.uniform() > epsilon:
                action = q.argmax(dim=-1).item()
            else:
                action = np.random.randint(0, self.action_dim)
            return action

    def learn(self, replay_buffer, total_steps):
        batch, batch_index, IS_weight = replay_buffer.sample(total_steps)

        # convert batch items to tensors on the correct device
        def to_device(x, dtype=torch.float):
            if isinstance(x, torch.Tensor):
                return x.to(self.device)
            return torch.tensor(x, dtype=dtype, device=self.device)

        state = to_device(batch['state'], dtype=torch.float)
        next_state = to_device(batch['next_state'], dtype=torch.float)
        reward = to_device(batch['reward'], dtype=torch.float)
        terminal = to_device(batch['terminal'], dtype=torch.float)
        action = to_device(batch['action'], dtype=torch.long)

        # ensure action has a trailing dim for gather if needed
        if action.dim() == 1:
            action = action.unsqueeze(-1)

        if self.use_per:
            IS_weight = to_device(IS_weight, dtype=torch.float)

        with torch.no_grad():  # q_target has no gradient
            if self.use_double:  # Whether to use the 'double q-learning'
                # Use online_net to select the action
                a_argmax = self.net(next_state).argmax(dim=-1, keepdim=True)  # shape：(batch_size,1)
                # Use target_net to estimate the q_target
                q_target = reward + self.gamma * (1 - terminal) * self.target_net(next_state).gather(-1, a_argmax).squeeze(-1)  # shape：(batch_size,)
            else:
                q_target = reward + self.gamma * (1 - terminal) * self.target_net(next_state).max(dim=-1)[0]  # shape：(batch_size,)

        q_current = self.net(state).gather(-1, action).squeeze(-1)  # shape：(batch_size,)
        td_errors = q_current - q_target  # shape：(batch_size,)

        if self.use_per:
            loss = (IS_weight * (td_errors ** 2)).mean()
            replay_buffer.update_batch_priorities(batch_index, td_errors.detach().cpu().numpy())
        else:
            loss = (td_errors ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.optimizer.step()

        if self.use_soft_update:  # soft update
            for param, target_param in zip(self.net.parameters(), self.target_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        else:  # hard update
            self.update_count += 1
            if self.update_count % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.net.state_dict())

        if self.use_lr_decay:  # learning rate Decay
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        lr_now = 0.9 * self.lr * (1 - total_steps / self.max_train_steps) + 0.1 * self.lr
        for p in self.optimizer.param_groups:
            p['lr'] = lr_now

    def save_model(self, path):
        torch.save(self.net.state_dict(), path)
    
    def load_model(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.net.state_dict())
