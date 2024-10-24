import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
import rl_utils
import tool_functions


# class TwoLayerFC(torch.nn.Module):
#     def __init__(self, num_in, num_out, hidden_dim):
#         super().__init__()
#         self.fc1 = torch.nn.Linear(num_in, hidden_dim)
#         self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = torch.nn.Linear(hidden_dim, num_out)
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = torch.tanh(self.fc2(x))
#         return self.fc3(x)
# three layer network
class TwoLayerFC(torch.nn.Module):
    def __init__(self, num_in, num_out, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(num_in, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, num_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DDPG:
    ''' DDPG '''

    def __init__(self, state_dim, action_dim, critic_input_dim, hidden_dim,
                 actor_lr, critic_lr, device, flag, agt_i, real_num, env_id):
        # train
        if flag:
            self.actor = TwoLayerFC(state_dim, action_dim, hidden_dim).to(device)
            self.target_actor = TwoLayerFC(state_dim, action_dim, hidden_dim).to(device)

            self.critic = TwoLayerFC(critic_input_dim, 1, hidden_dim).to(device)
            self.target_critic = TwoLayerFC(critic_input_dim, 1, hidden_dim).to(device)

            self.target_critic.load_state_dict(self.critic.state_dict())
            self.target_actor.load_state_dict(self.actor.state_dict())

            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        # test
        else:
            if real_num == 0:
                self.actor = TwoLayerFC(state_dim, action_dim, hidden_dim).to(device)
                self.critic = TwoLayerFC(critic_input_dim, 1, hidden_dim).to(device)
                self.actor.load_state_dict(torch.load('./model/actor%d.pth' % agt_i))
            else:
                # Define different migration relationships based on different environments
                if env_id == "simple_spread":
                    self.simple_spread(real_num, state_dim, action_dim, hidden_dim, device, agt_i)

    # action
    def take_action(self, state, explore=False):
        action = self.actor(state)
        # sample
        if explore:
            action = tool_functions.gumbel_softmax(action)
        else:
            action = tool_functions.onehot_from_logits(action)
        return action.detach().cpu().numpy()[0]

    # 更新目标网络参数
    def soft_update(self, net, target_net, tau):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

        # self.actor.fc1.weight = torch.nn.Parameter(self.actor.fc1.weight.data * 3/(real_num+3))
