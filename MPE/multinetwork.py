import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
import rl_utils
import tool_functions
import network as DP
class MADDPG:
    def __init__(self, env, device, actor_lr, critic_lr, hidden_dim, state_dims, action_dims, critic_input_dim, gamma,
                 tau, real_num, env_id):
        self.agents = []
        for i in range(len(env.agents)):
            self.agents.append(
                DP.DDPG(state_dims[i], action_dims[i], critic_input_dim, hidden_dim, actor_lr, critic_lr, device,False,i, real_num, env_id))
        self.gamma = gamma
        self.tau = tau
        self.critic_criterion = torch.nn.MSELoss()
        self.device = device

    @property
    def policies(self):
        return [agt.actor for agt in self.agents]

    @property
    def target_policies(self):
        return [agt.target_actor for agt in self.agents]

    def take_action(self, env,states, explore):
        states = [
            torch.tensor([states[i]], dtype=torch.float, device=self.device)
            for i in range(len(env.agents))
        ]
        return [
            agent.take_action(state, explore)
            for agent, state in zip(self.agents, states)
        ]

    def update(self, sample, i_agent):
        obs, act, rew, next_obs, done = sample
        cur_agent = self.agents[i_agent]

        cur_agent.critic_optimizer.zero_grad()
        all_target_act = [tool_functions.onehot_from_logits(pi(_next_obs)) for pi, _next_obs in zip(self.target_policies, next_obs)]
        target_critic_input = torch.cat((*next_obs, *all_target_act), dim=1)
        target_critic_value = rew[i_agent].view(-1, 1) + self.gamma * cur_agent.target_critic(target_critic_input) * (
                    1 - done[i_agent].view(-1, 1))

        critic_input = torch.cat((*obs, *act), dim=1)
        critic_value = cur_agent.critic(critic_input)
        critic_loss = self.critic_criterion(critic_value, target_critic_value.detach())
        critic_loss.backward()
        cur_agent.critic_optimizer.step()

        cur_agent.actor_optimizer.zero_grad()
        cur_actor_out = cur_agent.actor(obs[i_agent])
        cur_act_vf_in = tool_functions.gumbel_softmax(cur_actor_out)
        all_actor_acs = []
        for i, (pi, _obs) in enumerate(zip(self.policies, obs)):
            if i == i_agent:
                all_actor_acs.append(cur_act_vf_in)
            else:
                all_actor_acs.append(tool_functions.onehot_from_logits(pi(_obs)))
        vf_in = torch.cat((*obs, *all_actor_acs), dim=1)
        actor_loss = -cur_agent.critic(vf_in).mean()
        actor_loss += (cur_actor_out ** 2).mean() * 1e-3
        actor_loss.backward()
        cur_agent.actor_optimizer.step()

    def update_all_targets(self):
        for agt in self.agents:
            agt.soft_update(agt.actor, agt.target_actor, self.tau)
            agt.soft_update(agt.critic, agt.target_critic, self.tau)

    def save(self,agt_i):
        torch.save(self.agents[agt_i].actor.state_dict(),'./model/actor%s.pth'%(agt_i))
        torch.save(self.agents[agt_i].critic.state_dict(),'./model/critic%s.pth'%(agt_i))
