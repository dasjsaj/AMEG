import torch
import os
from network.base_net import RNN
from network.qmix_net import QMixNet
import torch.nn as nn
import torch.nn.functional as f


class RNN1(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, rnn_hidden_dim, n_actions):
        super(RNN1, self).__init__()
        self.fc1 = nn.Linear(input_shape, rnn_hidden_dim)
        self.rnn = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)
        self.fc2 = nn.Linear(rnn_hidden_dim, n_actions)

    def forward(self, obs, hidden_state):
        x = f.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h


# test1 = RNN1(102, 64, 14)


"""QMIX 9m_vs_8m"""
# test1.load_state_dict(torch.load(r"./model/qmix/8m/rnn_net_params.pkl", map_location='cuda:0'))
# #修改代码
# x1 = test1.fc1.weight[:,:84]
# #因为队友增加的信息
# x2= test1.fc1.weight[:,79:84]
# #剩余不变信息
# x3 = test1.fc1.weight[:,84:]
# #最后额外增加的信息
# x4 = test1.fc1.weight[:,-1:]
# test1.fc1.weight = torch.nn.Parameter(
#                     torch.cat((x1, x2), dim=1))
# test1.fc1.weight = torch.nn.Parameter(
#                     torch.cat((test1.fc1.weight, x3), dim=1))
# test1.fc1.weight = torch.nn.Parameter(
#                     torch.cat((test1.fc1.weight, x4), dim=1))
#
# torch.save(test1.state_dict(), './model//qmix/9m_vs_8m/rnn_net_params.pkl')

# """QMIX 2m_vs_3m"""
# x1 = test1.fc1.weight[:,:24]
# #因为队友增加的信息
# x2= test1.fc1.weight[:,29:41]
# test1.fc1.weight = torch.nn.Parameter(
#                     torch.cat((x1, x2), dim=1))
#
# torch.save(test1.state_dict(), './model//qmix/2m_vs_3m/rnn_net_params.pkl')

"""VDN 9m_vs_8m"""
# test1.load_state_dict(torch.load(r"./model/vdn/8m/rnn_net_params.pkl", map_location='cuda:0'))
# # #修改代码
# x1 = test1.fc1.weight[:,:84]
# #因为队友增加的信息
# x2= test1.fc1.weight[:,79:84]
# #剩余不变信息
# x3 = test1.fc1.weight[:,84:]
# #最后额外增加的信息
# x4 = test1.fc1.weight[:,-1:]
# test1.fc1.weight = torch.nn.Parameter(
#                     torch.cat((x1, x2), dim=1))
# test1.fc1.weight = torch.nn.Parameter(
#                     torch.cat((test1.fc1.weight, x3), dim=1))
# test1.fc1.weight = torch.nn.Parameter(
#                     torch.cat((test1.fc1.weight, x4), dim=1))
#
# torch.save(test1.state_dict(), './model//vdn/9m_vs_8m/vdn_net_params.pkl')
"""VDN 2m_vs_3m"""
# x1 = test1.fc1.weight[:,:24]
# #因为队友增加的信息
# x2= test1.fc1.weight[:,29:41]
# test1.fc1.weight = torch.nn.Parameter(
#                     torch.cat((x1, x2), dim=1))
#
# torch.save(test1.state_dict(), './model//vdn/2m_vs_3m/rnn_net_params.pkl')


"""QTRAN 9m_vs_8m"""
# test1.load_state_dict(torch.load(r"./model/qtran_base/8m/rnn_net_params.pkl", map_location='cuda:0'))
# # #修改代码
# x1 = test1.fc1.weight[:,:84]
# #因为队友增加的信息
# x2= test1.fc1.weight[:,79:84]
# #剩余不变信息
# x3 = test1.fc1.weight[:,84:]
# #最后额外增加的信息
# x4 = test1.fc1.weight[:,-1:]
# test1.fc1.weight = torch.nn.Parameter(
#                     torch.cat((x1, x2), dim=1))
# test1.fc1.weight = torch.nn.Parameter(
#                     torch.cat((test1.fc1.weight, x3), dim=1))
# test1.fc1.weight = torch.nn.Parameter(
#                     torch.cat((test1.fc1.weight, x4), dim=1))
#
# torch.save(test1.state_dict(), './model//qtran_base/9m_vs_8m/rnn_net_params.pkl')

"""COMA 9m_vs_8m"""
# test1.load_state_dict(torch.load(r"./model/coma/8m/rnn_params.pkl", map_location='cuda:0'))
# # #修改代码
# x1 = test1.fc1.weight[:,:84]
# #因为队友增加的信息
# x2= test1.fc1.weight[:,79:84]
# #剩余不变信息
# x3 = test1.fc1.weight[:,84:]
# #最后额外增加的信息
# x4 = test1.fc1.weight[:,-1:]
# test1.fc1.weight = torch.nn.Parameter(
#                     torch.cat((x1, x2), dim=1))
# test1.fc1.weight = torch.nn.Parameter(
#                     torch.cat((test1.fc1.weight, x3), dim=1))
# test1.fc1.weight = torch.nn.Parameter(
#                     torch.cat((test1.fc1.weight, x4), dim=1))
#
# torch.save(test1.state_dict(), './model//coma/9m_vs_8m/rnn_params.pkl')





"""Reinforce+commnet"""
# class CommNet(nn.Module):
#     def __init__(self, input_shape, rnn_hidden_dim,n_actions):
#         super(CommNet, self).__init__()
#         self.encoding = nn.Linear(input_shape, rnn_hidden_dim)  # 对所有agent的obs解码
#         self.f_obs = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)  # 每个agent根据自己的obs编码得到hidden_state，用于记忆之前的obs
#         self.f_comm = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)  # 用于通信
#         self.decoding = nn.Linear(rnn_hidden_dim, n_actions)
#         self.input_shape = input_shape
# test1 = CommNet(42, 64, 9)
#
# test1.load_state_dict(torch.load(r"./model/reinforce+commnet/3m/rnn_params.pkl", map_location='cuda:0'))
# # #修改代码
# x1 = test1.encoding.weight[:,:29]
# #因为队友增加的信息
# x2= test1.encoding.weight[:,24:29]
# #剩余不变信息
# x3 = test1.encoding.weight[:,29:]
# #最后额外增加的信息
# x4 = test1.encoding.weight[:,-1:]
# test1.encoding.weight = torch.nn.Parameter(
#                     torch.cat((x1, x2), dim=1))
# test1.encoding.weight = torch.nn.Parameter(
#                     torch.cat((test1.encoding.weight, x3), dim=1))
# test1.encoding.weight = torch.nn.Parameter(
#                     torch.cat((test1.encoding.weight, x4), dim=1))
#
# torch.save(test1.state_dict(), './model//reinforce+commnet/4m_vs_3m/rnn_params.pkl')


"""Reinforcement+G2ANET"""
# class G2ANet(nn.Module):
#     def __init__(self, input_shape, rnn_hidden_dim,attention_dim,n_actions):
#         super(G2ANet, self).__init__()
#
#         # Encoding
#         self.encoding = nn.Linear(input_shape, rnn_hidden_dim)  # 对所有agent的obs解码
#         self.h = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)  # 每个agent根据自己的obs编码得到hidden_state，用于记忆之前的obs
#
#         # Hard
#         # GRU输入[[h_i,h_1],[h_i,h_2],...[h_i,h_n]]与[0,...,0]，输出[[h_1],[h_2],...,[h_n]]与[h_n]， h_j表示了agent j与agent i的关系
#         # 输入的iputs维度为(n_agents - 1, batch_size * n_agents, rnn_hidden_dim * 2)，
#         # 即对于batch_size条数据，输入每个agent与其他n_agents - 1个agents的hidden_state的连接
#         self.hard_bi_GRU = nn.GRU(rnn_hidden_dim * 2, rnn_hidden_dim, bidirectional=True)
#         # 对h_j进行分析，得到agent j对于agent i的权重，输出两维，经过gumble_softmax后取其中一维即可，如果是0则不考虑agent j，如果是1则考虑
#         self.hard_encoding = nn.Linear(rnn_hidden_dim * 2, 2)  # 乘2因为是双向GRU，hidden_state维度为2 * hidden_dim
#
#         # Soft
#         self.q = nn.Linear(rnn_hidden_dim, attention_dim, bias=False)
#         self.k = nn.Linear(rnn_hidden_dim, attention_dim, bias=False)
#         self.v = nn.Linear(rnn_hidden_dim, attention_dim)
#
#         # Decoding 输入自己的h_i与x_i，输出自己动作的概率分布
#         self.decoding = nn.Linear(rnn_hidden_dim + attention_dim, n_actions)
#         self.input_shape = input_shape
#
# test1 = G2ANet(42, 64,32, 9)
#
# test1.load_state_dict(torch.load(r"./model/reinforce+g2anet/8m/rnn_params.pkl", map_location='cuda:0'))
# # #修改代码
# x1 = test1.encoding.weight[:,:29]
# #因为队友增加的信息
# x2= test1.encoding.weight[:,24:29]
# #剩余不变信息
# x3 = test1.encoding.weight[:,29:]
# #最后额外增加的信息
# x4 = test1.encoding.weight[:,-1:]
# test1.encoding.weight = torch.nn.Parameter(
#                     torch.cat((x1, x2), dim=1))
# test1.encoding.weight = torch.nn.Parameter(
#                     torch.cat((test1.encoding.weight, x3), dim=1))
# test1.encoding.weight = torch.nn.Parameter(
#                     torch.cat((test1.encoding.weight, x4), dim=1))
#
# torch.save(test1.state_dict(), './model//reinforce+g2anet/9m_vs_8m/rnn_params.pkl')


"""Maven"""
# class HierarchicalPolicy(nn.Module):
#     def __init__(self, state_shape,noise_dim):
#         super(HierarchicalPolicy, self).__init__()
#         self.fc_1 = nn.Linear(state_shape, 128)
#         self.fc_2 = nn.Linear(128, noise_dim)
#
#     def forward(self, state):
#         x = f.relu(self.fc_1(state))
#         q = self.fc_2(x)
#         prob = f.softmax(q, dim=-1)
#         return prob
#
# test1 = HierarchicalPolicy(48,16)
#
# test1.load_state_dict(torch.load(r"./model/maven/3m/z_policy_params.pkl", map_location='cuda:0'))
# #修改代码
# x1 = test1.fc_1.weight[:,:29]
# #因为队友增加的信息
# x2= test1.fc_1.weight[:,24:29]
# #剩余不变信息
# x3 = test1.fc_1.weight[:,29:]
# #最后额外增加的信息
# x4 = test1.fc_1.weight[:,-1:]
# test1.fc_1.weight = torch.nn.Parameter(
#                     torch.cat((x1, x2), dim=1))
# test1.fc_1.weight = torch.nn.Parameter(
#                     torch.cat((test1.fc_1.weight, x3), dim=1))
# test1.fc_1.weight = torch.nn.Parameter(
#                     torch.cat((test1.fc_1.weight, x4), dim=1))
#
# torch.save(test1.state_dict(), './model//maven/4m_vs_3m/z_policy_params.pkl')

# class BootstrappedRNN(nn.Module):
#     def __init__(self, input_shape, rnn_hidden_dim,noise_dim,n_agents,n_actions):
#         super(BootstrappedRNN, self).__init__()
#         self.fc = nn.Linear(input_shape, rnn_hidden_dim)
#         self.rnn = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)
#         self.hyper_w = nn.Linear(noise_dim + n_agents, rnn_hidden_dim * n_actions)
#         self.hyper_b = nn.Linear(noise_dim + n_agents, n_actions)
#
#     def forward(self, obs, hidden_state, z):
#         agent_id = obs[:, -self.args.n_agents:]
#         hyper_input = torch.cat([z, agent_id], dim=-1)
#
#         x = f.relu(self.fc(obs))
#         h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
#         h = self.rnn(x, h_in)
#         h = h.view(-1, 1, self.args.rnn_hidden_dim)
#
#         hyper_w = self.hyper_w(hyper_input)
#         hyper_b = self.hyper_b(hyper_input)
#         hyper_w = hyper_w.view(-1, self.args.rnn_hidden_dim, self.args.n_actions)
#         hyper_b = hyper_b.view(-1, 1, self.args.n_actions)
#
#         q = torch.bmm(h, hyper_w) + hyper_b
#         q = q.view(-1, self.args.n_actions)
#         return q, h
#
# test1 = BootstrappedRNN(102,64,16,8,14)
# #
# test1.load_state_dict(torch.load(r"./model/maven/8m/rnn_net_params.pkl", map_location='cuda:0'))
# #修改代码
# x1 = test1.fc.weight[:,:84]
# #因为队友增加的信息
# x2= test1.fc.weight[:,79:84]
# #剩余不变信息
# x3 = test1.fc.weight[:,84:]
# #最后额外增加的信息
# x4 = test1.fc.weight[:,-1:]
# test1.fc.weight = torch.nn.Parameter(
#                     torch.cat((x1, x2), dim=1))
# test1.fc.weight = torch.nn.Parameter(
#                     torch.cat((test1.fc.weight, x3), dim=1))
# test1.fc.weight = torch.nn.Parameter(
#                     torch.cat((test1.fc.weight, x4), dim=1))
# x5 = test1.hyper_w.weight[:,-1:]
# test1.hyper_w.weight = torch.nn.Parameter(
#                     torch.cat((test1.hyper_w.weight, x5), dim=1))
#
# x6 = test1.hyper_b.weight[:,-1:]
# test1.hyper_b.weight = torch.nn.Parameter(
#                     torch.cat((test1.hyper_b.weight, x6), dim=1))
#
# torch.save(test1.state_dict(), './model//maven/9m_vs_8m/rnn_net_params.pkl')