import time
import gym
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
import torch
import rl_utils
from multinetwork import MADDPG
import numpy as np

def make_env(scenario_name):
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                        scenario.observation)
    return env
#The initial number of training sessions
num_episodes = 50000
#The length of each training round
episode_length = 100  # 每条序列的最大长度
#Capacity of experience replay pool
buffer_size = 100000
#Size of hidden layer
hidden_dim = 64
#Learning rate of actor network
actor_lr = 1e-2
#Learning rate of critic network
critic_lr = 1e-2
#Attenuation factor
gamma = 0.95
#soft update
tau = 1e-2
# sampling
batch_size = 1024
#cpu or gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#update cycle
update_interval = 200
#Minimum sampling frequency
minimal_size = 4000
#Expanded number of intelligent agents
add_num = 13
#environment
env_id = "simple_spread"
env = make_env(env_id)
replay_buffer = rl_utils.ReplayBuffer(buffer_size)

state_dims = []
action_dims = []
for action_space in env.action_space:
    action_dims.append(action_space.n)
for state_space in env.observation_space:
    state_dims.append(state_space.shape[0])
critic_input_dim = sum(state_dims) + sum(action_dims)

maddpg = MADDPG(env, device, actor_lr, critic_lr, hidden_dim, state_dims,
                action_dims, critic_input_dim, gamma, tau, add_num, env_id)


def evaluate(env_id, maddpg, n_episode=10, episode_length=25):

    env = make_env(env_id)
    returns = np.zeros(len(env.agents))
    for _ in range(n_episode):
        obs = env.reset()
        for t_i in range(episode_length):
            actions = maddpg.take_action(obs, explore=False)
            obs, rew, done, info = env.step(actions)
            rew = np.array(rew)
            returns += rew / n_episode
    return returns.tolist()


return_list = []  # record for the return
total_step = 0
#Train
for i_episode in range(num_episodes):
    state = env.reset()
    # ep_returns = np.zeros(len(env.agents))
    for e_i in range(episode_length):
        actions = maddpg.take_action(env,state, explore=True)
        next_state, reward, done, _ = env.step(actions)
        replay_buffer.add(state, actions, reward, next_state, done)
        state = next_state

        total_step += 1
        # env.render()
        if replay_buffer.size(
        ) >= minimal_size and total_step % update_interval == 0:
            sample = replay_buffer.sample(batch_size)

            def stack_array(x):
                rearranged = [[sub_x[i] for sub_x in x]
                              for i in range(len(x[0]))]
                return [
                    torch.FloatTensor(np.vstack(aa)).to(device)
                    for aa in rearranged
                ]

            sample = [stack_array(x) for x in sample]
            for a_i in range(len(env.agents)):
                maddpg.update(sample, a_i)
            maddpg.update_all_targets()
    if i_episode % 1000==0:
        for a_i in range(len(env.agents)):
            maddpg.save(i_episode,a_i)
    if i_episode == num_episodes-1:
        for a_i in range(len(env.agents)):
            maddpg.save(a_i)
    # if (i_episode + 1) % 100 == 0:
    #     ep_returns = evaluate(env_id, maddpg, n_episode=100)
    #     return_list.append(ep_returns)
    #     print(f"Episode: {i_episode+1}, {ep_returns}")

state = env.reset()
#Test
while True :
    state = env.reset()
    for i in range(100):
        actions = maddpg.take_action(env, state, explore=False)
        next_state, reward, done, _ = env.step(actions)
        # print(reward)
    # print(env.agents[0].state.p_pos)
        state = next_state
        time.sleep(0.05)
        env.render()