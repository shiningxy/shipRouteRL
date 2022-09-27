# -*- coding: utf-8 -*-

import time
from gridworld import shipRouteWapper
from agent import QLearningAgent
from parl.utils import summary
from shiproute import shipRouteEnv
from utils import save_results, plot_rewards, smooth

def train(env, agent, render=False):
    total_steps = 0  # 记录每个episode走了多少step
    total_reward = 0
    obs = env.reset()  # 重置环境, 重新开一局（即开始新的一个episode）
    action = agent.sample(obs)  # 根据算法选择一个动作

    while True:
        next_obs, reward, done, _, info = env.step(action)  # 与环境进行一个交互
        next_action = agent.sample(next_obs)  # 根据算法选择一个动作
        # 训练 Q-Learning 算法
        agent.learn(obs, action, reward, next_obs, done)

        action = next_action  # 更新动作
        obs = next_obs  # 更新观察
        total_reward += reward  # 更新累计回报（奖励）
        total_steps += 1  # 计算step数

        if render:
            env.render()  # 进行一次渲染
        if done:
            break
    return total_reward, total_steps


def test(env, agent):
    rewards = []  # record rewards for all episodes
    steps = []
    total_steps = 0  # 记录每个episode走了多少step
    total_reward = 0
    obs = env.reset()
    while True:
        total_steps += 1
        action = agent.predict(obs)  # greedy
        next_obs, reward, done, _, info = env.step(action)
        total_reward += reward
        obs = next_obs
        # time.sleep(0.2)
        env.render()  # 进行一次渲染
        if done:
            print('test reward = %.1f' % (total_reward))
            steps.append(total_steps)
            rewards.append(total_reward)
            break
        steps.append(total_steps)
        rewards.append(total_reward)
    env.close()
    return rewards, steps


def main():
    # 创建环境
    env = shipRouteEnv()  # 0 up, 1 right, 2 down, 3 left
    env = shipRouteWapper(env)

    # 创建智能体
    agent = QLearningAgent(
        obs_n=env.observation_space.n,
        act_n=env.action_space.n,
        learning_rate=0.1,
        gamma=0.9,
        e_greed=0.1)

    # 读取Q表格（策略） 第一次训练生成策略，之后的训练可以读取策略以加快收敛速度
    # agent.restore(npy_file='./q_table.npy')

    is_render = False
    rewards = []  # record rewards for all episodes
    steps = []
    times = []
    epochs = 5000
    for episode in range(epochs):
        t1 = time.time()
        ep_reward, ep_steps = train(env, agent, is_render)
        print('Episode %s: steps = %s , reward = %.1f' % (episode, ep_steps,
                                                          ep_reward))
        steps.append(ep_steps)
        rewards.append(ep_reward)
        # 保存训练过程中，每个回合的累计回报（奖励）
        summary.add_scalar('q_learning/episode rewards', ep_reward,
                           episode)
        t2 = time.time()
        times.append(round(t2-t1,3))
        # 训练过程不渲染，只在测试阶段渲染一次
        if episode % 50 == 0:
            is_render = False
        else:
            is_render = False
    # 保存训练结果并画图
    res_dic = {'step':steps, 'rewards':rewards, 'times':times}
    save_results(res_dic, tag='train', path="results")
    plot_rewards(res_dic['rewards'],  path = "results", tag = "train", save_fig=True, show_fig=True)
    # 训练结束，查看算法效果
    # 训练过程不渲染，只在测试阶段渲染一次
    rewards, steps = test(env, agent)
    # 保存测试结果并画图
    res_dic = {'step':steps, 'rewards':rewards}
    save_results(res_dic, tag='test', path="results")
    plot_rewards(res_dic['rewards'],  path = "results", tag = "test", save_fig=True, show_fig=True)

    # 保存Q表格（策略）
    agent.save()
    # 参考VisualDL可视化分析工具使用介绍.ipynb
    # os.system("visualdl --logdir='train_log/main' --host=192.168.1.2")

if __name__ == "__main__":
    main()
