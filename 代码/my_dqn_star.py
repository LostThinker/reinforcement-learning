# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Activation, Conv2D, Dense, Flatten
import cv2
from keras.layers import MaxPooling2D

EPISODES = 1000  # 训练次数为1000轮


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # 记忆池队列
        self.gamma = 0.95    # 伽马
        self.epsilon = 1.0  # 随机性探索概率
        self.epsilon_min = 0.01  # 随机性探索最小概率
        self.epsilon_decay = 0.995  # 随机性探索概率衰减
        self.learning_rate = 0.001
        self.replace_target_iter = 300
        self.learn_step_counter = 0
        self._build_model()

    def _build_model(self):
        # CNN
        # ------------------ 构建预测神经网络 ------------------
        model_eval = Sequential()
        # 第一层卷积
        model_eval.add(Conv2D(padding="same", kernel_size=(8, 8), filters=16, strides=4,  input_shape=(84, 84, 1)))
        model_eval.add(Activation('relu'))
        # 第二层卷积
        model_eval.add(Conv2D(padding="same", kernel_size=(4, 4), filters=32))
        model_eval.add(Activation('relu'))
        model_eval.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))

        # 最后一层为全连接层
        model_eval.add(Flatten())
        model_eval.add(Dense(action_size, activation='linear'))
        self.model_eval = model_eval
        # 编译
        self.model_eval.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        # ------------------ 构建目标神经网络 ------------------

        model_target = Sequential()
        # 第一层卷积
        model_target.add(Conv2D(padding="same", kernel_size=(8, 8), filters=16, strides=4,  input_shape=(84, 84, 1)))
        model_target.add(Activation('relu'))

        # 第二层卷积
        model_target.add(Conv2D(padding="same", kernel_size=(4, 4), filters=32))
        model_target.add(Activation('relu'))
        model_target.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))

        # 最后一层为全连接层
        model_target.add(Flatten())
        model_target.add(Dense(action_size, activation='linear'))
        self.model_target = model_target


    def remember(self, state, action, reward, next_state, done):
        # 记忆池的增加
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # 行为的决策
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model_eval.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        # DQN算法的核心，进行参数的更新
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.model_target.set_weights(self.model_eval.get_weights())
        minibatch = random.sample(self.memory, batch_size)
        states, targets_f = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model_target.predict(next_state)))
            target_f = self.model_eval.predict(state)
            target_f[0][action] = target
            # Filtering out states and targets for training
            states.append(state[0])
            targets_f.append(target_f[0])
        history = self.model_eval.fit(np.array(states), np.array(targets_f), epochs=1, verbose=0)
        # Keeping track of loss
        loss = history.history['loss'][0]
        # 随机性进行衰减
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.learn_step_counter += 1
        return loss

    def load(self, name):
        # 加载已经生成的模型
        self.model_target.load_weights(name)

    def save(self, name):
        # 保存模型
        self.model_target.save_weights(name)


if __name__ == "__main__":

    env = gym.make('SpaceInvaders-v0')
    env = env.unwrapped
    state_size = env.observation_space.shape[0]
    print('state_size:{}'.format(state_size))
    action_size = env.action_space.n
    print('action_size:{}'.format(action_size))
    agent = DQNAgent(state_size, action_size)
    # 若利用保存的模型，下一行不要注释
    # agent.load("star-dqn.h5")
    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()
        # 按照论文中要求对图像处理，处理为84*84的灰度图像
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        state = state[50:210,0:160]
        state = cv2.resize(state,(84,84))
        state = state[np.newaxis, :, :, np.newaxis]
        sum_reward = 0
        score = 0
        a = 3  # 命数
        for time in range(0, 100000000):
            #  能保证游戏不会因为时间过长而结束，而是以惩罚结束。
            env.render()
            if time % 1 == 0:
                # 按照论文所说的进行帧抽取，三帧重复一样的动作，三帧更新一次
                action = agent.act(state)
            next_state, reward, done, new_a = env.step(action)  # a是剩下几条命
            score += reward

            if not done:
                #  死命减分
                if new_a['ale.lives'] < a:
                    reward = -10
                    a = new_a['ale.lives']
            else:
                reward = -10

            sum_reward += reward
            # 图像的格式修改
            next_state = cv2.cvtColor(next_state, cv2.COLOR_BGR2GRAY)
            next_state = next_state[50:210,0:160]
            next_state = cv2.resize(next_state,(84,84))
            next_state = next_state[np.newaxis, :, :, np.newaxis]
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, e: {:.2}"
                      .format(e, EPISODES,  agent.epsilon))
                break
            if len(agent.memory) > batch_size:

                loss = agent.replay(batch_size)

                if time % 30 == 0:
                    print("episode: {}/{}, score: {}, loss: {:.4f}".format(e, EPISODES, score, loss))
        # 保存模型
        # if e % 10 == 0 :
            # agent.save("star-dqn.h5")
