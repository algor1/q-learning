# -*- coding: utf-8 -*-
'''3 модели вместе'''
import random
import gym
import numpy as np
from collections import deque
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense,LSTM,Dropout
from tensorflow.contrib.keras.api.keras.optimizers import Adam

EPISODES = 3000


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = 1
        self.reward_size=1
        self.done_size=1
        self.num_of_previous_turns=4 #сколько предыдущих ходов показывать сети
        self.mem_len=10
        # 0:4,4:8,8:12
        self.memory_ranges_state=[self.state_size*i for i in range(1,self.num_of_previous_turns+1) ]
        self.memory_ranges_action=[self.memory_ranges_state[-1]+self.action_size*i for i in range(1,self.num_of_previous_turns+1) ]
        self.memory = None
        self.memory_pred = None
        self.clearmem()
        self.clearmem_pred()
        self.fit_pred_loss=1
        self.fit_env_loss=1
        self.stop_fit_pred_loss=0.05
        self.stop_fit_env_loss=0.05
        self.stop_fit_done_loss = 0.1


        self.model_done = self._build_model_done()


    def _build_model_done(self):

        input_dim = self.state_size
        model = Sequential()
        model.add(Dense(input_dim*2,input_shape=(input_dim,),activation='relu',kernel_initializer='normal'))
        # model.add(Dropout(0.4))
        model.add(Dense(input_dim*2, activation='relu'))
        # model.add(Dropout(0.4))
        model.add(Dense(self.reward_size, kernel_initializer='normal', activation='sigmoid'))
        print(model.summary())
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.output_shape)
        print(model.input_shape)
        return model


    def remember(self, state, action, reward, done):
        self.memory[0:-1] = self.memory[1:]
        self.memory[-1] = np.concatenate((self.memory[-2, self.memory_ranges_state[0]:self.memory_ranges_state[-1]]
                                          , state
                                          , self.memory[-2, self.memory_ranges_action[0]:self.memory_ranges_action[-1]]
                                          , [action]
                                          , [reward]
                                          , [done]))
    def remember_pred(self,state_prev,state,action_prev,action, reward, done):
        self.memory_pred[0:-1]=self.memory_pred[1:]
        self.memory_pred[-1]=np.concatenate((self.memory_pred[-2,self.memory_ranges_state[0]:self.memory_ranges_state[-1]]
                                        ,state
                                        ,self.memory_pred[-2,self.memory_ranges_action[0]:self.memory_ranges_action[-1]]
                                        ,[action]
                                        ,[reward]
                                        ,[done]))


    def clearmem(self):
        self.memory = np.zeros((self.mem_len,
                                self.state_size * self.num_of_previous_turns +
                                self.action_size * self.num_of_previous_turns +
                                self.reward_size + self.done_size
                                ),
                               dtype=float)
    def clearmem_pred(self):
        self.memory_pred = np.zeros((self.mem_len,
                                self.state_size * self.num_of_previous_turns +
                                self.action_size * self.num_of_previous_turns +
                                self.reward_size + self.done_size
                                ),
                               dtype=float)


    def act_done(self):
        batch = self.memory[:,self.memory_ranges_state[-2]:self.memory_ranges_state[-1]]

        done_values = self.model_done.predict(batch)
        # print(pred_values)
        return done_values  # returns action


    def fit_done(self):

        batch = self.memory[:,self.memory_ranges_state[-2]:self.memory_ranges_state[-1]]

        y = self.memory[:, -1]
        print(y)
        for i in range(self.mem_len):
            loss = self.model_done.evaluate(np.array([batch[i], ]), np.array([y[i], ]), batch_size=1, verbose=0)
            if loss[0] > self.stop_fit_done_loss:
                hist = self.model_done.fit(np.array([batch[i], ]), np.array([y[i], ]), batch_size=1, epochs=int(loss[0]*10),
                                           verbose=0)
                print('done', hist.history['loss'][0])




    def load_done(self, name):
        self.model_done.load_weights(name)


    def save_done(self, name):
        self.model_done.save_weights(name)


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    print('load previous? (y/n)')
    if input()=="y":

        agent.load_done("done6")
    done = False


    time_prev = 0
    rand_multipl=1
    for e in range(EPISODES):
        agent.clearmem()

        state = env.reset()
        state = np.array(state)
        state_prev = agent.memory[-1, agent.memory_ranges_state[0]:agent.memory_ranges_state[-1]]
        action_prev = agent.memory[-1, agent.memory_ranges_action[0]:agent.memory_ranges_action[-1]]
        for time in range(500):
            # env.render()
            action=random.randint(0,1)
            next_state, reward, done, _ = env.step(action)

            reward = reward if not done else 0
            agent.remember(state, action, reward, done)
            state = next_state

            if (time in range(10,500,10))or done:
                print("episode: {}/{}, score: {}, e: {}"
                      .format(e, EPISODES, time, time_prev))
                d=1 if done else 0
                print (np.concatenate(([agent.memory[:,-1]],[agent.act_done()[:,0]]),axis=0))
                agent.fit_done()

                if done:
                    break

    print('save? (y/n)')
    if input()=="y":

        agent.save_done("done6")


