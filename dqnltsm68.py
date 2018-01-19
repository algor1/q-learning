# -*- coding: utf-8 -*-
'''3 модели вместе'''
import random
import gym
import numpy as np
from collections import deque
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense,LSTM,Dropout
from tensorflow.contrib.keras.api.keras.optimizers import Adam

EPISODES = 10


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
        self.stop_fit_done_loss = 0.05

        self.model_pred = self._build_model_predict()
        self.model_env = self._build_model_enviroment()
        self.model_act = self._build_model_action()
        self.model_done = self._build_model_done()

    def _build_model_predict(self):

        input_dim = (self.state_size+self.action_size)*self.num_of_previous_turns
        model = Sequential()
        model.add(Dense(input_dim*2,input_shape=(input_dim,),activation='relu'))
        # model.add(Dropout(0.4))
        model.add(Dense(input_dim*2, activation='relu'))
        # model.add(Dropout(0.4))
        model.add(Dense(1, activation='tanh'))
        print(model.summary())
        model.compile(loss='mean_squared_error',
                      optimizer='adam')
        print(model.output_shape)
        print(model.input_shape)
        return model

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

    def _build_model_enviroment(self):
        # Neural Net for Deep-Q learning Model
        input_dim = self.state_size*(self.num_of_previous_turns-1)+self.action_size*self.num_of_previous_turns
        model = Sequential()
        model.add(Dense(input_dim*2,input_shape=(input_dim,),activation='relu'))
        # model.add(Dropout(0.4))
        model.add(Dense(input_dim*2, activation='relu'))
        # model.add(Dropout(0.4))
        model.add(Dense(self.state_size, activation='tanh'))
        print(model.summary())
        model.compile(loss='mean_squared_error',
                      optimizer='adam')
        print(model.output_shape)
        print(model.input_shape)
        return model

    def _build_model_action(self):

        # Neural Net for Deep-Q learning Model
        input_dim = self.state_size * self.num_of_previous_turns + self.action_size * (self.num_of_previous_turns - 1)
        model = Sequential()
        model.add(Dense(input_dim * 2, input_shape=(input_dim,), activation='relu'))
        # model.add(Dropout(0.4))
        model.add(Dense(input_dim * 2, activation='tanh'))
        #model.add(Dropout(0.4))
        model.add(Dense(self.action_size, activation='softmax'))
        print(model.summary())
        model.compile(loss='mean_squared_error',
                      optimizer='adam')
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

    def act_act(self, prev_state,state,action_prev):
        batch=np.array([np.concatenate((prev_state,state,action_prev),axis=0)])
        act_values = self.model_act.predict(batch)
        act_return= 0 if act_values[0,0]<0.5 else 1
        return act_return  # returns action

    def act_pred(self):
        batch=np.concatenate((self.memory[:,0:self.memory_ranges_state[-1]]
                                        ,self.memory[:,self.memory_ranges_state[-1]:self.memory_ranges_action[-1]]),axis=1)
        pred_values = self.model_pred.predict(batch)
        # print(pred_values)
        return pred_values  # returns action

    def act_done(self):
        batch = self.memory_pred[:,self.memory_ranges_state[-2]:self.memory_ranges_state[-1]]

        done_values = self.model_done.predict(batch)
        if np.max(done_values)>0.5 :print('max  done     ',np.max(done_values))
        return done_values  # returns action

    def act_env (self,state_prev,action_prev,action):
        batch = np.array([np.concatenate((state_prev,action_prev,action), axis=0)])
        env =self.model_env.predict(batch,verbose=0)
        # print(env)
        return env[0]

    def fit_act (self,prev_state,state,action_prev,action):

        batch = np.array([np.concatenate((prev_state, state, action_prev), axis=0)])
        y=np.array([action])
        hist =self.model_act.fit(batch,y,batch_size=1,epochs=1,verbose=0)
        print('act ', hist.history['loss'][0])

    def fit_done(self):

        batch = self.memory[:,self.memory_ranges_state[-2]:self.memory_ranges_state[-1]]

        y = self.memory[:, -1]
        print(y)
        for i in range(self.mem_len):
            loss = self.model_done.evaluate(np.array([batch[i], ]), np.array([y[i], ]), batch_size=1, verbose=0)
            if loss[0] > self.stop_fit_pred_loss:
                hist = self.model_done.fit(np.array([batch[i], ]), np.array([y[i], ]), batch_size=1, epochs=1,
                                           verbose=0)
                print('done', hist.history['loss'][0])

    def fit_pred (self):

        batch=np.concatenate((self.memory[:,0:self.memory_ranges_state[-1]]
                                        ,self.memory[:,self.memory_ranges_state[-1]:self.memory_ranges_action[-1]]),axis=1)
        y =list()
        for i,yi in enumerate(self.memory[:,-2]):
            y.append(yi/(i+1))
        print (y)
        y=np.array(y)
        i=0
        loss= self.model_pred.evaluate(np.array( [batch[i],]),np.array( [y.sum(),]),batch_size=1,verbose=0)
        if loss>self.stop_fit_pred_loss:

            hist =self.model_pred.fit(np.array( [batch[i],]),np.array( [y.sum(),]),batch_size=1,epochs=1,verbose=0)
            print('pred',hist.history['loss'][0])
            # if hist.history['loss'][0]**2<self.fit_pred_loss:
            #     self.fit_pred_loss=hist.history['loss'][0]
            #     self.save_pred('pred3')

    def fit_env (self):
        batch=np.concatenate((self.memory[:,self.memory_ranges_state[0]:self.memory_ranges_state[-1]]
                                        ,self.memory[:,self.memory_ranges_state[-1]:self.memory_ranges_action[-1]]),axis=1)
        y=self.memory[:,:self.memory_ranges_state[0]]

        loss =self.model_env.evaluate(batch,y,batch_size=self.mem_len,verbose=0)
        if loss>self.stop_fit_env_loss:
            hist =self.model_env.fit(batch,y,batch_size=self.mem_len,epochs=1,verbose=0)
            print('env ',hist.history['loss'][0])
            # if hist.history['loss'][0]**2<self.fit_env_loss:
            #     self.fit_env_loss=hist.history['loss'][0]
            #     self.save_env('env2')


    def load_act(self, name):
        self.model_act.load_weights(name)
    def load_env(self, name):
        self.model_env.load_weights(name)
    def load_pred(self, name):
        self.model_pred.load_weights(name)
    def load_done(self, name):
        self.model_done.load_weights(name)

    def save_act(self, name):
        self.model_act.save_weights(name)
    def save_env(self, name):
        self.model_env.save_weights(name)
    def save_pred(self, name):
        self.model_pred.save_weights(name)
    def save_done(self, name):
        self.model_done.save_weights(name)

    def check_answer(self,action):
        self.clearmem_pred()

        p_state_prev = self.memory[-1, self.memory_ranges_state[0]:self.memory_ranges_state[-1]]
        p_action = action
        p_action_prev=self.memory[-1, self.memory_ranges_action[0]:self.memory_ranges_action[-1]]

        for i in range(agent.mem_len):
            p_state=self.act_env(p_state_prev,p_action_prev,[p_action])
            self.remember_pred(p_state_prev,p_state,p_action_prev,p_action,0,0)

            p_state_prev = self.memory_pred[-1, self.memory_ranges_state[0]:self.memory_ranges_state[-1]]
            p_action_prev = self.memory_pred[-1, self.memory_ranges_action[0]:self.memory_ranges_action[-1]]
            p_action=self.act_act(p_state_prev,p_state,p_action_prev)
        # проверим reward
        reward=self.act_done()
        return reward

if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    print('load previous? (y/n)')
    if input()=="y":
        # agent.load_act("act5")
        agent.load_env("env5")
        agent.load_pred("pred5")
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
            guess=False
            # while not guess:
            #
            #     action = agent.act_act(state_prev,state,action_prev)
            #     print("action before  ",action)
            #     check_answ1 =agent.check_answer(action)
            #     check_answ2 =agent.check_answer((action - 1) ** 2)
            #
            #     if np.max(check_answ1)>0.2 and not(np.max(check_answ2)>0.2 and np.argmax(check_answ1)>np.argmax(check_answ2)):
            #         result_action = (action - 1) ** 2
            #         agent.fit_act(state_prev,state,action_prev,result_action)
            #         guess = False
            #     else:
            #         result_action = action
            #         agent.fit_act(state_prev, state, action_prev, result_action)
            #         guess = True
            #     if time <= agent.num_of_previous_turns:
            #         guess = True
            # # print (check_answ1,np.argmin(check_answ1),check_answ2,np.argmin(check_answ2))
            # # print("action after  ", action)
            action=random.randint(0,1)

            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else 0



            agent.remember(state, action, reward, done)
            state = next_state
            if (time >=10)or done:
                print("episode: {}/{}, score: {}, e: {}"
                      .format(e, EPISODES, time, time_prev))
                d=1 if done else 0
                # agent.fit_env()
                agent.fit_pred()
                # agent.fit_done()


                if done:
                    break
                # elif time>=time_prev*0.2: # последовательность не короче 20% от лучшей
                #     if time> time_prev: time_prev=time
                #     agent.fit_act(d)

    # print('save? (y/n)')
    # if input()=="y":
        # agent.save_act("act5")
        # agent.save_env("env5")
    agent.save_pred("pred8")
        # agent.save_done("done5")


