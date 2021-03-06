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
        self.num_of_previous_turns=1 #сколько предыдущих ходов показывать сети
        self.mem_len=50
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


        self.model_act = self._build_model_action()
        self.model_done = self._build_model_done()
        self.loss_done_mem=deque(maxlen=100)
        # self.loss_done_clearmem()


    def _build_model_done(self):

        input_dim = (self.state_size+self.action_size)*self.num_of_previous_turns
        model = Sequential()
        model.add(Dense(input_dim*2,input_shape=(input_dim,),activation='relu',kernel_initializer='normal'))
        # model.add(Dropout(0.4))
        model.add(Dense(self.mem_len*2, activation='relu'))
        # model.add(Dropout(0.4))
        model.add(Dense(self.reward_size, kernel_initializer='normal', activation='sigmoid'))
        print(model.summary())
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
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
        model.add(Dense(self.action_size, activation='linear'))
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


    def act_done(self,state,action):
        batch = np.array([np.concatenate((state, [action]), axis=0)])
        done_values = self.model_done.predict(batch)
        return done_values  # returns action


    def fit_act (self,prev_state,state,action_prev,action):

        batch = np.array([np.concatenate((prev_state, state, action_prev), axis=0)])
        y=np.array([action])
        hist =self.model_act.fit(batch,y,batch_size=1,epochs=1,verbose=0)
        print('act ', hist.history['loss'][0])

    def fit_done(self,end):
        max_loss_done=0.0

        batch = self.memory[:,:self.memory_ranges_action[-1]]
        y=list()

        done = self.memory[:, -1]
        for i in range (done.shape[0]):
            y.append(sum([done[j]*2**(i-j) for j in range (i,done.shape[0])]))

        # print(y)
        numsteps=self.mem_len if end==1 else 1

        for i in range(numsteps):
            loss = self.model_done.evaluate(np.array([batch[i], ]), np.array([y[i], ]), batch_size=1, verbose=0)
            if loss[0] > self.stop_fit_pred_loss:
                max_loss_done=loss[0] if max_loss_done<loss[0] else max_loss_done
                hist = self.model_done.fit(np.array([batch[i], ]), np.array([y[i], ]), batch_size=1,
                                           epochs=int(loss[0]*20), verbose=0)
                print('done', hist.history['loss'][0])
            self.loss_done_mem.append(max_loss_done)

        return  sum(self.loss_done_mem)/float(len(self.loss_done_mem))

    def loss_done(self):
        m=self.memory
        m_del=list()
        for i in range(self.memory.shape[0]):
            if sum(m[i, :] == 0) == m.shape[1]:
                m_del.append(i)
        m=np.delete(m,m_del,axis=0)

        batch = m[:,:self.memory_ranges_action[-1]]
        done = m[:, -1]
        if m.shape[0]==0:
            loss=[0,0]
        else:
            loss = self.model_done.evaluate(batch,done, batch_size=1, verbose=0)
        print (loss[0])

        return  loss[0]


    def load_act(self, name):
        self.model_act.load_weights(name)
    def load_done(self, name):
        self.model_done.load_weights(name)

    def save_act(self, name):
        self.model_act.save_weights(name)
    def save_done(self, name):
        self.model_done.save_weights(name)

    # def check_answer(self,action):
    #     self.clearmem_pred()
    #
    #     p_state_prev = self.memory[-1, self.memory_ranges_state[0]:self.memory_ranges_state[-1]]
    #     p_action = action
    #     p_action_prev=self.memory[-1, self.memory_ranges_action[0]:self.memory_ranges_action[-1]]
    #
    #     for i in range(agent.mem_len):
    #         p_state=self.act_env(p_state_prev,p_action_prev,[p_action])
    #         self.remember_pred(p_state_prev,p_state,p_action_prev,p_action,0,0)
    #
    #         p_state_prev = self.memory_pred[-1, self.memory_ranges_state[0]:self.memory_ranges_state[-1]]
    #         p_action_prev = self.memory_pred[-1, self.memory_ranges_action[0]:self.memory_ranges_action[-1]]
    #         p_action=self.act_act(p_state_prev,p_state,p_action_prev)
    #     # проверим reward
    #     reward=self.act_done()
    #     return reward

if __name__ == "__main__":
    gym.envs.register(
        id='CartPoleExtraLong-v0',
        entry_point='gym.envs.classic_control:CartPoleEnv',
        max_episode_steps=5000,
        reward_threshold=4500.0,
    )
    env = gym.make('CartPoleExtraLong-v0')


    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    print('load previous? (y/n)')
    if input()=="y":
        # agent.load_act("act5")
        agent.load_done("done50-3-100000")
    done = False


    time_prev = 0
    rand_multipl=1
    score=list()
    agent_loss = 1

    for e in range(EPISODES):
        agent.clearmem()
        state = env.reset()
        state = np.array(state)
        state_prev = agent.memory[-1, agent.memory_ranges_state[0]:agent.memory_ranges_state[-1]]
        action_prev = agent.memory[-1, agent.memory_ranges_action[0]:agent.memory_ranges_action[-1]]
        time=0
        while True :
            time+=1
            env.render()
            # agent_loss = 0

            action=random.randint(0,1)
            # agent_loss = agent.loss_done()
            if True or agent_loss<0.05:
            # if len(score)>50 and sum(score[-50:-25])<sum(score[-25:]):
                check0=agent.act_done(state,action)
                check1 = agent.act_done(state, (action - 1) ** 2)
                if check0>check1:
                    action = (action - 1) ** 2
            else: print('-----')
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else 0
            agent.remember(state, action, reward, done)
            state = next_state
            if (time >10 ) or done:
                d=1 if done else 0
                # agent.fit_env()
                # agent.fit_pred()
                # agent_loss=agent.fit_done(d)

                # print(agent.memory[:,agent.memory_ranges_action[-1]])

                if done:
                    print("episode: {}/{}, score: {}, e: {}"
                          .format(e, EPISODES, time, time_prev))
                    score.append(time)
                    if time_prev < time:
                        agent.save_done("done50-3-")
                    break
            if time_prev<time:
                time_prev=time
            if time//100*100==time:
                print('+')
            if time==4999:
                agent.save_done("done50-3-"+str(time))

    print("min",min(score))
    print("max", max(score))
    print("sum", sum(score))
    print('save? (y/n)')
    if input()=="y":
        # agent.save_act("act5")
        # agent.save_env("env5")
        # agent.save_pred("pred8")
        agent.save_done("done50")


