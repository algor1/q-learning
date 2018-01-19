import numpy as np
from collections import deque
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense,LSTM,Dropout


class game:
    def __init__(self):
        self.pole=np.array([['.','.','.'],['.','.','.'],['.','.','.']])
    def show(self):
        print(self.pole)
    def turn(self,x,y,v):
        if self.pole[x,y]=='.':
            self.pole[x,y]=v
            return True
        else:
            return False

    def check(self):
        for i in range(3):
            if self.pole[i,0]==self.pole[i,1]==self.pole[i,2] and self.pole[i,0] != '.':
                return self.pole[i,0]
        for j in range(3):
            if self.pole[0,j]==self.pole[1,j]==self.pole[2,j] and self.pole[0,j] != '.':
                return self.pole[0,j]
        if self.pole[0, 0] == self.pole[1, 1] == self.pole[2, 2] and self.pole[1, 1] != '.':
            return self.pole[1, 1]
        if self.pole[2, 0] == self.pole[1, 1] == self.pole[0, 2] and self.pole[1, 1] != '.':
            return self.pole[1, 1]

        return ('.')
    def reset(self):
        self.pole=np.array([['.','.','.'],['.','.','.'],['.','.','.']])



class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.reward_size=1
        self.done_size=1
        self.num_of_previous_turns=1 #сколько предыдущих ходов показывать сети
        self.mem_len=9
        # 0:4,4:8,8:12
        self.memory_ranges_state=[self.state_size*i for i in range(1,self.num_of_previous_turns+1) ]
        self.memory_ranges_action=[self.memory_ranges_state[-1]+self.action_size*i for i in range(1,self.num_of_previous_turns+1) ]
        self.memory = None
        self.memory_pred = None
        self.clearmem()
        self.fit_pred_loss=1
        self.fit_env_loss=1
        self.stop_fit_pred_loss=0.05
        self.stop_fit_env_loss=0.05
        self.stop_fit_done_loss = 0.1
        self.model_rew = self._build_model_rew()

        self.loss_done_mem=deque(maxlen=100)
        # self.loss_done_clearmem()

    def remember(self, state, action, reward, done):
        self.memory[0:-1] = self.memory[1:]
        self.memory[-1] = np.concatenate((self.memory[-2, self.memory_ranges_state[0]:self.memory_ranges_state[-1]]
                                          , state
                                          , self.memory[-2,
                                            self.memory_ranges_action[0]:self.memory_ranges_action[-1]]
                                          , [action]
                                          , [reward]
                                          , [done]))


    def clearmem(self):
        self.memory = np.zeros((self.mem_len,
                                self.state_size * self.num_of_previous_turns +
                                self.action_size * self.num_of_previous_turns +
                                self.reward_size + self.done_size
                                ),
                               dtype=float)

    def _build_model_rew(self):
        input_dim = (self.state_size + self.action_size) * self.num_of_previous_turns
        model = Sequential()
        model.add(Dense(input_dim * 2, input_shape=(input_dim,), activation='relu', kernel_initializer='normal'))
        # model.add(Dropout(0.4))
        model.add(Dense(self.mem_len * 2, activation='relu'))
        # model.add(Dropout(0.4))
        model.add(Dense(self.reward_size, kernel_initializer='normal', activation='sigmoid'))
        print(model.summary())
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        print(model.output_shape)
        print(model.input_shape)
        return model


if __name__ == "__main__":

    g = game()
    agentx=DQNAgent(9,9)
    agento = DQNAgent(9, 9)
    g.show()
    # g.turn(0, 2, 'x')
    for e in range(EPISODES):
        agentx.clearmem()
        agento.clearmem()
        state = g.reset()
        state = np.array(state)
        time=0
        while True :
            time+=1
            g.show()
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
            if time//100000*100000==time:
                agent.save_done("done50-3-"+str(time))
