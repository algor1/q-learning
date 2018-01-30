import numpy as np
import random
from collections import deque
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense,LSTM,Dropout


EPISODES = 10
agents_loss=0.2 # порог доверия к нейросети
rewards=[1,-1,0.3]#[Выигрыш,проигрыш , ничья]

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
        if np.argwhere(g.pole == '.').shape[0]==0:
            return 'N'
        return '.'

        return ('.')
    def reset(self):
        self.pole=np.array([['.','.','.'],['.','.','.'],['.','.','.']])

    def polex(self):
        x=self.pole.copy()
        x[x == '.'] = '0'
        x[x == 'x'] = '1'
        x[x == 'o'] = '2'
        x=np.float16(x)
        x[x == 2.] = -1.
        return x
    def poleo(self):
        x=self.pole.copy()
        x[x == '.'] = '0'
        x[x == 'o'] = '1'
        x[x == 'x'] = '2'
        x=np.float16(x)
        x[x == 2.] = -1.
        return x

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.reward_size=1
        self.done_size=1
        self.num_of_previous_turns=1 #сколько предыдущих ходов показывать сети
        self.mem_len=5
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
                                          , action
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
        model.add(Dense(self.reward_size, kernel_initializer='normal', activation='tanh'))
        print(model.summary())
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        print(model.output_shape)
        print(model.input_shape)
        return model

    def save_rew(self, name):
        self.model_rew.save_weights(name)
    def load_rew(self, name):
        self.model_rew.load_weights(name)

    def act_rew(self,state,action):

        batch = np.array([np.concatenate((state, action), axis=0)])
        done_values = self.model_rew.predict(batch)
        return done_values[0,0]  # returns action

    def fit_rew(self):
        max_loss_done=0.0

        batch = self.memory[:,:self.memory_ranges_action[-1]]
        y=list()

        rew = self.memory[:,-2: -1]
        for i in range (rew.shape[0]):
            y.append(sum([rew[j]*10**(i-j) for j in range (i,rew.shape[0])]))

        # print(y)
        numsteps=self.mem_len

        for i in range(numsteps):
            loss = self.model_rew.evaluate(np.array([batch[i], ]), np.array([y[i], ]), batch_size=1, verbose=0)
            if loss[0] > self.stop_fit_pred_loss:
                max_loss_done=loss[0] if max_loss_done<loss[0] else max_loss_done
                hist = self.model_rew.fit(np.array([batch[i], ]), np.array([y[i], ]), batch_size=1,
                                           epochs=int(loss[0]*20), verbose=0)
                print('rew', hist.history['loss'][0])
            self.loss_done_mem.append(max_loss_done)

        return  sum(self.loss_done_mem)/float(len(self.loss_done_mem))


if __name__ == "__main__":

    g = game()
    agentx=DQNAgent(9,9)
    agento = DQNAgent(9, 9)
    agento.load_rew("rew-o")

    aggentx_loss=1
    aggento_loss=1
    # g.turn(0, 2, 'x')
    for e in range(EPISODES):
        agentx.clearmem()
        agento.clearmem()
        state = g.reset()
        state = np.array(state)
        time=0
        rewardx=0
        rewardo=0
        actionx=np.zeros(9)
        actionx[0]=1
        actiono=np.zeros(9)
        actiono[0]=1
        done=False
        while True :
            time+=1


            # ход Х
            if not done:
                # Проверить все варианты, выбрать лучший
                check_max_x = -100
                g.show()
                index=int(input("ход Х. Введите индекс от 0 до 8:  "))
                #делаем ход Х
                g.turn(index//3,index-3*(index//3),'x')

                # проверить выйгрыш check
                if g.check()!='.':
                    # если выйгрыш
                    done= True
                    if g.check() == 'x':
                        rewardx = rewards[0]
                        rewardo = rewards[1]
                    if g.check() == 'o':
                        rewardx = rewards[1]
                        rewardo = rewards[0]
                    if g.check() =='N':
                            rewardx = rewards[2]
                            rewardo = rewards[2]
                # Записать в память Х
            agentx.remember(g.polex().flat,actionx,rewardx,done)

            # ход O
            if not done:
                # Проверить все варианты, выбрать лучший
                check_max_o = -100
                for ni, pl in enumerate(g.pole.flat):
                    acto = np.zeros(9)
                    acto[ni] = 1
                    if pl == '.':
                        if aggento_loss < agents_loss or True:
                            check = agento.act_rew(g.poleo().flat, acto)
                        else:
                            check = random.random()
                        if check_max_o < check:
                            check_max_o = check
                            actiono = acto
                            index=ni
                #делаем ход О
                g.turn(index//3,index-3*(index//3),'o')
                # проверить выйгрыш check
                if g.check() != '.':
                    # если выйгрыш
                    done = True
                    if g.check() == 'x':
                        rewardx = rewards[0]
                        rewardo = rewards[1]
                    if g.check() == 'o':
                        rewardx = rewards[1]
                        rewardo = rewards[0]
                    if g.check() == 'N':
                        rewardx = rewards[2]
                        rewardo = rewards[2]

                # Записать в память О
            agento.remember(g.poleo().flat, actiono, rewardo, done)

            if done:
                # fit X O
                agento.memory[-1,-2]=rewardo
                agentx.memory[-1, -2] = rewardx
                aggentx_loss = agentx.fit_rew()
                aggento_loss = agento.fit_rew()
                g.show()
                break
    # agentx.save_rew("rew-x")
    # agento.save_rew("rew-o")




            # action=random.randint(0,1)
            # # agent_loss = agent.loss_done()
            # if True or agent_loss<0.05:
            # # if len(score)>50 and sum(score[-50:-25])<sum(score[-25:]):
            #     check0=agent.act_done(state,action)
            #     check1 = agent.act_done(state, (action - 1) ** 2)
            #     if check0>check1:
            #         action = (action - 1) ** 2
            # else: print('-----')
            # next_state, reward, done, _ = env.step(action)
            # reward = reward if not done else 0
            # agent.remember(state, action, reward, done)
            # state = next_state
            # if (time >10 ) or done:
            #     d=1 if done else 0
            #     # agent.fit_env()
            #     # agent.fit_pred()
            #     # agent_loss=agent.fit_done(d)
            #
            #     # print(agent.memory[:,agent.memory_ranges_action[-1]])
            #
            #     if done:
            #         print("episode: {}/{}, score: {}, e: {}"
            #               .format(e, EPISODES, time, time_prev))
            #         score.append(time)
            #         if time_prev < time:
            #             agent.save_done("done50-3-")
            #         break
            # if time_prev<time:
            #     time_prev=time
            # if time//100*100==time:
            #     print('+')
            # if time//100000*100000==time:
            #     agent.save_done("done50-3-"+str(time))