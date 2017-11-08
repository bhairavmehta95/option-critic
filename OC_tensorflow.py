import numpy as np
import tensorflow as tf

from model import Model

class AOC_Tensorflow():
    def __init__(self, num_actions, num_options, id_num, sess):
        self.id_num = id_num
        self.num_actions = num_actions
        self.num_options = num_options
        self.sess = sess

        self.reset_storing()

        self.rng = np.random.RandomState(100+id_num)

        model = [
            {"model_type": "conv", "filter_size": [8,8], "pool": [1,1], "stride": [4,4], "out_size": 32, "name": "conv1"},
            {"model_type": "conv", "filter_size": [4,4], "pool": [1,1], "stride": [2,2], "out_size": 64, "name": "conv2"},
            {"model_type": "conv", "filter_size": [3,3], "pool": [1,1], "stride": [1,1], "out_size": 64, "name": "conv3"},
            {"model_type": "flatten"},
            {"model_type": "mlp", "out_size": 512, "activation": "sigmoid", "name": "fc1"},
            {"model_type": "option"}
        ]

        self.model = Model(model)  


    def get_new_action_in_option(self):
        self.current_option_policy = self.model.intra_options[self.current_option]
        self.current_action = np.argmax(self.model.current_option_policy) \
                if self.rng.rand() > self.args.action_epsilon \
                else self.rng.randint(self.args.num_actions)


    def store(self, x, new_x, action, raw_reward, done, death):
        end_ep = done or (death and self.args.death_ends_episode)
        self.frame_counter += 1
        
        self.total_rewards += raw_reward
        reward = np.clip(raw_reward, -1, 1)

        self.x_seq[self.t_counter] = np.copy(x)
        self.o_seq[self.t_counter] = np.copy(self.current_o)
        self.a_seq[self.t_counter] = np.copy(action)
        self.r_seq[self.t_counter] = np.copy(float(reward)) - (float(self.terminated)*self.delib*float(self.frame_counter > 1))

        self.terminated = self.model.terminations[self.current_o] > self.rng.rand()
        self.termination_counter += self.terminated
        self.t_counter += 1

        option_term = (self.terminated and self.t_counter >= self.args.update_freq)

        if self.t_counter == self.args.max_update_freq or end_ep or option_term:
            d = (self.delib * float(self.frame_counter > 1))
            V = self.value if self.terminated else self.q_values_options
            R = 0 if end_ep else V
            V = []

            for j in range(self.t_counter-1, -1, -1):
                R = np.float32(self.r_seq[j] + self.args.gamma * R) # Discount
                V.append(R)

            # TODO: Update weights?

            self.reset_storing()

        if not end_ep:
            self.update_internal_state(new_x)


    def reset_storing(self):
        self.a_seq = np.zeros((self.args.max_update_freq,), dtype="int32")
        self.o_seq = np.zeros((self.args.max_update_freq,), dtype="int32")
        self.r_seq = np.zeros((self.args.max_update_freq,), dtype="float32")
        self.x_seq = np.zeros((self.args.max_update_freq, self.args.concat_frames*(1 if self.args.grayscale else 3),84,84),dtype="float32")
        self.t_counter = 0


    def update_internal_state(self, x):
        self.current_s = x # TODO ?
        self.delib = self.args.delib_cost

        if self.terminated:
            self.current_o = self.model.policy_over_options # TODO: Eps
            self.o_tracker_chosen[self.current_o] += 1

        self.o_tracker_steps[self.current_o] += 1

    def get_action(self, x):
        p = self.get_policy([self.current_s], [self.current_o])
        return self.rng.choice(range(self.num_actions), p=p[-1])

    def get_policy_over_options(self, s):
        return self.get_q(s)[0].argmax() if self.rng.rand() > self.args.option_epsilon else self.rng.randint(self.args.num_options)
