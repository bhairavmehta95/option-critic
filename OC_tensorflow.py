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

        self.Y = tf.placeholder(dtype=tf.float32, shape=[None])
        self.A = tf.placeholder(dtype=tf.int32, shape=[None])
        self.O = tf.placeholder(dtype=tf.int32, shape=[None])      


    def get_new_action_in_option(self):
        self.current_option_policy = self.model.intra_options[self.current_option]
        self.current_action = np.argmax(self.model.current_option_policy) \
                if self.rng.rand() > self.args.action_epsilon \
                else self.rng.randint(self.args.num_actions)


    def train(self, x, new_x, action, raw_reward, done, death):
        end_ep = done or (death and self.args.death_ends_episode)
        self.frame_counter += 1
        
        self.total_reward += raw_reward
        reward = np.clip(raw_reward, -1, 1)

        self.x_seq[self.t_counter] = np.copy(x)
        self.o_seq[self.t_counter] = np.copy(self.current_o)
        self.a_seq[self.t_counter] = np.copy(action)
        self.r_seq[self.t_counter] = np.copy(float(reward)) - (float(self.terminated)*self.delib*float(self.frame_counter > 1))

        # self.terminated = self.get_termination([self.current_s])[0][self.current_o] > self.rng.rand()
        # self.termination_counter += self.terminated

        self.t_counter += 1

        option_terminated = (self.terminated and self.t_counter >= self.args.update_freq)

        if self.t_counter == self.args.max_update_freq or end_ep or option_term: # Time to update
            d = (self.delib*float(self.frame_counter > 1))
            value = self.value
            reward = 0 if end_ep else value
            values = []


        if self.terminated: # TODO: Where to do this?
            self.current_option = np.argmax(self.model.policy_over_options) \
                if self.rng.rand() > self.args.option_epsilon \
                else self.rng.randint(self.args.num_options)





    def reset_storing(self):
        self.a_seq = np.zeros((self.args.max_update_freq,), dtype="int32")
        self.o_seq = np.zeros((self.args.max_update_freq,), dtype="int32")
        self.r_seq = np.zeros((self.args.max_update_freq,), dtype="float32")
        self.x_seq = np.zeros((self.args.max_update_freq, self.args.concat_frames*(1 if self.args.grayscale else 3),84,84),dtype="float32")
        self.t_counter = 0
