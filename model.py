import numpy as np
import tensorflow as tf
import math, csv, time, sys, os, pdb, copy


def get_activation(activation):
    if activation == "softmax":
        output = tf.nn.softmax
    elif activation is None:
        output = None
    elif activation == "tanh":
        output = tf.nn.tanh
    elif activation == "relu":
        output = tf.nn.relu
    elif "leaky_relu" in activation:
        output = lambda x: tf.nn.relu(x, alpha=float(activation.split(" ")[1]))
    elif activation == "linear":
        output = None
    elif activation == "sigmoid":
        output = tf.nn.sigmoid
    else:
        print("activation not recognized:", activation)
        raise NotImplementedError

    return output


def get_init(model, t, conv=False):
    initializers = {"zeros": tf.constant_initializer(0.), "norm": tf.random_normal_initializer(0.1)}

    if conv:
        return tf.random_normal_initializer()

    if t not in model:
        if t == "b":
            return tf.constant_initializer(0.)
    
        return tf.random_normal_initializer()

    elif isinstance(model[t], basestring):
        return initializers[model[t]]

    elif isinstance(model[t], int):
        return tf.constant_initializer(model[t])

    else:
        return model[t]


class Model():
    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    def get_activation(self, model):
        activation = model["activation"] if "activation" in model else "linear"
        return get_activation(activation)

    def create_layer(self, inputs, model, dnn_type=True, name=None):
        layer = None
        if model["model_type"] == "conv":
            poolsize = tuple(model["pool"]) if "pool" in model else (1,1)
            stride = tuple(model["stride"]) if "stride" in model else (1,1)

            layer = tf.layers.conv2d(
                inputs=inputs, 
                filters=model["out_size"], 
                kernel_size=model["filter_size"], 
                strides=stride, 
                activation=self.get_activation(model),
                kernel_initializer=get_init(model, "W", conv=True),
                bias_initializer=get_init(model, "b"),
                padding="valid" if "pad" not in model else model["pad"],
                name=model["name"]
            )

        elif model["model_type"] == "flatten":
            return tf.reshape(inputs, [-1, 3136]) # TODO: Use Reshape and Model size

        elif model["model_type"] == "mlp":
            layer = tf.layers.dense(
                inputs=inputs, 
                units=model["out_size"],
                activation=self.get_activation(model),
                kernel_initializer=get_init(model, "W"),
                bias_initializer=get_init(model, "b"),
                name=model["name"]
            )


        elif model["model_type"] == "value":
            layer = tf.layers.dense(
                inputs=inputs,
                units=1,
                activation=None,
                kernel_initializer=get_init(model, "W"),
                bias_initializer=get_init(model, "b"),
                name='value'
            )

        elif model["model_type"] == "option":
            if name == 'termination_fn':
                layer = tf.layers.dense(
                    inputs=inputs,
                    units=self.num_options,
                    activation=tf.nn.sigmoid,
                    kernel_initializer=get_init(model, "W"),
                    bias_initializer=get_init(model, "b"),
                    name='termination_fn'
                )

            elif name == 'q_values_options':
                layer = tf.layers.dense(
                    inputs=inputs,
                    units=self.num_options,
                    activation=None,
                    kernel_initializer=get_init(model, "W"),
                    bias_initializer=get_init(model, "b"),
                    name='q_values_options'
                )

            else:
                layer = tf.layers.dense(
                    inputs=inputs,
                    units=self.num_actions,
                    activation=None,
                    kernel_initializer=get_init(model, "W"),
                    bias_initializer=get_init(model, "b"),
                    name=name
                )

        else:
            print("UNKNOWN LAYER NAME")
            raise NotImplementedError

        return layer

    def setup_tensorflow(self, sess, writer):
        self.sess = sess
        self.writer = writer

    def __init__(self, model_in, scope, args, trainer, input_size=None, rng=1234, dnn_type=False, num_options=4, num_actions=3):
        """
        example model:
        model = [{"model_type": "conv", "filter_size": [5,5], "pool": [1,1], "stride": [1,1], "out_size": 5},
                 {"model_type": "conv", "filter_size": [7,7], "pool": [1,1], "stride": [1,1], "out_size": 15},
                 {"model_type": "mlp", "out_size": 300, "activation": "tanh"},
                 {"model_type": "mlp", "out_size": 10, "activation": "softmax"}]
        """

        self.args = args
        self.reset_storing()

        with tf.variable_scope(scope):
            tf.set_random_seed(rng)
            self.rng = np.random.RandomState(rng + 1) # Should every single one get same seed?

            self.num_options = num_options
            self.num_actions = num_actions
            self.observations = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.float32)

            self.summary = []

            input_tensor = self.observations

            print("Building following model...")
            print(model)

            self.model = model
            self.input_size = input_size
            self.out_size = model_in[-3]["out_size"]
            self.dnn_type = dnn_type

            # Build Main NN
            for i, m in enumerate(model):
                if m["model_type"] == 'option' or m["model_type"] == 'value':
                    break

                new_layer = self.create_layer(input_tensor, m, dnn_type=dnn_type)
                input_tensor = new_layer

            self.state_representation = input_tensor

            m = dict()
            m["model_type"] = 'value'

            self.value_fn = self.create_layer(input_tensor, m, dnn_type=dnn_type, name='value')

            m = dict()
            m["model_type"] = 'option'
            
            # Build Option Related End Networks 
            self.termination_fn = self.create_layer(input_tensor, m, dnn_type=dnn_type, name='termination_fn')
            self.q_values_options = self.create_layer(input_tensor, m, dnn_type, name='q_values_options')

            self.intra_options_q_vals = list()
            for i in range(self.num_options):
                intra_option = self.create_layer(input_tensor, m, dnn_type=dnn_type, name='intra_option_{}'.format(i))
                self.intra_options_q_vals.append(intra_option)

            print("Build complete.")

            if scope != 'global':
                print("Building worker specific operations.")

                self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
                self.options = tf.placeholder(shape=[None], dtype=tf.int32)
                self.targets = tf.placeholder(shape=[None], dtype=tf.float32)

                self.batch_size = tf.range(tf.shape(self.actions)[0])

                self.responsible_options = tf.stack([self.batch_size, self.options], axis=1)
                self.responsible_actions = tf.stack([self.batch_size, self.actions], axis=1)

                self.option_q_vals = tf.gather_nd(params=self.q_values_options, indices=self.responsible_options) # Extract q values for each option
                
                self.disconnected_q_vals = tf.stop_gradient(self.q_values_options)
                self.disconnected_option_q_vals = tf.gather_nd(params=self.disconnected_q_vals, indices=self.responsible_options) # Extract q values for each option
                
                self.terminations = tf.gather_nd(params=self.termination_fn, indices=self.responsible_options)
                
                self.action_values = tf.gather_nd(params=self.intra_options_q_vals, indices=self.responsible_options)
                self.action_values = tf.gather_nd(params=self.action_values, indices=self.responsible_actions)

                # TODO: Check axis?
                self.value = tf.reduce_max(self.q_values_options, axis=1) * (1 - self.args.option_eps) + (self.args.option_eps * tf.reduce_mean(self.q_values_options, axis=1))
                self.disconnected_value = tf.stop_gradient(self.value)

                # Losses
                self.value_loss = 0.5 * tf.reduce_mean(self.args.critic_coef * tf.square(self.targets - self.option_q_vals))
                self.policy_loss = -1 * tf.reduce_mean((tf.log(self.action_values) + self.args.log_eps) * (self.targets - self.disconnected_option_q_vals))
                self.termination_gradient_loss = tf.reduce_mean(self.terminations * ((self.disconnected_option_q_vals - self.disconnected_value) + self.args.delib))
                
                # TODO: Sum over all actions, not just action_values  ; -/+ 1
                self.entropy = -1 * tf.reduce_mean(self.action_values*tf.log(self.action_values + self.args.log_eps))
                
                self.loss = self.policy_loss - self.entropy - self.value_loss - self.termination_gradient_loss

                self.summary.append(tf.summary.scalar('policy_loss', self.policy_loss))
                self.summary.append(tf.summary.scalar('value_loss', self.value_loss))
                self.summary.append(tf.summary.scalar('termination_gradient', self.termination_gradient_loss))
                self.summary.append(tf.summary.scalar('entropy', self.entropy))
                
                # Move to A2C Variant
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)

                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))

                self.summary_op = tf.summary.merge(self.summary)


    def get_policy_over_options(self, observations):
        q_values_options = self.sess.run(self.q_values_options, {self.observations: observations})
        return q_values_options.argmax() if self.rng.rand() > self.args.option_eps else self.rng.randint(self.num_options)


    def get_action(self, observations, current_option):
        actions = self.sess.run(self.intra_options_q_vals[current_option], {self.observations: observations})
        return self.rng.choice(range(self.num_actions), p=actions)


    def get_termination(self, observations, current_option):
        termination_prob = self.sess.run(self.termination_fn, {self.observations: observations})
        return termination_prob[current_option] > self.rng.rand()


    def reset_storing(self):
        self.a_seq = np.zeros((self.args.max_update_freq,), dtype="int32")
        self.o_seq = np.zeros((self.args.max_update_freq,), dtype="int32")
        self.r_seq = np.zeros((self.args.max_update_freq,), dtype="float32")
        self.x_seq = np.zeros((self.args.max_update_freq, self.args.concat_frames*(1 if self.args.grayscale else 3),84,84),dtype="float32")
        self.t_counter = 0

    # TODO: Start here - November 8th Work Session
    def store(self, x, new_x, action, raw_reward, done, death):
        end_ep = done or (death and self.args.death_ends_episode)
        self.frame_counter += 1

        self.total_reward += raw_reward
        reward = np.clip(raw_reward, -1, 1)

        self.x_seq[self.t_counter] = np.copy(x)
        self.o_seq[self.t_counter] = np.copy(self.current_o)
        self.a_seq[self.t_counter] = np.copy(action)
        self.r_seq[self.t_counter] = np.copy(float(reward)) - (float(self.terminated)*self.delib*float(self.frame_counter > 1))

        # Where is best place to get the state?
        self.terminated = self.get_termination()
        self.termination_counter += self.terminated

        self.t_counter += 1

        # do n-step return to option termination. 
        # cut off at self.args.max_update_freq
        # min steps: self.args.update_freq (usually 5 like a3c)
        # this doesn't make option length a minimum of 5 (they can still terminate). only batch size
        option_term = (self.terminated and self.t_counter >= self.args.update_freq)
        if self.t_counter == self.args.max_update_freq or end_ep or option_term: # Time to update
            d = (self.delib*float(self.frame_counter > 1)) # add delib if termination because it isn't part of V
            V = self.get_V([self.current_s])[0]-d if self.terminated else self.get_q([self.current_s])[0][self.current_o]
            R = 0 if end_ep else V
            V = []

            for j in range(self.t_counter-1,-1,-1): # Easy way to reset to 0
                R = np.float32(self.r_seq[j] + self.args.gamma*R) # discount
                V.append(R)
                self.update_weights(self.x_seq[:self.t_counter], self.a_seq[:self.t_counter], V[::-1], 
                self.o_seq[:self.t_counter], self.t_counter, self.delib+self.args.margin_cost)
                self.reset_storing()

        if not end_ep:
            self.update_internal_state(new_x)

if __name__ == '__main__':
    model = [
        {"model_type": "conv", "filter_size": [8,8], "pool": [1,1], "stride": [4,4], "out_size": 32, "name": "conv1"},
        {"model_type": "conv", "filter_size": [4,4], "pool": [1,1], "stride": [2,2], "out_size": 64, "name": "conv2"},
        {"model_type": "conv", "filter_size": [3,3], "pool": [1,1], "stride": [1,1], "out_size": 64, "name": "conv3"},
        {"model_type": "flatten"},
        {"model_type": "mlp", "out_size": 512, "activation": "relu", "name": "fc1"},
        {"model_type": "option"},
        {"model_type": "value"}
    ]

    # Hack; TODO: Add Argparser
    args = type('', (), {})()

    args.option_eps = 0.01
    args.critic_coef = 0.01
    args.delib = 0.001
    args.max_update_freq = 20
    args.concat_frames = 1
    args.grayscale = 1
    args.log_eps = 0.01

    trainer = tf.train.RMSPropOptimizer(learning_rate=0.001)
    m_global = Model(model, scope='global', trainer=trainer, args=args)
    m = Model(model, scope='worker_1', trainer=trainer, args=args)

    with tf.Session() as sess:

        init_op = tf.global_variables_initializer()
        l_init_op = tf.local_variables_initializer()

        writer = tf.summary.FileWriter('log', sess.graph)
        m.setup_tensorflow(sess, writer)

        sess.run(init_op)
        sess.run(l_init_op)


        while True: 
            obs = np.random.rand(4, 84, 84, 4)
            actions = np.random.randint(3, size=4)
            options = np.random.randint(4, size=4)
            targets = np.random.random(size=4)

            feed_dict = {
                m.observations : obs,
                m.actions: actions,
                m.options: options,
                m.targets: targets
            }

            init_ops = [m.grad_norms, m.summary_op]
            _, summary = sess.run(init_ops, feed_dict=feed_dict)
            writer.add_summary(summary)