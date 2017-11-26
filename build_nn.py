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


class Network():
    def __init__(self, model_in, nopt, ob_space, ac_space, nenvs, nsteps, nstack, reuse=False):
        with tf.variable_scope("model", reuse=reuse):
            self.nbatches = nenvs * nsteps
            self.nh, self.nw, self.nc = ob_space.shape
            self.ob_shape = (self.nbatches, self.nh, self.nw, self.nc*nstack)
            self.nact = ac_space.n
            self.nopt = nopt
            self.rng = np.random.RandomState(0) # TODO - what seed to send?

            self.observations = tf.placeholder(shape=self.ob_shape, dtype=tf.uint8)

            self.summary = []

            input_tensor = tf.cast(self.observations, tf.float32) / 255.0

            print("Building following model...")
            print(model_in)

            self.model = model_in
            self.input_size = ob_space.shape
            self.out_size = model_in[-3]["out_size"]

            dnn_type = True # TODO: Make this cuda enabled?

            # Build Main NN
            for i, m in enumerate(model_in):
                if m["model_type"] == 'option' or m["model_type"] == 'value':
                    break

                new_layer = self.create_layer(input_tensor, m, dnn_type=dnn_type)
                input_tensor = new_layer

            self.state_representation = input_tensor

            m = dict()
            m["model_type"] = 'value'

            # Build Value output
            self.value_fn = self.create_layer(input_tensor, m, dnn_type=dnn_type, name='value')

            m = dict()
            m["model_type"] = 'option'

            # Build Option Related End Networks
            self.termination_fn = self.create_layer(input_tensor, m, dnn_type=dnn_type, name='termination_fn')
            self.q_values_options = self.create_layer(input_tensor, m, dnn_type, name='q_values_options')

            self.intra_option_policies = list()
            for i in range(self.nopt):
                intra_option = self.create_layer(input_tensor, m, dnn_type=dnn_type, name='intra_option_{}'.format(i))
                self.intra_option_policies.append(intra_option)

            self.initial_state = [] # For reproducability with OpenAI code

            print("Build complete.")


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
                name=model["name"]
            )


        elif model["model_type"] == "value":
            layer = tf.layers.dense(
                inputs=inputs,
                units=1,
                activation=None,
                name='value'
            )

        elif model["model_type"] == "option":
            if name == 'termination_fn':
                layer = tf.layers.dense(
                    inputs=inputs,
                    units=self.nopt,
                    activation=tf.nn.sigmoid,
                    name='termination_fn'
                )

            elif name == 'q_values_options':
                layer = tf.layers.dense(
                    inputs=inputs,
                    units=self.nopt,
                    activation=None,
                    name='q_values_options'
                )

            else: # Intraoption Policy
                layer = tf.layers.dense(
                    inputs=inputs,
                    units=self.nact,
                    activation=tf.nn.softmax,
                    name=name
                )

        else:
            print("UNKNOWN LAYER NAME")
            raise NotImplementedError

        return layer


    def setup_tensorflow(self, sess, writer):
        self.sess = sess
        self.writer = writer

        tf.global_variables_initializer().run(session=sess)
        tf.local_variables_initializer().run(session=sess)


    def get_policy_over_options(self, observations):
        q_values_options, value = self.sess.run(self.q_values_options, self.value_fn, feed_dict={self.observations: observations})
        return q_values_options.argmax() if self.rng.rand() > self.args.option_eps else self.rng.randint(self.nopt), value


    def value(self, observations):
        value = self.sess.run(self.value_fn, feed_dict={self.observations: observations})
        return value


    def step(self, observations, current_options):
        option_action_probabilities, value = self.sess.run([
                tf.nn.softmax(self.intra_option_policies, dim=2),
                self.value_fn,
            ],
            feed_dict={self.observations: observations}
        )

        act_to_take = []
        for idx, option in enumerate(current_options):
            act_prob = option_action_probabilities[option][idx]
            act_to_take.append(self.rng.choice(range(self.nact), p=act_prob))

        return act_to_take, value[:, 0]


    def update_options(self, observations, current_options, option_eps, delib_cost):
        costs = np.zeros(len(current_options))

        q_values_options, termination_prob = self.sess.run([
                self.q_values_options,
                self.termination_fn
            ],
            feed_dict={self.observations: observations}
        )

        for idx, option in enumerate(current_options):
            # Terminated, pick a new option
            if termination_prob[idx][option] > self.rng.rand():
                new_option = q_values_options[idx].argmax() \
                    if self.rng.rand() > option_eps \
                    else self.rng.randint(self.nopt)

                costs[idx] = delib_cost
                current_options[idx] = new_option
            else:
                continue # Did not terminate
                
        return current_options, costs
