import numpy as np
import tensorflow as tf
import math, csv, time, sys, os, pdb, copy


def get_activition(activation):
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
        print "activation not recognized:", activation
        raise NotImplementedError

    return output


def get_init(model, t):
    initializers = {"zeros": tf.constant_initializer(0.), "norm": tf.normal_initializer(0.1)}
    if t not in m:
        if t == "b":
            return tf.constant_initializer(0.)
    
        return tf.glorot_uniform_initializer()

    elif isinstance(m[t], basestring):
        return inits[m[t]]

    elif isinstance(m[t], int):
        return tf.constant_initializer(m[t])

    else:
        return m[t]


class MLP3D():
    def __init__(self, input_size=None, num_options=None, num_outputs=None, activation="softmax"):
        option_out_size = num_options

        def _initializer(num_options, input_size, option_out_size):
            limits = (6./np.sqrt(input_size + option_out_size))/num_options
            return np.random.uniform(size=(num_options, input_size, option_out_size), high=limits, low=-limits)

        self.options_W = tf.get_variable(
            name="options_W", 
            size=[num_options, input_size, option_out_size], 
            dtype=tf.float32,
            initializer=_modified_glorot(num_options, input_size, option_out_size)
        )

        self.options_b = tf.get_variable(
            name="options_b",
            size=[num_options, option_out_size],
            dtype=tf.float32,
            initializer=tf.zeros_initializer()
        )

        self.activation = get_activation(activation)
        self.params = [self.options_W, self.options_b]

    def apply(self, inputs, option=None):
        W = self.options_W[option]
        b = self.options_b[option]

        inputs = tf.expand_dims(inputs, len(inputs) - 1)
        out = tf.matmul(inputs, W) + b

        # Original: T.sum(inputs.dimshuffle(0,1,'x')*W, axis=1) + b

        return out if self.activation is None else self.activation(out)

    def save_params(self):
        return [i.get_value() for i in self.params]

    def load_params(self, values):
        print("LOADING NNET..")
        for p, value in zip(self.params, values):
            p.set_value(value.astype("float32"))
        print("LOADED")


class Model():
    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    def get_activation(self, model):
        activation = model["activation"] if "activation" in model else "linear"
        return get_activation(activation)

    def create_layer(self, inputs, model, dnn_type=True):
        if model["model_type"] == "conv":
            conv_type = tf.layers.conv2d

            poolsize = tuple(model["pool"]) if "pool" in model else (1,1)
            stride = tuple(model["stride"]) if "stride" in model else (1,1)

            layer = conv_type(
                inputs=inputs, 
                filters=model["out_size"], 
                kernel_size=model["filter_size"], 
                strides=stride, 
                nonlinearity=self.get_activation(model),
                kernel_initializer=get_init(model, "W"),
                bias_initializer=get_init(model, "b"),
                padding="valid" if "pad" not in model else model["pad"]
            )

        elif model["model_type"] == "mlp":
            layer = tf.layers.dense(
                inputs=inputs, 
                units=model["out_size"],
                activation=self.get_activation(model),
                kernel_initializer=get_init(model, "W"),
                bias_initializer=get_init(model, "b")
            )
    
        elif model["model_type"] == "option":
            layer = MLP3D(model, inputs, nonlinearity=self.get_activation(model))

        else:
            print "UNKNOWN LAYER NAME"
            raise NotImplementedError

        return layer

    def __init__(self, model_in, input_size=None, rng=1234, dnn_type=False):
        """
        example model:
        model = [{"model_type": "conv", "filter_size": [5,5], "pool": [1,1], "stride": [1,1], "out_size": 5},
                 {"model_type": "conv", "filter_size": [7,7], "pool": [1,1], "stride": [1,1], "out_size": 15},
                 {"model_type": "mlp", "out_size": 300, "activation": "tanh"},
                 {"model_type": "mlp", "out_size": 10, "activation": "softmax"}]
        """
        # self.theano_rng = RandomStreams(rng)
        # rng = np.random.RandomState(rng)
        # lasagne.random.set_rng(rng)

        tf.set_random_seed(rng)

        new_layer = tuple(input_size) if isinstance(input_size, list) else input_size
        model = [model_in] if isinstance(model_in, dict) else model_in

        X = tf.placeholder(dtype=tf.float32, shape=[None, 84, 84, 4]) # Input to network
        
        input_tensor = X

        print("Building following model...")
        print(model)

        self.model = model
        self.input_size = input_size
        self.out_size = model_in[-1]["out_size"]
        self.dnn_type = dnn_type

        # Build NN
        for i, m in enumerate(model):
            new_layer = self.create_layer(input_tensor, m, dnn_type=dnn_type)
            input_tensor = new_layer

        print("Build complete.")

        return input_tensor

    # Don't need to do this anymore, can just pass in X as input tensor with feed dict
    # def apply(self, x):
    #     last_layer_inputs = x
    #     for i, m in enumerate(self.model):
    #         if m["model_type"] in ["mlp", "logistic", "advantage"] and last_layer_inputs.ndim > 2:
    #             last_layer_inputs = last_layer_inputs.flatten(2)

    #         last_layer_inputs = self.layers[i].get_output_for(last_layer_inputs)
    #     return last_layer_inputs

    def save_params(self):
        return [i.get_value() for i in self.params]

    def load_params(self, values):
        print("LOADING NNET..")

        for p, value in zip(self.params, values):
            p.set_value(value.astype("float32"))

        print("LOADED")