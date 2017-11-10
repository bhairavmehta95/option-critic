import numpy as np
import tensorflow as tf
import math, csv, time, sys, os, pdb, copy

from build_nn import Network

from baselines.a2c.utils import Scheduler, discount_with_dones

class Model():
    def __init__(self, policy, ob_space, ac_space, nenvs, nsteps, nstack, num_procs,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear'):
        """
        example model:
        model = [{"model_type": "conv", "filter_size": [5,5], "pool": [1,1], "stride": [1,1], "out_size": 5},
                 {"model_type": "conv", "filter_size": [7,7], "pool": [1,1], "stride": [1,1], "out_size": 15},
                 {"model_type": "mlp", "out_size": 300, "activation": "tanh"},
                 {"model_type": "mlp", "out_size": 10, "activation": "softmax"}]
        """
        config = tf.ConfigProto(allow_soft_placement=True,
            intra_op_parallelism_threads=num_procs,
            inter_op_parallelism_threads=num_procs)

        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        nact = ac_space.n
        nbatch = nenvs*nsteps

        print("Building worker specific operations.")

        self.actions = tf.placeholder(shape=[nbatch], dtype=tf.int32)
        self.options = tf.placeholder(shape=[nbatch], dtype=tf.int32)
        self.raw_rewards = tf.placeholder(shape=[nbatch], dtype=tf.float32)
        self.lr = tf.placeholder(shape=[], dtype=tf.float32)

        self.step_model = policy(sess, ob_space, ac_space, nenvs, 1, nstack, reuse=False)
        self.train_model = policy(sess, ob_space, ac_space, nenvs, nsteps, nstack, reuse=True)

        self.responsible_options = tf.stack([self.batch_size, self.options], axis=1)
        self.responsible_actions = tf.stack([self.batch_size, self.actions], axis=1)

        self.disconnected_q_vals = tf.stop_gradient(self.q_values_options)
        self.option_q_vals = tf.gather_nd(params=self.q_values_options, indices=self.responsible_options) # Extract q values for each option
        self.disconnected_option_q_vals = tf.gather_nd(params=self.disconnected_q_vals, indices=self.responsible_options) # Extract q values for each option
        self.terminations = tf.gather_nd(params=self.termination_fn, indices=self.responsible_options)

        self.action_values = tf.gather_nd(params=self.intra_options_q_vals, indices=self.responsible_options)
        self.action_values = tf.gather_nd(params=self.action_values, indices=self.responsible_actions)

        self.value = tf.reduce_max(self.q_values_options) * (1 - self.args.option_eps) + (self.args.option_eps * tf.reduce_mean(self.q_values_options))
        self.disconnected_value = tf.stop_gradient(self.value)

        # Losses
        self.value_loss = 0.5 * tf.reduce_sum(self.args.critic_coef * tf.square(self.targets - tf.reshape(self.value_fn, [-1])))
        self.policy_loss = -1 * tf.reduce_sum(tf.log(self.action_values)*(self.raw_rewards - self.disconnected_option_q_vals))
        self.termination_gradient = tf.reduce_sum(self.terminations * ((self.disconnected_option_q_vals - self.disconnected_value) + self.args.delib) )
        self.entropy = -1 * tf.reduce_sum(self.action_values*tf.log(self.action_values))

        self.loss = self.policy_loss + self.entropy - self.value_loss - self.termination_gradient

        # Gradients
        self.vars = tf.get_collection('model')        
        self.gradients = tf.gradients(self.loss, self.vars)
        grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, max_grad_norm)
        grads = list(zip(grads, self.vars))
        self.trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        self.apply_grads = self.trainer.apply_gradients(grads)

        # Summary
        self.summary.append(tf.summary.scalar('policy_loss', self.policy_loss))
        self.summary.append(tf.summary.scalar('value_loss', self.value_loss))
        self.summary.append(tf.summary.scalar('termination_gradient', self.termination_gradient))
        self.summary.append(tf.summary.scalar('entropy', self.entropy))
        self.summary_op = tf.summary.merge(self.summary)


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
            raw_rewards = np.random.random(size=4)
            targets = np.random.random(size=4)

            feed_dict = {
                m.observations : obs,
                m.actions: actions,
                m.options: options,
                m.raw_rewards: raw_rewards,
                m.targets: targets
            }

            init_ops = [m.grad_norms, m.summary_op]
            _, summary = sess.run(init_ops, feed_dict=feed_dict)
            writer.add_summary(summary)