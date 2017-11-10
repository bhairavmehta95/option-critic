import numpy as np
import tensorflow as tf
import math, csv, time, sys, os, pdb, copy

from build_nn import Network

from baselines.common import set_global_seeds, explained_variance
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import wrap_deepmind
from baselines.a2c.utils import Scheduler, discount_with_dones

class Model():
    def __init__(self, model_template, ob_space, ac_space, nenvs, nsteps, nstack, num_procs,
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
        self.rewards = tf.placeholder(shape=[nbatch], dtype=tf.float32)
        self.lr = tf.placeholder(shape=[], dtype=tf.float32)

        self.step_model = Network(model_template, ob_space, ac_space, nenvs, 1, nstack, reuse=False)
        self.train_model = Network(model_template, ob_space, ac_space, nenvs, nsteps, nstack, reuse=True)

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
        self.termination_loss = tf.reduce_sum(self.terminations * ((self.disconnected_option_q_vals - self.disconnected_value) + self.args.delib) )
        
        # TODO: Look at entropy!
        self.entropy = -1 * tf.reduce_sum(self.action_values*tf.log(self.action_values))

        self.loss = self.policy_loss + self.entropy - self.value_loss - self.termination_loss

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

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, options, actions, rewards):
            feed_dict = {
                m.observations : obs,
                m.actions: actions,
                m.options: options,
                m.rewards: rewards,
            }

            init_ops = [m.grad_norms, m.summary_op]
            _, summary = sess.run(init_ops, feed_dict=feed_dict)

        return summary

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        # self.step = step_model.step
        # self.value = step_model.value
        self.initial_state = step_model.initial_state

        tf.global_variables_initializer().run(session=sess)

class Runner(object):

    def __init__(self, env, model, nsteps=5, nstack=4, gamma=0.99):
        self.env = env
        self.model = model
        nh, nw, nc = env.observation_space.shape
        nenv = env.num_envs
        self.batch_ob_shape = (nenv*nsteps, nh, nw, nc*nstack)
        self.obs = np.zeros((nenv, nh, nw, nc*nstack), dtype=np.uint8)
        self.nc = nc
        obs = env.reset()
        self.update_obs(obs)
        self.gamma = gamma
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    def update_obs(self, obs):
        # Do frame-stacking here instead of the FrameStack wrapper to reduce
        # IPC overhead
        self.obs = np.roll(self.obs, shift=-self.nc, axis=3)
        self.obs[:, :, :, -self.nc:] = obs

    def run(self):
        mb_obs, mb_options, mb_rewards, mb_actions, mb_values, mb_dones = [],[],[],[],[],[]

        for n in range(self.nsteps):
            actions, values = self.model.step(self.obs)

            mb_obs.append(np.copy(self.obs))
            mb_options.append(np.copy(self.options))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)

            obs, rewards, dones, _ = self.env.step(actions)
            self.dones = dones

            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n]*0

            self.update_obs(obs)
            mb_rewards.append(rewards)

        mb_dones.append(self.dones)

        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_options = np.asarray(mb_options, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)

        mb_dones = mb_dones[:, 1:]
        last_values = self.model.value(self.obs).tolist()
        
        #discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards

        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_options = mb_options.flatten()
        mb_values = mb_values.flatten()

        return mb_obs, mb_options, mb_rewards, mb_masks, mb_actions, mb_values

def learn(model_template, env, seed, nsteps=5, nstack=4, total_timesteps=int(80e6), vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, lr=7e-4, lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99, log_interval=100):
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    num_procs = len(env.remotes) # HACK
    model = Model(model_template=model_template, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, nstack=nstack, num_procs=num_procs, ent_coef=ent_coef, vf_coef=vf_coef,
        max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule)
    runner = Runner(env, model, nsteps=nsteps, nstack=nstack, gamma=gamma)

    nbatch = nenvs*nsteps
    tstart = time.time()
    for update in range(1, total_timesteps//nbatch+1):
        obs, options, rewards, masks, actions, values = runner.run()
        policy_loss, value_loss, policy_entropy = model.train(obs, options, rewards, actions)
    env.close()


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