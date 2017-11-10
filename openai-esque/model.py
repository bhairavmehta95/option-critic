import numpy as np
import tensorflow as tf
import math, csv, time, sys, os, pdb, copy

from build_nn import Network

from baselines.common import set_global_seeds, explained_variance
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import wrap_deepmind
from baselines.a2c.utils import Scheduler, discount_with_dones

class Model():
    def __init__(self, model_template, num_options, ob_space, ac_space, nenvs, nsteps, nstack, num_procs,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4, alpha=0.99, epsilon=1e-5, 
            total_timesteps=int(80e6), lrschedule='linear', option_eps=0.001, delib_cost=0.001):
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
        nopt = num_options

        batch_indexer = tf.range(nbatch)

        print("Building rest of the graph.")

        self.actions = tf.placeholder(shape=[nbatch], dtype=tf.int32)
        self.options = tf.placeholder(shape=[nbatch], dtype=tf.int32)
        self.rewards = tf.placeholder(shape=[nbatch], dtype=tf.float32)
        self.lr = tf.placeholder(shape=[], dtype=tf.float32)

        self.summary = []

        self.step_model = Network(model_template, nopt, ob_space, ac_space, nenvs, 1, nstack, reuse=False)
        self.train_model = Network(model_template, nopt, ob_space, ac_space, nenvs, nsteps, nstack, reuse=True)

        self.responsible_options = tf.stack([batch_indexer, self.options], axis=1)
        self.responsible_actions = tf.stack([batch_indexer, self.actions], axis=1)

        self.disconnected_q_vals = tf.stop_gradient(self.train_model.q_values_options)
        self.option_q_vals = tf.gather_nd(params=self.train_model.q_values_options, indices=self.responsible_options) # Extract q values for each option

        self.disconnected_option_q_vals = tf.gather_nd(params=self.disconnected_q_vals, indices=self.responsible_options) # Extract q values for each option
        self.terminations = tf.gather_nd(params=self.train_model.termination_fn, indices=self.responsible_options)

        self.action_values = tf.gather_nd(params=self.train_model.intra_options_q_vals, indices=self.responsible_options)
        self.action_values = tf.gather_nd(params=self.action_values, indices=self.responsible_actions)

        self.value = tf.reduce_max(self.train_model.q_values_options) * (1 - option_eps) + (option_eps * tf.reduce_mean(self.train_model.q_values_options))
        self.disconnected_value = tf.stop_gradient(self.value)

        # Losses
        self.value_loss = 0.5 * tf.reduce_sum(vf_coef * tf.square(self.rewards - tf.reshape(self.train_model.value_fn, [-1])))
        self.policy_loss = -1 * tf.reduce_sum(tf.log(self.action_values)*(self.rewards - self.disconnected_option_q_vals))
        self.termination_loss = tf.reduce_sum(self.terminations * ((self.disconnected_option_q_vals - self.disconnected_value) + delib_cost) )
        
        # TODO: Look at entropy!
        self.entropy = -1 * tf.reduce_sum(self.action_values*tf.log(self.action_values))

        self.loss = self.policy_loss + self.entropy - self.value_loss - self.termination_loss

        # Gradients
        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'model')        
        self.gradients = tf.gradients(self.loss, self.vars)
        grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, max_grad_norm)
        grads = list(zip(grads, self.vars))
        self.trainer = tf.train.RMSPropOptimizer(learning_rate=lr, decay=alpha, epsilon=epsilon)
        self.apply_grads = self.trainer.apply_gradients(grads)

        # Summary
        self.summary.append(tf.summary.scalar('policy_loss', self.policy_loss))
        self.summary.append(tf.summary.scalar('value_loss', self.value_loss))
        self.summary.append(tf.summary.scalar('termination_loss', self.termination_loss))
        self.summary.append(tf.summary.scalar('entropy', self.entropy))
        self.summary_op = tf.summary.merge(self.summary)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, options, actions, rewards):
            print(sess.run(tf.report_uninitialized_variables()))

            feed_dict = {
                self.train_model.observations : obs,
                self.actions: actions,
                self.options: options,
                self.rewards: rewards,
            }

            train_ops = [self.grad_norms, self.summary_op]
            _, summary = sess.run(train_ops, feed_dict=feed_dict)

            return summary

        def setup_tensorflow(sess, writer):
            self.step_model.setup_tensorflow(sess, writer)
            self.train_model.setup_tensorflow(sess, writer)            

        self.train = train
        self.setup_tensorflow = setup_tensorflow

        self.initial_state = self.step_model.initial_state
        self.step = self.step_model.step
        self.value = self.step_model.value

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

        self.options = np.zeros(nenv) # current options; TODO

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
                # TODO: why isn't bootstrapping working?
                rewards = discount_with_dones(rewards + [0.0], dones+[0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)

            mb_rewards[n] = rewards

        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_options = mb_options.flatten()
        mb_values = mb_values.flatten()

        return mb_obs, mb_options, mb_rewards, mb_actions, mb_values

def learn(model_template, env, seed, nsteps=5, nstack=4, total_timesteps=int(80e6), args=None):   
    vf_coef=args.vf_coef
    ent_coef= args.ent_coef
    max_grad_norm= args.max_grad_norm
    lr= args.lr
    lrschedule= args.lrschedule
    epsilon= args.epsilon
    alpha= args.alpha
    gamma= args.gamma
    log_interval= args.log_interval
    delib_cost =args.delib_cost

    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    num_options = args.nopts
    option_eps = args.opt_eps

    num_procs = len(env.remotes) # HACK
    model = Model(model_template=model_template, num_options=num_options, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, nstack=nstack, num_procs=num_procs, ent_coef=ent_coef, vf_coef=vf_coef,
        max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule, option_eps=option_eps, delib_cost=delib_cost)
    runner = Runner(env, model, nsteps=nsteps, nstack=nstack, gamma=gamma)

    nbatch = nenvs*nsteps
    tstart = time.time()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('log', sess.graph)
        model.setup_tensorflow(sess=sess, writer=writer)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        for update in range(1, total_timesteps//nbatch+1):
            obs, options, rewards, actions, values = runner.run()
            policy_loss, value_loss, policy_entropy = model.train(obs, options, rewards, actions)

    env.close()
