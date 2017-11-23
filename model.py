import numpy as np
import tensorflow as tf

from build_nn import Network
from runner import Runner

from baselines.common import set_global_seeds
from baselines.a2c.utils import Scheduler

class Model():
    def __init__(self, model_template, num_options, ob_space, ac_space, nenvs, nsteps, nstack, num_procs,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4, alpha=0.99, epsilon=1e-5,
            total_timesteps=int(80e6), lrschedule='linear', option_eps=0.001, delib_cost=0.001):

        config = tf.ConfigProto(allow_soft_placement=True,
            intra_op_parallelism_threads=num_procs,
            inter_op_parallelism_threads=num_procs)

        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        self.sess = sess

        self.rng = np.random.RandomState(0) # TODO

        nact = ac_space.n
        nbatch = nenvs*nsteps
        nopt = num_options

        self.option_eps = option_eps
        self.action_eps = epsilon

        batch_indexer = tf.range(nbatch)

        print("Building rest of the graph.")

        self.actions = tf.placeholder(shape=[nbatch], dtype=tf.int32)
        self.options = tf.placeholder(shape=[nbatch], dtype=tf.int32)
        self.rewards = tf.placeholder(shape=[nbatch], dtype=tf.float32)
        self.deliberation_costs = tf.placeholder(shape=[nbatch], dtype=tf.float32)
        self.lr = tf.placeholder(shape=[], dtype=tf.float32)

        summary = []

        # Networks
        self.step_model = Network(model_template, nopt, ob_space, ac_space, nenvs, 1, nstack, reuse=False)
        self.train_model = Network(model_template, nopt, ob_space, ac_space, nenvs, nsteps, nstack, reuse=True)

        # Indexers
        self.responsible_options = tf.stack([batch_indexer, self.options], axis=1)
        self.responsible_actions = tf.stack([batch_indexer, self.actions], axis=1)
        self.network_indexer = tf.stack([self.options, batch_indexer], axis=1)

        # Q Values OVER options
        self.disconnected_q_vals = tf.stop_gradient(self.train_model.q_values_options)

        # Q values of each option that was taken
        self.responsible_opt_q_vals = tf.gather_nd(params=self.train_model.q_values_options, indices=self.responsible_options) # Extract q values for each option
        self.disconnected_q_vals_option = tf.gather_nd(params=self.disconnected_q_vals, indices=self.responsible_options)

        # Termination probability of each option that was taken
        self.terminations = tf.gather_nd(params=self.train_model.termination_fn, indices=self.responsible_options)

        # Q values for each action that was taken
        relevant_networks = tf.gather_nd(params=self.train_model.intra_option_policies, indices=self.network_indexer)
        relevant_networks = tf.nn.softmax(relevant_networks, dim=1)
        
        self.action_values = tf.gather_nd(params=relevant_networks, indices=self.responsible_actions)

        # Weighted average value
        self.value = tf.reduce_max(self.train_model.q_values_options) * (1 - option_eps) + (option_eps * tf.reduce_mean(self.train_model.q_values_options))
        disconnected_value = tf.stop_gradient(self.value)

        # Losses; TODO: Why reduce sum vs reduce mean?
        self.value_loss = vf_coef * tf.reduce_mean(vf_coef * tf.square(self.rewards - self.responsible_opt_q_vals))
        self.policy_loss = tf.reduce_mean(tf.log(self.action_values) * (self.rewards - self.disconnected_q_vals_option))
        self.termination_loss = tf.reduce_mean(self.terminations * ((self.disconnected_q_vals_option - disconnected_value) + self.deliberation_costs) )

        action_probabilities = tf.nn.softmax(self.train_model.intra_option_policies, dim=1)
        self.entropy = ent_coef * tf.reduce_mean(action_probabilities * tf.log(action_probabilities))

        self.loss = -self.policy_loss - self.entropy - self.value_loss - self.termination_loss

        # Gradients
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'model')
        gradients = tf.gradients(self.loss, train_vars)
        grads, grad_norms = tf.clip_by_global_norm(gradients, max_grad_norm)
        grads = list(zip(grads, train_vars))
        trainer = tf.train.RMSPropOptimizer(learning_rate=lr, decay=alpha, epsilon=epsilon)
        self.apply_grads = trainer.apply_gradients(grads)

        # Summary
        avg_reward = tf.reduce_mean(self.rewards)

        summary.append(tf.summary.scalar('policy_loss', self.policy_loss))
        summary.append(tf.summary.scalar('value_loss', self.value_loss))
        summary.append(tf.summary.scalar('termination_loss', self.termination_loss))
        summary.append(tf.summary.scalar('entropy', self.entropy))
        summary.append(tf.summary.scalar('avg_reward', avg_reward))
        self.summary_op = tf.summary.merge(summary)

        self.print_op = [relevant_networks[0], self.action_values, self.policy_loss, self.value_loss, self.termination_loss, avg_reward]

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, options, actions, rewards, costs):
            feed_dict = {
                self.train_model.observations : obs,
                self.actions: actions,
                self.options: options,
                self.rewards: rewards,
                self.deliberation_costs: costs
            }

            train_ops = [self.apply_grads, self.summary_op, self.print_op]
            _, summary, summary_str = sess.run(train_ops, feed_dict=feed_dict)

            print(summary_str)

            return summary


        def setup_tensorflow(sess, writer):
            self.step_model.setup_tensorflow(sess, writer)
            self.train_model.setup_tensorflow(sess, writer)

        self.train = train
        self.setup_tensorflow = setup_tensorflow

        self.initial_state = self.step_model.initial_state
        self.step = self.step_model.step
        self.value = self.step_model.value
        self.update_options = self.step_model.update_options

        tf.global_variables_initializer().run(session=sess)


    def initialize_options(self, observations):
        q_values_options = self.sess.run(self.step_model.q_values_options, {self.step_model.observations: observations})
        options = [np.argmax(q_vals) \
            if self.rng.rand() > self.option_eps \
            else self.rng.randint(self.num_options) for q_vals in q_values_options
        ]

        return options


def learn(model_template, env, seed, nsteps=5, nstack=4, total_timesteps=int(80e6), args=None):
    vf_coef = args.vf_coef
    ent_coef = args.ent_coef
    max_grad_norm = args.max_grad_norm
    lr = args.lr
    lrschedule = args.lrschedule
    epsilon = args.epsilon
    alpha = args.alpha
    gamma = args.gamma
    log_interval = args.log_interval
    delib_cost = args.delib_cost
    log_dir = args.log_dir

    num_options = args.nopts
    option_eps = args.opt_eps

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space

    tf.reset_default_graph()
    set_global_seeds(seed)

    num_procs = len(env.remotes) # HACK
    model = Model(model_template=model_template, num_options=num_options, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, nstack=nstack, num_procs=num_procs, ent_coef=ent_coef, vf_coef=vf_coef,
        max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule, option_eps=option_eps, delib_cost=delib_cost)
    runner = Runner(env, model, nsteps=nsteps, nstack=nstack, gamma=gamma, option_eps=option_eps, delib_cost=delib_cost)

    nbatch = nenvs*nsteps
    # tstart = time.time()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(log_dir, sess.graph)
        model.setup_tensorflow(sess=sess, writer=writer)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        for update in range(1, total_timesteps//nbatch+1):
            obs, options, rewards, actions, values, costs = runner.run()
            summary = model.train(obs, options, actions, rewards, costs)

            writer.add_summary(summary, global_step=update)

    env.close()
