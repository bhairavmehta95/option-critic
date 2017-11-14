import numpy as np


class Runner(object):
    def __init__(self, env, model, nsteps=20, nstack=4, gamma=0.99, option_eps=0.001, delib_cost=0.020):
        self.env = env
        self.model = model
        nh, nw, nc = env.observation_space.shape
        nenv = env.num_envs
        self.batch_ob_shape = (nenv*nsteps, nh, nw, nc*nstack)
        self.obs = np.zeros((nenv, nh, nw, nc*nstack), dtype=np.uint8)
        self.nc = nc
        self.option_eps = option_eps
        self.delib_cost = delib_cost

        self.options = np.zeros(nenv)

        obs = env.reset()
        self.update_obs(obs)
        self.options = self.model.initialize_options(self.obs)

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
        mb_obs, mb_options, mb_rewards, mb_actions, mb_values, mb_dones, mb_costs = [],[],[],[],[],[],[]

        for n in range(self.nsteps):
            actions, values = self.model.step(self.obs, self.options)

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

            # Update current option
            self.options, costs = self.model.update_options(self.obs, self.options, self.option_eps, self.delib_cost)
            mb_costs.append(costs)

            mb_rewards.append(rewards)

        mb_dones.append(self.dones)

        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_options = np.asarray(mb_options, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_costs = np.asarray(mb_costs, dtype=np.float32).swapaxes(1, 0)

        mb_dones = mb_dones[:, 1:]
        last_values = self.model.value(self.obs).tolist()

        #discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()

            if dones[-1] == 0:
                rewards = discount_with_dones(rewards + [value[0]], dones+[0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)

            mb_rewards[n] = rewards

        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_options = mb_options.flatten()
        mb_values = mb_values.flatten()
        mb_costs = mb_costs.flatten()

        return mb_obs, mb_options, mb_rewards, mb_actions, mb_values, mb_costs
