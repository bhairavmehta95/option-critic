from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common import set_global_seeds

import argparse
from model import learn

def train(env_id, num_timesteps, seed, lrschedule, num_cpu):
    def make_env(rank):
        def _thunk():
            env = make_atari(env_id)
            env.seed(seed + rank)
            # env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            # gym.logger.setLevel(logging.WARN)
            return wrap_deepmind(env)
        return _thunk
        
    set_global_seeds(seed)
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--vf_coef',        help='critic coefficient',      default=0.5)
    parser.add_argument('--ent_coef',       help='entropy coefficient',     default=0.01)
    parser.add_argument('--opt_eps',        help='option eps',              default=0.01)
    parser.add_argument('--delib_cost',     help='deliberation cost',       default=0.001)
    parser.add_argument('--max_grad_norm',  help='max gradient norm',       default=0.5)
    parser.add_argument('--lrschedule',     help='learning rate schedule',  default='linear')
    parser.add_argument('--epsilon',        help='epsilon for exploration', default=1e-5)
    parser.add_argument('--alpha',          help='alpha',                   default=0.99)
    parser.add_argument('--gamma',          help='gamma (discounting)',     default=0.99)
    parser.add_argument('--log_interval',   help='log_interval',            default=100)
    parser.add_argument('--lr',             help='learning rate',           default=0.001)
    parser.add_argument('--nopts',          help='number of options' ,      default=4)

    args = parser.parse_args()

    model_template = [
        {"model_type": "conv", "filter_size": [8,8], "pool": [1,1], "stride": [4,4], "out_size": 32, "name": "conv1"},
        {"model_type": "conv", "filter_size": [4,4], "pool": [1,1], "stride": [2,2], "out_size": 64, "name": "conv2"},
        {"model_type": "conv", "filter_size": [3,3], "pool": [1,1], "stride": [1,1], "out_size": 64, "name": "conv3"},
        {"model_type": "flatten"},
        {"model_type": "mlp", "out_size": 512, "activation": "relu", "name": "fc1"},
        {"model_type": "option"},
        {"model_type": "value"}
    ]

    learn(model_template, env, seed, total_timesteps=int(num_timesteps * 1.1), args=args)
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))

    args = parser.parse_args()

    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
        lrschedule=args.lrschedule, num_cpu=16)

if __name__ == '__main__':
    main()