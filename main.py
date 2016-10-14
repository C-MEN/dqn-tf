import random
import tensorflow as tf

from dqn.agent import Agent
from dqn.environment import GymEnvironment
from config import get_config

flags = tf.app.flags

# Model
flags.DEFINE_string('model', 'm1', 'Type of model')
flags.DEFINE_boolean('dueling', False, 'Whether to use dueling deep q-network')
flags.DEFINE_boolean('double_q', False, 'Whether to use double q-learning')

# Environment
flags.DEFINE_string('env_name', 'Breakout-v0', 'The name of gym environment to use')
flags.DEFINE_integer('action_repeat', 4, 'The number of action to be repeated')

# Etc
flags.DEFINE_boolean('use_gpu', True, 'Whether to use gpu or not')
flags.DEFINE_string('gpu_fraction', '1/1', 'fraction of GPU memory to allocate')
flags.DEFINE_boolean('display', False, 'Whether to do display the game screen or not')
flags.DEFINE_boolean('train', True, 'Whether to do training or testing')
flags.DEFINE_integer('random_seed', 13, 'Value of random seed')

FLAGS = flags.FLAGS

# Set random seed
tf.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)


def calc_gpu_fraction(fraction_string):
    num, denom = fraction_string.split('/')
    num, denom = float(num), float(denom)

    fraction = num / denom
    print(" [*] GPU : %.4f" % fraction)
    return fraction


def main(_):
    # Trying to request all the GPU memory will fail, since the system
    # always allocates a little memory on each GPU for itself. Only set
    # up a GPU configuration if fractional amount of memory is requested.
    tf_config = None
    gpu_fraction = calc_gpu_fraction(FLAGS.gpu_fraction)
    if gpu_fraction < 1:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
        tf_config = tf.ConfigProto(gpu_options=gpu_options)

    with tf.Session(config=tf_config) as sess:
        config = get_config(FLAGS) or FLAGS
        env = GymEnvironment(config)

        # Change data format for running on a CPU.
        if not FLAGS.use_gpu:
            config.cnn_format = 'NHWC'

        agent = Agent(config, env, sess)

        if FLAGS.train:
            agent.train()
        else:
            agent.play()


if __name__ == '__main__':
    tf.app.run()
