from __future__ import absolute_import, division, print_function

import os
from tqdm import tqdm
import base64
import imageio
import IPython
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import gym
import tensorflow as tf 
#import gym_backcheetah

from absl import flags
from absl import app
tf.compat.v1.enable_v2_behavior()

from tf_agents.agents.ddpg import critic_network
import latent_agent
from tf_agents.agents.sac import sac_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
import latent_actor_network
#from tf_agents.networks import actor_distribution_network
from tf_agents.networks import normal_projection_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

FLAGS = flags.FLAGS
flags.DEFINE_string('root_dir', './', 'root dir for output')
flags.DEFINE_string('env_name', 'Hopper-v2', 'Name of a mujoco env')
flags.DEFINE_float('lr', 3e-4, 'learning rate of neural networks')
flags.DEFINE_integer('num_iterations', 1000000, 'number of iterations')
flags.DEFINE_boolean('anneal_lr', False, 'whether or not to anneal learning rate')
flags.DEFINE_float('target_entropy', None, 'target entropy to learn to')

def main(_):
  env_name = FLAGS.env_name
  num_iterations = FLAGS.num_iterations # @param {type:"integer"}
  initial_collect_steps = 1
# initial_collect_steps = 10000 # @param {type:"integer"}
  collect_steps_per_iteration = 1 # @param {type:"integer"}
  replay_buffer_capacity = 1000000 # @param {type:"integer"}
  
  root_dir = FLAGS.root_dir
  batch_size = 256 # @param {type:"integer"}

  critic_learning_rate = FLAGS.lr # @param {type:"number"}
  actor_learning_rate = FLAGS.lr # @param {type:"number"}
  alpha_learning_rate = FLAGS.lr # @param {type:"number"}
  anneal_lr = FLAGS.anneal_lr 
  target_update_tau = 0.005 # @param {type:"number"}
  target_update_period = 1 # @param {type:"number"}
  gamma = 0.99
  reward_scale_factor = 1.0 # @param {type:"number"}
  gradient_clipping = True # @param

  actor_fc_layer_params = (256, 256)
  critic_joint_fc_layer_params = (256, 256)

  log_interval = 5000 # @param {type:"integer"}

 
  num_eval_episodes = 1 # @param {type:"integer"}
 # num_eval_episodes = 30 # @param {type:"integer"}
  eval_interval = 10000 # @param {type:"integer"}
  video_interval = 10000000
  plot_interval = 100000
  train_checkpoint_interval = 10000000
  target_entropy = FLAGS.target_entropy
  
  def embed_mp4(filename):
    """Embeds an mp4 file in the notebook."""
    video = open(filename,'rb').read()
    b64 = base64.b64encode(video)
    tag = '''
    <video width="640" height="480" controls>
      <source src="data:video/mp4;base64,{0}" type="video/mp4">
    Your browser does not support the video tag.
    </video>'''.format(b64.decode())

    return IPython.display.HTML(tag) 

  train_py_env = suite_gym.load(env_name)
  eval_py_env = suite_gym.load(env_name)
  train_env = tf_py_environment.TFPyEnvironment(train_py_env)
  eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

  observation_spec = train_env.observation_spec()
  action_spec = train_env.action_spec()
  critic_net = critic_network.CriticNetwork(
      (observation_spec, action_spec),
      observation_fc_layer_params=None,
      action_fc_layer_params=None,
      joint_fc_layer_params=critic_joint_fc_layer_params)

  def normal_projection_net(action_spec,init_means_output_factor=0.1):
    return normal_projection_network.NormalProjectionNetwork(
        action_spec,
        mean_transform=None,
        state_dependent_std=True,
        init_means_output_factor=init_means_output_factor,
        std_transform=latent_agent.std_clip_transform,
        scale_distribution=True)


  actor_net = latent_actor_network.ActorDistributionNetwork(
      observation_spec,
      action_spec,
      fc_layer_params=actor_fc_layer_params,
      continuous_projection_net=normal_projection_net)
  
  global_step = tf.compat.v1.train.get_or_create_global_step()
  if anneal_lr:
    boundaries = [600000, 1000000]
    values = [3e-4, 9e-5, 5e-5]
    learning_rate = tf.compat.v1.train.piecewise_constant(global_step, boundaries, values)
  tf_agent = latent_agent.SacAgent(
      train_env.time_step_spec(),
      action_spec,
      actor_network=actor_net,
      critic_network=critic_net,
      actor_optimizer=tf.compat.v1.train.AdamOptimizer(
          learning_rate=actor_learning_rate),
      critic_optimizer=tf.compat.v1.train.AdamOptimizer(
          learning_rate=critic_learning_rate),
      alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
          learning_rate=alpha_learning_rate),
      target_update_tau=target_update_tau,
      target_update_period=target_update_period,
      td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error,
      gamma=gamma,
      reward_scale_factor=reward_scale_factor,
      gradient_clipping=gradient_clipping,
      train_step_counter=global_step,
      target_entropy=target_entropy)
  tf_agent.initialize()

  eval_policy = greedy_policy.GreedyPolicy(tf_agent.policy)
  collect_policy = tf_agent.collect_policy

  def compute_avg_return(environment, policy, num_episodes=5):

    total_return = 0.0
    for _ in range(num_episodes):

      time_step = environment.reset()
      episode_return = 0.0

      while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = environment.step(action_step.action)
        episode_return += time_step.reward
      total_return += episode_return
    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


  replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
      data_spec=tf_agent.collect_data_spec,
      batch_size=train_env.batch_size,
      max_length=replay_buffer_capacity)

  initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
          train_env,
          collect_policy,
          observers=[replay_buffer.add_batch],
          num_steps=initial_collect_steps)
  print("Running initial collection policy...")
  initial_collect_driver.run()

  # Dataset generates trajectories with shape [Bx2x...]
  dataset = replay_buffer.as_dataset(
      num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).prefetch(3)

  iterator = iter(dataset)

  collect_driver = dynamic_step_driver.DynamicStepDriver(
      train_env,
      collect_policy,
      observers=[replay_buffer.add_batch],
      num_steps=collect_steps_per_iteration)

  # #@test {"skip": true}
  # %%time

  # (Optional) Optimize by wrapping some of the code in a graph using TF function.
  tf_agent.train = common.function(tf_agent.train)
  collect_driver.run = common.function(collect_driver.run)

  # Reset the train step
  tf_agent.train_step_counter.assign(0)
  
  # Set up checkpointing of tf agents model
  train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric(),
    ]
  train_dir = os.path.join(root_dir, 'train_ckpt')
  train_checkpointer = common.Checkpointer(
        ckpt_dir=train_dir,
        agent=tf_agent,
        global_step=global_step,
        metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
  train_checkpointer.initialize_or_restore()

  # Evaluate the agent's policy once before training.
  print("Initial policy evaluation...")
  avg_return = compute_avg_return(eval_env, eval_policy, num_eval_episodes)
  returns = [avg_return]
  
  returnsCache = []
  print("Training...")
  for _ in tqdm(range(num_iterations)):
    # Collect a few steps using collect_policy and save to the replay buffer.
    for _ in range(collect_steps_per_iteration):
      collect_driver.run()

    # Sample a batch of data from the buffer and update the agent's network.
    experience, unused_info = next(iterator)
    train_loss = tf_agent.train(experience)

    step = tf_agent.train_step_counter.numpy()
    if step % log_interval == 0:
      print('step = {0}: loss = {1}'.format(step, train_loss.loss))

    if step % eval_interval == 0:
      avg_return = compute_avg_return(eval_env, eval_policy, num_eval_episodes)
      print('step = {0}: Average Return = {1}'.format(step, avg_return))
      returnsCache.append((step, avg_return))

    if step % train_checkpoint_interval == 0:
      print("Saving model checkpoint...")
      train_checkpointer.save(global_step=global_step)

    if step % video_interval == 0:
      print('Creating video at {}k iterations...'.format(int(step/1000)))

      num_episodes = 3
      video_filename = FLAGS.root_dir + '/videos/sac' + FLAGS.env_name[:-3] + str(int(step/1000)) + 'k.mp4'
      
      with imageio.get_writer(video_filename, fps=60) as video:
        for _ in range(num_episodes):
          time_step = eval_env.reset()
          video.append_data(eval_py_env.render())
          while not time_step.is_last():
            action_step = tf_agent.policy.action(time_step)
            time_step = eval_env.step(action_step.action)
            video.append_data(eval_py_env.render())
    
      embed_mp4(video_filename)
  
    if step % plot_interval == 0:
      print("Plotting returns...") 
      steps, returns = zip(*returnsCache)
      # steps = [x - 1100000 for x in steps]
      plt.plot(steps, returns)
      plt.ylabel('Average Return')
      plt.xlabel('Step')
      plt.ylim()
      plt.savefig(FLAGS.root_dir + '/plots/sac' + FLAGS.env_name[:-3] + str(int(step/1000)) + 'k.png')
      print("Done plotting...")

if __name__ == '__main__':
  print("Running SAC...")
app.run(main)
