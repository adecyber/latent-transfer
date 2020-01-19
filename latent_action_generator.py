from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf
import latent_network_robovat
from tf_agents.specs import tensor_spec
from tf_agents.networks import categorical_projection_network
from tf_agents.networks import normal_projection_network
from tf_agents.utils import nest_utils
import numpy as np


nest = tf.contrib.framework.nest
slim = tf.contrib.slim

DIM_Z = 16
DIM_FC_ACTION = 6
DIM_FC_Z = DIM_Z + DIM_FC_ACTION
NORMALIZER_FN = None
NORMALIZER_PARAMS = None

def generate_action_single_step(inputs, dim_fc_action):
    with slim.arg_scope(
            [slim.fully_connected],
            activation_fn=tf.nn.relu,
            normalizer_fn=NORMALIZER_FN,
            normalizer_params=NORMALIZER_PARAMS):

        with tf.variable_scope('actions'):
            net = inputs
            net = slim.fully_connected(
                net, dim_fc_action, scope='fc')
            actions = slim.fully_connected(
                net,
                dim_fc_action,
                activation_fn=None,
                normalizer_fn=None,
                scope='output_actions')
            actions = tf.identity(tf.tanh(actions / 5.0) * 5.0,
                                 'softly_clipped_starts')
    
    return actions

def _categorical_projection_net(action_spec, logits_init_output_factor=0.1):
  return categorical_projection_network.CategoricalProjectionNetwork(
      action_spec, logits_init_output_factor=logits_init_output_factor)


def _normal_projection_net(action_spec,
                           init_action_stddev=0.35,
                           init_means_output_factor=0.1):
  std_bias_initializer_value = np.log(np.exp(init_action_stddev) - 1)

  return normal_projection_network.NormalProjectionNetwork(
      action_spec,
      init_means_output_factor=init_means_output_factor,
      std_bias_initializer_value=std_bias_initializer_value,
      scale_distribution=False)

class ActionGenerator(latent_network_robovat.Network):

    def __init__(self,
                 input_tensor_spec,
                 output_tensor_spec,
                 name='ActionGenerator'):
        super(ActionGenerator, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec = (),
            name=name)
        
        print("Initializing action generator")
        self._dim_z = DIM_Z
        self._dim_fc_z = DIM_FC_Z
        self._dim_fc_action = DIM_FC_ACTION
        self._output_tensor_spec = output_tensor_spec

        discrete_projection_net=_categorical_projection_net
        continuous_projection_net=_normal_projection_net

        def map_proj(spec):
          if tensor_spec.is_discrete(spec):
            return discrete_projection_net(spec)
          else:
            return continuous_projection_net(spec)
      
        projection_networks = tf.nest.map_structure(map_proj, self._output_tensor_spec)

        self._input_tensor_spec = input_tensor_spec
        self._projection_networks = projection_networks

    def call(self, input_tensor, step_type=None, network_states=()):
        del step_type    # unused.
        print("Running action generator")
        observations = input_tensor[0]
        zs = input_tensor[1]
        decoder_input = tf.concat([observations, zs], 1)
        outer_rank = nest_utils.get_outer_rank(observations, self.input_tensor_spec[0])
        state = generate_action_single_step(
           decoder_input,
           dim_fc_action=self._dim_fc_action)
        state = tf.dtypes.cast(state, dtype=tf.float32)
        output_actions = tf.nest.map_structure(
            lambda proj_net: proj_net(state, outer_rank)[0], self._projection_networks)
        return output_actions
