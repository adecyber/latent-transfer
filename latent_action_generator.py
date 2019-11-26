from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
#import latent_network
from tf_agents.networks import network
import tensorflow as tf

nest = tf.contrib.framework.nest
slim = tf.contrib.slim

DIM_Z = 256
DIM_FC_Z = 273
DIM_FC_ACTION = 6
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
                6,
                activation_fn=None,
                normalizer_fn=None,
                scope='actions')
            actions = tf.identity(tf.tanh(actions / 5.0) * 5.0,
                                 'softly_clipped_starts')
    
    #import pdb
    #pdb.set_trace()
    return actions

class ActionGenerator(network.Network):

    def __init__(self,
                 input_tensor_spec,
                 name='ActionGenerator'):
        super(ActionGenerator, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec = (),
            name=name)

        self._dim_z = DIM_Z
        self._dim_fc_z = DIM_FC_Z
        self._dim_fc_action = DIM_FC_ACTION

    def call(self, observations, zs, step_type=None, network_states=()):
        del step_type    # unused.
        
        decoder_input = tf.concat([observations, zs], 1)
        
        return generate_action_single_step(
           decoder_input,
           dim_fc_action=self._dim_fc_action)
        
        """
        with slim.arg_scope(
                [slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=NORMALIZER_FN,
                normalizer_params=NORMALIZER_PARAMS):

            with tf.variable_scope('global_pool'):
                effects = global_pool(
                    effects,
                    axis=1,
                    mask=states['body_mask'],
                    mode='weighted_sum')

            with tf.variable_scope('transform_z'):
                net = effects
                net = slim.fully_connected(
                    net, self._dim_fc_z, scope='fc')
                gaussian_params = slim.fully_connected(
                    net,
                    2 * self._dim_z,
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='gaussian_params')
                means = tf.identity(
                    gaussian_params[..., :self._dim_z], name='means')
                stddevs = tf.add(
                    tf.nn.softplus(gaussian_params[..., self._dim_z:]),
                    1e-6,
                    name='stddevs')
                transformed_zs = tf.identity(means + stddevs * zs,
                                             'transformed_zs')
        """
