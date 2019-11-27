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

class ZInferenceNetwork(network.Network):

    def __init__(self,
                 input_tensor_spec,
                 name='ZInferenceNetwork'):
        super(ZInferenceNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name)

        self._dim_z = DIM_Z
        self._dim_fc_z = DIM_FC_Z
        self._dim_fc_action = DIM_FC_ACTION

    def call(self, observations, actions, step_type=None, network_states=()):
        del step_type    # unused.


        with slim.arg_scope(
                [slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=NORMALIZER_FN,
                normalizer_params=NORMALIZER_PARAMS):

            with tf.variable_scope('inference'):
                net = tf.concat([observations, actions], axis=-1)
                net = slim.fully_connected(
                    net, self._dim_fc_z, scope='fc')
                gaussian_params = slim.fully_connected(
                    net,
                    2 * self._dim_z,
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='gaussian_params')
                z_means = tf.identity(
                    gaussian_params[..., :self._dim_z], name='z_means')
                z_stddevs = tf.add(
                    tf.nn.softplus(gaussian_params[..., self._dim_z:]),
                    1e-6,
                    name='z_stddevs')

        return z_means, z_stddevs
