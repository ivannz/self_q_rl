# -*- coding: utf-8 -*-

"""
Preliminary implementation of batch normalization for Lasagne.
Does not include a way to properly compute the normalization factors over the
full training set for testing, but can be used as a drop-in for training and
validation.

Author: Jan Schlüter
"""

import numpy as np
import lasagne
import theano
import theano.tensor as T

class BatchNormLayer(lasagne.layers.Layer):

    def __init__(self, incoming, axes=None, epsilon=0.01, alpha=0.5,
            nonlinearity=None, **kwargs):
        """
        Instantiates a layer performing batch normalization of its inputs,
        following Ioffe et al. (http://arxiv.org/abs/1502.03167).
        
        @param incoming: `Layer` instance or expected input shape
        @param axes: int or tuple of int denoting the axes to normalize over;
            defaults to all axes except for the second if omitted (this will
            do the correct thing for dense layers and convolutional layers)
        @param epsilon: small constant added to the standard deviation before
            dividing by it, to avoid numeric problems
        @param alpha: coefficient for the exponential moving average of
            batch-wise means and standard deviations computed during training;
            the larger, the more it will depend on the last batches seen
        @param nonlinearity: nonlinearity to apply to the output (optional)
        """
        super(BatchNormLayer, self).__init__(incoming, **kwargs)
        if axes is None:
## default: normalize over all but the second axis. The input shape is in the
##  case of an imape gropresscing net usually B x K x M x N, where B is the batch
##  size, K -- the number of features, N x M -- the spatial dimension of each
##  feature. Thus, by default, the features are aggregated across batches,
##  and space, but not over distinct features.
            axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(axes, int):
            axes = (axes,)
        self.axes = axes
## Set variance regularization and weight for EWMA mean estimate.
        self.epsilon = epsilon
        self.alpha = alpha
## By default this layer has linear activation.
        if nonlinearity is None:
            nonlinearity = lasagne.nonlinearities.identity
        self.nonlinearity = nonlinearity
## Initialize the shape 
        shape = list(self.input_shape)
        broadcast = [False] * len(shape)
## Over all axes that participate in normalization, permit broadcasting.
        for axis in self.axes:
            shape[axis] = 1
            broadcast[axis] = True
        if any(size is None for size in shape):
            raise ValueError("BatchNormLayer needs specified input sizes for "
                             "all dimensions/axes not normalized over.")
## Initialize the paramters
        dtype = theano.config.floatX
## Accumulated mean and standard deviation of the normalisation interlayer
        self.mean = T.addbroadcast(
                        self.add_param(lasagne.init.Constant(0), shape, 'mean',
                                       trainable=False, regularizable=False),
                        *self.axes )
        self.std = T.addbroadcast(
                        self.add_param(lasagne.init.Constant(1), shape, 'std',
                                       trainable=False, regularizable=False),
                        *self.axes )
## The linear transform of the normalization layer
        self.beta = T.addbroadcast(
                        self.add_param(lasagne.init.Constant(0), shape, 'beta',
                                       trainable=True, regularizable=True),
                        *self.axes )
        self.gamma = T.addbroadcast(
                        self.add_param(lasagne.init.Constant(1), shape, 'gamma',
                                       trainable=True, regularizable=False),
                        *self.axes )

    def get_output_for(self, input, deterministic=False, **kwargs):
        if deterministic:
            # use stored mean and std
            mean = self.mean
            std = self.std
        else:
            # use this batch's mean and std
            mean = T.addbroadcast(T.mean(input, self.axes, keepdims=True), *self.axes)
            std = T.addbroadcast(T.sqrt(T.var(input, self.axes, keepdims=True) + self.epsilon), *self.axes)
            # and update the stored mean and std:
            # we create (memory-aliased) clones of the stored mean and std
            # set a default update for them
            self.mean.default_update = ( ( 1 - self.alpha ) * self.mean + self.alpha * mean )
            self.std.default_update = ( ( 1 - self.alpha ) * self.std + self.alpha * std )
            # and include them in the graph so their default updates will be
            # applied (although the expressions will be optimized away later,
            #   they are still going to be updated)
            mean += 0 * self.mean
            std += 0 * self.std
        input_norm = (input - mean) / std
        normalized = input_norm * self.gamma + self.beta
        return self.nonlinearity(normalized)

def batch_norm(layer):
    """
    Convenience function to apply batch normalization to a given layer's output.
    Will steal the layer's nonlinearity if there is one (effectively introducing
    the normalization right before the nonlinearity), and will remove the
    layer's bias if there is one (because it would be redundant).

    @param layer: The `Layer` instance to apply the normalization to; note that
        it will be irreversibly modified as specified above
    @return: A `BatchNormLayer` instance stacked on the given `layer`
    """
## Clone nonlinearity in the source layer's activation, and force its to linear passthough
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = lasagne.nonlinearities.identity
## Batch Norm layer already provides a linear coefficient mirroring the bias
##  in the source layer.
    if hasattr(layer, 'b'):
        del layer.params[layer.b]
        layer.b = None
    return BatchNormLayer(layer, nonlinearity=nonlinearity)