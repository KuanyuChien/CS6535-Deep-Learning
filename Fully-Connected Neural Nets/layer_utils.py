pass
from cs6353.layers import *

def affine_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

# helper methods for batch normalization
def affine_batchnorm_relu_forward(x, w, b, gamma, beta, bn_param):
    """
    Convenience layer that performs an affine transform followed by batch 
    normalization and then ReLU.
    """
    # Affine forward
    a, fc_cache = affine_forward(x, w, b)
    
    # Batch normalization forward
    bn_out, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
    
    # ReLU forward
    out, relu_cache = relu_forward(bn_out)
    
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache


def affine_batchnorm_relu_backward(dout, cache):
    """
    Backward pass for the affine-batchnorm-relu convenience layer.
    """
    fc_cache, bn_cache, relu_cache = cache
    
    # ReLU backward
    dbn = relu_backward(dout, relu_cache)
    
    # Batch normalization backward
    da, dgamma, dbeta = batchnorm_backward_alt(dbn, bn_cache)
    
    # Affine backward
    dx, dw, db = affine_backward(da, fc_cache)
    
    return dx, dw, db, dgamma, dbeta


def affine_layernorm_relu_forward(x, w, b, gamma, beta, ln_param):
    """
    Convenience layer that performs an affine transform followed by layer 
    normalization and then ReLU.
    """
    # Affine forward
    a, fc_cache = affine_forward(x, w, b)
    
    # Layer normalization forward
    ln_out, ln_cache = layernorm_forward(a, gamma, beta, ln_param)
    
    # ReLU forward
    out, relu_cache = relu_forward(ln_out)
    
    cache = (fc_cache, ln_cache, relu_cache)
    return out, cache


def affine_layernorm_relu_backward(dout, cache):
    """
    Backward pass for the affine-layernorm-relu convenience layer.
    """
    fc_cache, ln_cache, relu_cache = cache
    
    # ReLU backward
    dln = relu_backward(dout, relu_cache)
    
    # Layer normalization backward
    da, dgamma, dbeta = layernorm_backward(dln, ln_cache)
    
    # Affine backward
    dx, dw, db = affine_backward(da, fc_cache)
    
    return dx, dw, db, dgamma, dbeta