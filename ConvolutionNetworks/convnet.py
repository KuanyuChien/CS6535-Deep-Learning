import numpy as np
from cs6353.layers import *
from cs6353.fast_layers import *
from cs6353.layer_utils import *

def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
  """ Forward pass for affine layer followed by batch normalization and ReLU. """
  a, fc_cache = affine_forward(x, w, b)
  b, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
  out, relu_cache = relu_forward(b)
  cache = (fc_cache, bn_cache, relu_cache)
  return out, cache

def affine_bn_relu_backward(dout, cache):
  """ Backward pass for affine layer followed by batch normalization and ReLU. """
  fc_cache, bn_cache, relu_cache = cache
  
  db = relu_backward(dout, relu_cache)
  da, dgamma, dbeta = batchnorm_backward_alt(db, bn_cache)
  dx, dw, dbias = affine_backward(da, fc_cache)
  
  return dx, dw, dbias, dgamma, dbeta

def conv_sbn_relu_pool_forward(x, w, b, gamma, beta, conv_param, bn_param, pool_param):
  """ 
  Forward pass for Conv layer, Spatial Batch Norm, ReLU, and Max Pool. 
  Uses fast implementations for Conv and Pool.
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  b, sbn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
  c, relu_cache = relu_forward(b)
  out, pool_cache = max_pool_forward_fast(c, pool_param)
  
  cache = (conv_cache, sbn_cache, relu_cache, pool_cache)
  return out, cache

def conv_sbn_relu_pool_backward(dout, cache):
  """ Backward pass for Conv layer, Spatial Batch Norm, ReLU, and Max Pool. """
  conv_cache, sbn_cache, relu_cache, pool_cache = cache

  dc = max_pool_backward_fast(dout, pool_cache)
  db = relu_backward(dc, relu_cache)
  da, dgamma, dbeta = spatial_batchnorm_backward(db, sbn_cache)
  dx, dw, dbias = conv_backward_fast(da, conv_cache)
  
  return dx, dw, dbias, dgamma, dbeta

class DeepConvNet(object):
  """
  A deep convolutional network with Batch Normalization.
  Architecture: [conv-sbn-relu-pool] x 2 - affine-bn-relu - affine - softmax
  
  The network uses Spatial Batch Normalization after convolutional layers 
  and standard Batch Normalization after the hidden affine layer.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=64, filter_size=3,
               hidden_dim=1024, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.bn_params = {}

    C, H, W = input_dim
    F1 = num_filters # Number of filters for Layer 1
    F2 = num_filters # Number of filters for Layer 2
    
    # 1. Layer 1: Conv1 
    # W1 shape: (F1, C, HH, WW)
    self.params['W1'] = np.random.normal(0, weight_scale, (F1, C, filter_size, filter_size))
    self.params['b1'] = np.zeros(F1)
    
    # Spatial BN 1
    self.params['gamma1'] = np.ones(F1)
    self.params['beta1'] = np.zeros(F1)
    self.bn_params['bn_param1'] = {'mode': 'train'}
    
    H1, W1 = H // 2, W // 2 # After 2x2 max pool
    
    # 2. Layer 2: Conv2 (Input channels F1, Output channels F2)
    # W2 shape: (F2, F1, HH, WW)
    self.params['W2'] = np.random.normal(0, weight_scale, (F2, F1, filter_size, filter_size))
    self.params['b2'] = np.zeros(F2)
    
    # Spatial BN 2
    self.params['gamma2'] = np.ones(F2)
    self.params['beta2'] = np.zeros(F2)
    self.bn_params['bn_param2'] = {'mode': 'train'}
    
    H2, W2 = H1 // 2, W1 // 2 # After 2x2 max pool
    
    # Input size for the first Affine layer
    D_in_3 = F2 * H2 * W2

    # 3. Layer 3: Affine3
    # W3 shape: (D_in_3, hidden_dim)
    self.params['W3'] = np.random.normal(0, weight_scale, (D_in_3, hidden_dim))
    self.params['b3'] = np.zeros(hidden_dim)
    
    # Vanilla BN 3 (after Affine3)
    self.params['gamma3'] = np.ones(hidden_dim)
    self.params['beta3'] = np.zeros(hidden_dim)
    self.bn_params['bn_param3'] = {'mode': 'train'}

    # 4. Layer 4: Output Affine4
    # W4 shape: (hidden_dim, num_classes)
    self.params['W4'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
    self.params['b4'] = np.zeros(num_classes)
        
    for k, v in self.params.items():
        self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    # Set the BN mode ('train' or 'test')
    mode = 'test' if y is None else 'train'
    for key, bn_param in self.bn_params.items():
        bn_param['mode'] = mode

    # Unpack parameters
    W1, b1, gamma1, beta1 = self.params['W1'], self.params['b1'], self.params['gamma1'], self.params['beta1']
    W2, b2, gamma2, beta2 = self.params['W2'], self.params['b2'], self.params['gamma2'], self.params['beta2']
    W3, b3, gamma3, beta3 = self.params['W3'], self.params['b3'], self.params['gamma3'], self.params['beta3']
    W4, b4 = self.params['W4'], self.params['b4']
    
    bn_param1, bn_param2, bn_param3 = self.bn_params['bn_param1'], self.bn_params['bn_param2'], self.bn_params['bn_param3']
    
    filter_size = W1.shape[2]
    # Conv parameters (3x3 filter, stride 1, pad 1 for same-size output before pooling)
    conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
    # Pool parameters (2x2 pool, stride 2)
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    
    # L1: Conv-SBN-ReLU-Pool (Output shape: N x F1 x H/2 x W/2)
    out1, cache1 = conv_sbn_relu_pool_forward(X, W1, b1, gamma1, beta1, conv_param, bn_param1, pool_param)
    
    # L2: Conv-SBN-ReLU-Pool (Output shape: N x F2 x H/4 x W/4)
    out2, cache2 = conv_sbn_relu_pool_forward(out1, W2, b2, gamma2, beta2, conv_param, bn_param2, pool_param)
    
    # L3: Affine-BN-ReLU (Output shape: N x hidden_dim)
    out3, cache3 = affine_bn_relu_forward(out2, W3, b3, gamma3, beta3, bn_param3)
    
    # L4: Affine (Output scores shape: N x num_classes)
    scores, cache4 = affine_forward(out3, W4, b4)

    if y is None:
        return scores

    loss, grads = 0, {}
    
    # Compute Softmax Loss
    loss, dscores = softmax_loss(scores, y)
    
    # Add L2 Regularization to loss
    loss += 0.5 * self.reg * (np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3) + np.sum(W4*W4))

    # Backward Pass
    
    # L4: Affine
    dout3, grads['W4'], grads['b4'] = affine_backward(dscores, cache4)
    grads['W4'] += self.reg * W4

    # L3: Affine-BN-ReLU
    dout2, grads['W3'], grads['b3'], grads['gamma3'], grads['beta3'] = affine_bn_relu_backward(dout3, cache3)
    grads['W3'] += self.reg * W3

    # L2: Conv-SBN-ReLU-Pool
    dout1, grads['W2'], grads['b2'], grads['gamma2'], grads['beta2'] = conv_sbn_relu_pool_backward(dout2, cache2)
    grads['W2'] += self.reg * W2
    
    # L1: Conv-SBN-ReLU-Pool
    dX, grads['W1'], grads['b1'], grads['gamma1'], grads['beta1'] = conv_sbn_relu_pool_backward(dout1, cache1)
    grads['W1'] += self.reg * W1

    return loss, grads