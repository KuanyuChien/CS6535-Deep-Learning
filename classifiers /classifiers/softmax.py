import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
    
  for i in range(num_train):
    scores = X[i].dot(W)
    scores -= np.max(scores) # Numerical stability: shift scores by max
    
    # Compute softmax probabilities
    exp_scores = np.exp(scores)
    softmax_probs = exp_scores / np.sum(exp_scores)
    
    # Add to loss (negative log likelihood of correct class)
    loss += -np.log(softmax_probs[y[i]])
    
    for j in range(num_classes):
        if j == y[i]:
            dW[:, j] += (softmax_probs[j] - 1) * X[i] # Gradient for correct class
        else:
            dW[:, j] += softmax_probs[j] * X[i] # Gradient for incorrect classes
    
  loss /= num_train
  dW /= num_train
  
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  scores = X.dot(W)  # (N, C)
  
  # Numerical stability: shift scores by max along each row
  scores -= np.max(scores, axis=1, keepdims=True)
  
  # Compute softmax probabilities
  exp_scores = np.exp(scores)
  softmax_probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
  
  # Compute loss
  correct_class_probs = softmax_probs[np.arange(num_train), y]
  loss = -np.sum(np.log(correct_class_probs)) / num_train
  
  # Compute gradient
  # Start with softmax probabilities
  dsoftmax = softmax_probs.copy()
  # Subtract 1 from correct classes
  dsoftmax[np.arange(num_train), y] -= 1
  # Compute gradient w.r.t. W
  dW = X.T.dot(dsoftmax) / num_train # (D, C)
  
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

