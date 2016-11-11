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
  num_dim = X.shape[1]
  num_train = X.shape[0]
  num_class = W.shape[1]

  for i in np.arange(num_train):
    scores = X[i].dot(W)    # (, D) * (D, C) = (, C)
    scores -= np.max(scores)    # to improve numerical stability..  subtracting array with highest value, making it zero and the rest negative
    exp_scores = np.exp(scores)

    dscores = np.zeros(num_class)

    for j in np.arange(num_class):
      dscores[j] = exp_scores[j]/(np.sum(exp_scores))
      if j == y[i]:
        correct_prob = exp_scores[j]/ (np.sum(exp_scores))
        loss += - np.log(correct_prob)
        dscores[j] -= 1

    #reshaping for matrix multiplication
    re_dscore = np.reshape(dscores, (1, num_class))
    re_Xi = np.reshape(X[i], (num_dim, 1))
    dW += np.dot(re_Xi, re_dscore)


  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  dW += reg * W

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
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  scores = X.dot(W)  # (N,D)*(D,c) = (N, C)
  scores -= np.max(scores)

  prob = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
  correct_logprob = -np.log(prob[np.arange(num_train), y])

  data_loss = np.sum(correct_logprob)/num_train
  reg_loss = 0.5 * reg * np.sum(W * W)

  loss = data_loss + reg_loss

  dscore = prob
  dscore[np.arange(num_train), y] -= 1
  dscore /= num_train

  dW = np.dot(X.T, dscore)
  dW += reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

