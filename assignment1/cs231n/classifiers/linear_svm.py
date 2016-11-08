import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  # (D, C)
  dW = np.zeros(W.shape) # initialize the gradient as zero

  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0

  for i in xrange(num_train):

    scores = X[i].dot(W) # ( ,D) * (D, C) = ( ,C) .. but since one dimensional, (C, )
    correct_class_score = scores[y[i]]
    class_over_margin = 0.0 # initialize counting class with excessive margin for each training data

    for j in xrange(num_classes):
      if j == y[i]:
        continue  # continue here tells to repeat the loop!, meaning don't go forward!
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i]  # gradient to other rows where j NOT equal to y[i]
        class_over_margin += 1.0

    dW[:, y[i]] -= X[i] * class_over_margin  # gradient to row W where j == y[i].. but by definition, takes summation of other rows.



  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
    Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  """
  num_classes = W.shape[1]
  num_train = X.shape[0]

  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  # scores = X.dot(W) #(N, D) * (D, C) = (N, C)
  # correct_class_score = scores[y]
  #
  # margins = scores - correct_class_score + 1
  # margins = np.maximum(0, margins) # np.maximum finds max by comparing element vs element.. 0 here is goode enough for zeros matrix
  #
  # # margins.shape = (N, C)
  # margins[y] = 0  # for all training data point, make the value of column y zero.
  #
  # work above gives difference of SINGLE digit.. not acceptable... something went wrong with broadcasting but answer still surprisingly close

  scores = X.dot(W)  # (N, D) * (D, C) = (N, C)
  correct_class_score = scores[np.arange(num_train), y]  # Select one element from each row of num_train using the indices in y

  margins = scores.T - correct_class_score.T + 1  # transpose to make broadcasting work
  margins = np.maximum(0, margins)  # np.maximum finds max by comparing element vs element.. 0 here is goode enough for zeros matrix

  margins[y, np.arange(num_train)] = 0  # for each num_train, select an element using indices from y, turning it to 0

  loss = np.sum(margins)
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  # margins dimension = (C, N) ...  10 X 500.. already no negative.. and already 0 at y indices for each row
  # first part, for weights where j NOT y
  binary = np.zeros(margins.shape)
  binary[margins > 0] = 1

  # (N, ) ... (500, )
  # second part, for weight where J IS Y
  class_over_margin = np.sum(binary, axis=0)
  binary[y, np.arange(num_train)] = -class_over_margin

  # (D, N) * (N, C) = (D, C)
  dW += np.dot(X.T, binary.T)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  dW /= num_train
  dW += reg * W

  return loss, dW
