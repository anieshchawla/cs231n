import numpy as np
from random import shuffle
from past.builtins import xrange

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
  num_train = X.shape[0]
  num_classes = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
    scores = X[i].dot(W)
    # doing the following to maintain numerical stability
    scores-=np.max(scores)
    
    scores = np.exp(scores)
    expscore = np.sum(scores)
    scores/=expscore
    loss-=np.log(scores[y[i]])

    for j in range(num_classes):
        dW[:,j]+= X[i]*scores[j]
        if(j==y[i]):
            dW[:,y[i]]-=X[i]
           
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss/=num_train
  dW/=num_train
  loss+=0.5*reg*np.sum(W * W)
  dW += reg*W
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
  scores = X.dot(W)
  scores = (scores.T - scores[np.arange(len(scores)),scores.argmax(axis=1)]).transpose()
  scores = np.exp(scores)
  expval = scores.sum(axis=1)
  scores = (scores.T/expval).transpose()
  loss = np.log(scores[np.arange(len(scores)),y])
  loss = -1*np.mean(loss)
  scores[np.arange(len(scores)),y]-=1
  dW = X.T.dot(scores)
  dW /=num_train
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss += 0.5*reg*np.sum(W*W)
  dW += (reg*W)
  return loss, dW

