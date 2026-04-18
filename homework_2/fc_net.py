from builtins import object
import os
import numpy as np

from .layers import *
from .layer_utils import *

class TwoLayerNet(object):
    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        self.params = {}
        self.reg = reg

        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)

    def loss(self, X, y=None):
        scores = None
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        layer1_out, layer1_cache = affine_relu_forward(X, W1, b1)
        scores, layer2_cache = affine_forward(layer1_out, W2, b2)

        if y is None:
            return scores

        loss, grads = 0, {}
        
        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2))

        dhidden, dW2, db2 = affine_backward(dscores, layer2_cache)
        dx, dW1, db1 = affine_relu_backward(dhidden, layer1_cache)

        grads['W1'] = dW1 + self.reg * W1
        grads['b1'] = db1
        grads['W2'] = dW2 + self.reg * W2
        grads['b2'] = db2

        return loss, grads

    def save(self, fname):
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      params = self.params
      if not os.path.exists(os.path.dirname(fpath)):
          os.makedirs(os.path.dirname(fpath))
      np.save(fpath, params)
      print(fname, "saved.")
    
    def load(self, fname):
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      if not os.path.exists(fpath):
        print(fname, "not available.")
        return False
      else:
        params = np.load(fpath, allow_pickle=True).item()
        self.params = params
        print(fname, "loaded.")
        return True