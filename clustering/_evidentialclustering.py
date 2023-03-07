from itertools import chain, combinations
from scipy import linalg
from sklearn.metrics import pairwise_distances
import numpy as np
from sklearn.utils import resample
from sklearn.base import BaseEstimator, ClusterMixin

def powerset(iterable):
    s = list(iterable)
    sets = chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    sets = [frozenset(elem) for elem in sets]
    return sets


class EvidentialCMeans(BaseEstimator,ClusterMixin):
  def __init__(self, n_clusters=2, epsilon=0.001, iters=100, random_state=None, n_init=10, metric='euclidean', alpha=3, beta=2, delta=1):
    self.n_clusters = n_clusters
    self.epsilon = epsilon
    self.iters = iters
    self.random_state = random_state
    self.n_init = n_init
    self.metric = metric
    self.alpha = alpha
    self.beta = beta
    self.delta = delta

  def fit(self, X, y=None):
    self.pset = powerset(range(self.n_clusters))
    self.base_centroids = resample(X, replace=False, n_samples=self.n_clusters, random_state=self.random_state)

    self.centroids = {}
    for s in self.pset:
      if s != frozenset():
        s_ind = np.array(list(s))
        self.centroids[s] = np.mean(self.base_centroids[s_ind], axis=0)

    self.m = []
    for i in range(X.shape[0]):
      self.m.append({})
      for s in self.pset:
        self.m[i][s] = 0.0

    for it in range(self.iters):
      
      for i in range(X.shape[0]):
        accum = 0
        for s in self.pset:
          if s != frozenset():
            dist = pairwise_distances(X[i].reshape(1,-1), self.centroids[s].reshape(1,-1), metric=self.metric)
            self.m[i][s] = len(s)**(-self.alpha/(self.beta - 1))*(dist+self.epsilon)**(-2/(self.beta-1))
            accum += self.m[i][s]
        total = 0
        for s in self.pset:
          if s != frozenset():
            self.m[i][s] /= (accum + self.delta**(-2/(self.beta - 1)))
            total += self.m[i][s]
        self.m[i][frozenset()] = 1 - total
        
      H = np.zeros((self.n_clusters, self.n_clusters))
      B = np.zeros((self.n_clusters, X.shape[1]))

      for l in range(self.n_clusters):
        for q in range(X.shape[1]):
          for i in range(X.shape[0]):
            accum = 0
            for s in self.pset:
              if l in s:
                accum += len(s)**(self.alpha - 1) * self.m[i][s]**self.beta
            B[l,q] += X[i,q]*accum

        for k in range(self.n_clusters):
          for i in range(X.shape[0]):
            for s in self.pset:
              if l in s and k in s:
                H[l,k] += len(s)**(self.alpha - 2)*self.m[i][s]**self.beta

      self.base_centroids = linalg.solve(H, B)
      self.centroids = {}
      for s in self.pset:
        if s != frozenset():
          s_ind = np.array(list(s))
          self.centroids[s] = np.mean(self.base_centroids[s_ind], axis=0)
    return self

  def predict(self, X):
    self.m = []
    for i in range(X.shape[0]):
      self.m.append({})
      for s in self.pset:
        self.m[i][s] = 0.0

    for i in range(X.shape[0]):
      accum = 0
      for s in self.pset:
        if s != frozenset():
          dist = pairwise_distances(X[i].reshape(1,-1), self.centroids[s].reshape(1,-1), metric=self.metric)
          self.m[i][s] = len(s)**(-self.alpha/(self.beta - 1))*(dist+self.epsilon)**(-2/(self.beta-1))
          accum += self.m[i][s]
    
      total = 0
      for s in self.pset:
        if s != frozenset():
          self.m[i][s] /= (accum + self.delta**(-2/(self.beta - 1)))
          total += self.m[i][s]
      self.m[i][frozenset()] = 1 - total
    return self.m

  def fit_predict(self, X, y=None):
    self.fit(X,y)
    return self.predict(X)