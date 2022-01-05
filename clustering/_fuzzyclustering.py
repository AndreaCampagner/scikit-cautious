from sklearn.metrics import pairwise_distances
import numpy as np
from sklearn.utils import resample
from sklearn.base import BaseEstimator, ClusterMixin

class FuzzyCMeans(BaseEstimator,ClusterMixin):
  def __init__(self, n_clusters=2, epsilon=0.001, iters=100, random_state=None, n_init=10, metric='euclidean', m=2, method='fuzzy'):
    self.n_clusters = n_clusters
    self.epsilon = epsilon
    self.iters = iters
    self.random_state = random_state
    self.n_init = n_init
    self.metric = metric
    self.m = m
    self.method = method

  def fit(self, X, y=None):
    self.centroids = resample(X, replace=False, n_samples=self.n_clusters)
    self.cluster_assignments = np.zeros((X.shape[0], self.n_clusters))

    for it in range(self.iters):
      dists = pairwise_distances(X, self.centroids, metric=self.metric)
      self.cluster_assignments = 1/(dists/np.sum(dists, axis=1)[:,np.newaxis] + self.epsilon)**self.m
      for k in range(self.n_clusters):
        self.centroids[k] = np.sum(self.cluster_assignments[:,k][:,np.newaxis]**self.m*X, axis=0)/np.sum(self.cluster_assignments[:,k]**self.m)

    return self

  def predict(self, X):
    dists = pairwise_distances(X, self.centroids, metric=self.metric)
    self.cluster_assignments = 1/(dists/np.sum(dists, axis=1)[:,np.newaxis] + self.epsilon)**self.m
    if self.method == 'fuzzy':
      self.cluster_assignments = self.cluster_assignments/np.sum(self.cluster_assignments, axis=1)[:, np.newaxis]
    elif self.method == 'possibility':
      self.cluster_assignments = self.cluster_assignments/np.max(self.cluster_assignments, axis=1)[:, np.newaxis]
    return self.cluster_assignments

  def fit_predict(self, X, y=None):
    self.fit(X,y)
    return self.predict(X)