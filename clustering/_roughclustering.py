import numpy as np
from sklearn.utils import resample
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator, ClusterMixin


class RoughKMeans(BaseEstimator, ClusterMixin):
  def __init__(self, n_clusters=2, epsilon=1.1, iters=100, random_state=None, n_init=10, metric='euclidean'):
    self.n_clusters = n_clusters
    self.epsilon = epsilon
    self.iters = iters
    self.random_state = random_state
    self.n_init = n_init
    self.metric = metric

  def fit(self, X, y=None):
    self.centroids = resample(X, replace=False, n_samples=self.n_clusters)

    for it in range(self.iters):
      self.cluster_assignments = np.zeros((X.shape[0], self.n_clusters))
      nn = NearestNeighbors(n_neighbors=1, metric=self.metric)
      nn.fit(self.centroids)
      dists, indices = nn.kneighbors(X)
      for i in range(X.shape[0]):
        self.cluster_assignments[i,indices[i]] = 1
      for i in range(X.shape[0]):
        _, indices = nn.radius_neighbors(X[i].reshape(1,-1), radius=dists[i]*self.epsilon)
        self.cluster_assignments[i,indices[0]] = 1
      
      divs = np.sum(self.cluster_assignments, axis=1)
      for k in range(self.n_clusters):
        indices = self.cluster_assignments[:,k] == 1
        if len(indices[indices]) > 0:
          self.centroids[k] = np.mean(X[indices,:]/divs[indices,np.newaxis],axis=0)
    return self

  def fit_predict(self, X, y=None):
    self.fit(X,y)
    return self.predict(X)

  def predict(self, X):
    nn = NearestNeighbors(n_neighbors=1, radius=self.epsilon, metric=self.metric)
    nn.fit(self.centroids)
    dists, indices = nn.kneighbors(X)
    for i in range(X.shape[0]):
        self.cluster_assignments[i,indices[i]] = 1
    for i in range(X.shape[0]):
      _, indices = nn.radius_neighbors(X[i].reshape(1,-1), radius=dists[i]*self.epsilon)
      self.cluster_assignments[i,indices[0]] = 1
    return self.cluster_assignments