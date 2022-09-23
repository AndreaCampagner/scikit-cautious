from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import resample
from sklearn.base import clone

class InductiveConformalClassifier(BaseEstimator,ClassifierMixin):

  def __init__(self, estimator, train_size=0.75, random_state=None, epsilon=0.95):
    self.estimator = estimator
    self.train_size = train_size
    self.random_state = random_state
    self.epsilon = epsilon

  def fit(self, X, y):
    self.X = X
    self.y = y

    X_train, self.X_cal, y_train, self.y_cal = train_test_split(X, y, train_size=self.train_size, random_state=self.random_state)

    self.estimator.fit(X_train, y_train)

    y_probs = self.estimator.predict_proba(self.X_cal)

    self.scores = np.max(y_probs, axis=1) - np.choose(self.y_cal, y_probs.T)

  def predict(self, X, epsilon=None):
    y_proba = self.estimator.predict_proba(X)
    y_pred = []

    if epsilon is None:
      epsilon = self.epsilon

    for i in range(X.shape[0]):
      y_temp = []
      for j in range(y_proba.shape[1]):
        score = np.max(y_proba[i,:]) - y_proba[i,j]
        pval = (np.sum(score <= self.scores) + 1)/(self.scores.shape[0] + 1)
        if pval >= epsilon:
          y_temp.append(j)
      y_pred.append(y_temp)
    
    return y_pred

  def predict_proba(self, X):
    y_proba = self.estimator.predict_proba(X)
    y_pred = np.zeros(y_proba.shape)

    for i in range(X.shape[0]):
      for j in range(y_proba.shape[1]):
        score = np.max(y_proba[i,:]) - y_proba[i,j]
        y_pred[i,j] = (np.sum(score <= self.scores) + 1)/(self.scores.shape[0] + 1)

    return y_pred


class InductiveEvidentialPredictor(BaseEstimator,ClassifierMixin):

  def __init__(self, estimator, rule='min', train_size = 0.75, multi_source=False, n_estimators=100, epsilon = 0.95, random_state=None, num_dims=None):
    self.estimator = estimator
    self.train_size = train_size
    self.random_state = random_state
    self.multi_source = multi_source
    self.rule = rule
    self.n_estimators = n_estimators
    self.epsilon=epsilon
    self.num_dims = num_dims

  def fit(self, X, y):
    self.X = X
    self.y = y

    self.Xs_train = []
    self.Xs_cal = []
    self.ys_train = []
    self.ys_cal = []

    if self.multi_source:
      if self.num_dims is None:
        self.num_dims = len(X)
      seq = train_test_split(*X, *y, train_size=0.75, random_state=self.random_state)
      s = 0
      while s < self.num_dims:
        self.Xs_train.append(seq[2*s])
        self.Xs_cal.append(seq[2*s+1])
        self.ys_train.append(seq[2*len(X) + 2*s])
        self.ys_cal.append(seq[2*len(X) + 2*s + 1])
        s+=1
    else:
      for s in range(self.n_estimators):
        X_res, y_res = resample(X, y, replace=True, random_state=self.random_state)
        X_train, X_cal, y_train, y_cal = train_test_split(X_res, y_res, train_size=self.train_size, random_state=self.random_state)
        self.Xs_train.append(X_train)
        self.Xs_cal.append(X_cal)
        self.ys_train.append(y_train)
        self.ys_cal.append(y_cal)
    
    self.estimators = []
    self.scores = []

    for s in range(len(self.Xs_train)):
      self.estimators.append(clone(self.estimator).fit(self.Xs_train[s], self.ys_train[s]))
      y_probs = self.estimators[s].predict_proba(self.Xs_cal[s])
      scores = np.max(y_probs, axis=1) - np.choose(self.ys_cal[s], y_probs.T)
      self.scores.append(scores)

  def predict_proba(self, X):
    Xs_pred = []
    if self.multi_source:
      for s in range(len(self.estimators)):
        Xs_pred.append(X[s])
    else:
      for s in range(len(self.estimators)):
        Xs_pred.append(X)

    if self.rule == 'count':
      y_probas = []
      for s in range(len(self.estimators)):
        y_proba = self.estimators[s].predict_proba(Xs_pred[s])
        y_pred = np.zeros(y_proba.shape)
        for i in range(y_proba.shape[0]):
          for j in range(y_proba.shape[1]):
            score = np.max(y_proba[i,:]) - y_proba[i,j]
            y_pred[i,j] = (np.sum(score <= self.scores[s]) + 1)/(self.scores[s].shape[0] + 1)
        y_probas.append(y_pred)
      return y_probas


    y_proba = self.estimators[0].predict_proba(Xs_pred[0])
    y_pred = np.zeros(y_proba.shape)

    accum = np.zeros(y_proba.shape[0])
    temp_min = np.zeros(y_proba.shape)
    temp_max = np.zeros(y_proba.shape)

    for i in range(y_proba.shape[0]):
      for j in range(y_proba.shape[1]):
        score = np.max(y_proba[i,:]) - y_proba[i,j]
        val = (np.sum(score <= self.scores[0]) + 1)/(self.scores[0].shape[0] + 1)

        if self.rule == 'mean':
          val = val/len(self.estimators)
        y_pred[i,j] = val

      if self.rule == 'weighted':
        y_pred[i,:] *= (np.max(y_pred[i,:]) - np.partition(y_pred[i,:], -2)[-2])
        accum[i] += (np.max(y_pred[i,:]) - np.partition(y_pred[i,:], -2)[-2])
      if self.rule == 'dempster':
        accum = 1

    for s in range(1,len(self.estimators)):
      y_proba = self.estimators[s].predict_proba(Xs_pred[s])

      for i in range(y_proba.shape[0]):
        y_pred_temp = np.zeros(y_proba.shape[1])
        for j in range(y_proba.shape[1]):
          score = np.max(y_proba[i,:]) - y_proba[i,j]
          y_pred_temp[j] = (np.sum(score <= self.scores[s]) + 1)/(self.scores[s].shape[0] + 1)

          if self.rule == 'min' and y_pred_temp[j] <= y_pred[i,j]:
            y_pred[i,j] = y_pred_temp[j]
          elif self.rule == 'max' and y_pred_temp[j] >= y_pred[i,j]:
            y_pred[i,j] = y_pred_temp[j]
          elif self.rule == 'mean':
            y_pred[i,j] += y_pred_temp[j]/len(self.estimators)
          elif self.rule == 'dempster' or self.rule == 'fisher':
            accum *= (1 - np.max(np.abs(y_pred[i,:] - y_pred_temp)))
            y_pred[i,j] *= y_pred_temp[j]

        if self.rule == 'weighted':
          y_pred[i,:] += y_pred_temp*(np.max(y_pred_temp) - np.partition(y_pred_temp, -2)[-2])
          accum[i] += np.max(y_pred_temp) - np.partition(y_pred_temp, -2)[-2]
        if self.rule == 'dempster':
          conflict = accum
          for j in range(y_proba.shape[1]):
            y_pred[i,j] = np.min([1.0, y_pred[i,j]/conflict]) if conflict > 0 else 1


    if self.rule == 'fisher':
      for i in range(y_proba.shape[0]):
        for j in range(y_proba.shape[1]):
          temp = 0
          for s in range(len(self.estimators)):
            temp += (-np.log(y_pred[i,j]))**s/(np.math.factorial(s))
          temp *= y_pred[i,j]
          y_pred[i,j] = temp

    if self.rule == 'weighted':
      y_pred /= accum[:, np.newaxis]
    return y_pred
    


    return y_pred

  def predict(self, X, epsilon=None):
    y_proba = self.predict_proba(X)
    y_pred = []

    if epsilon is None:
      epsilon = self.epsilon

    if self.rule != 'count':
      for i in range(y_proba.shape[0]):
        y_temp = []
        for j in range(y_proba.shape[1]):
          if y_proba[i,j] >= epsilon:
            y_temp.append(j)
        y_pred.append(y_temp)

    else:
        for i in range(y_proba[0].shape[0]):
          counters = np.zeros(y_proba[0].shape[1])
          y_temp = []
          for c in range(len(y_proba)):
            for j in range(y_proba[c].shape[1]):
              if y_proba[c][i,j] >= epsilon:
                counters[j] += 1/len(y_proba)
          for j in range(y_proba[0].shape[1]):
            if counters[j] >= epsilon:
              y_temp.append(j)
          y_pred.append(y_temp)

    return y_pred