from itertools import chain, combinations
import numpy as np

def powerset(iterable):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    xs = list(iterable)
    return chain.from_iterable(combinations(xs,n) for n in range(1,len(xs)+1))

def tw_layer(clf, X_test, epsilon, alpha):
  y_pred = []
  y_proba = clf.predict_proba(X_test)
  for i in range(y_proba.shape[0]):
    sets = powerset(range(y_proba.shape[1]))
    min_risk = np.max(epsilon) + 0.5
    decision = -1
    for s in sets:
      risk = alpha(len(s), y_proba.shape[1])
      for y in range(y_proba.shape[1]):
        if y not in s:
          risk += epsilon[y]*y_proba[i,y]
      if risk < min_risk:
        min_risk = risk
        decision = s
    y_pred.append(decision)
  return y_pred

def tw_layer_single(y_in, epsilon, alpha):
  sets = powerset(range(len(y_in)))
  min_risk = np.max(epsilon) + 0.5
  decision = -1
  for s in sets:
    risk = alpha(len(s), len(y_in))
    for y in range(len(y_in)):
      if y not in s:
        risk += epsilon[y]*y_in[y]
    if risk < min_risk:
      min_risk = risk
      decision = s
  return decision




  