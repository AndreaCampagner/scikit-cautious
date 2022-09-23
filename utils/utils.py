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

def conformal_correction(clf, X, y, X_test, epsilon, max_n, superset=False):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.90, random_state=0)
    clf.fit(X_train, y_train)

    scores_valid = np.zeros(X_valid.shape[0])
    y_pred = tw_layer(clf, X_valid, epsilon)
    for i in range(X_valid.shape[0]):
      if not superset:
        if (len(y_pred[i]) > 1) and (y_valid[i] in y_pred[i]):
          scores_valid[i] = alpha(len(y_pred[i]), max_n)
        elif (y_valid[i] in y_pred[i]) and len(y_pred[i]) == 1:
          scores_valid[i] = 0
        elif y_valid[i] == -1:
          scores_valid[i] = 0
        else:
          scores_valid[i] = epsilon[y_valid[i]]
      else:
        min_value = np.max(epsilon)
        temp_value = 0
        vals = np.where(y_valid[i,:] > 0)[0]
        for j in vals:
          if (len(y_pred[i]) > 1) and (j in y_pred[i]):
            temp_value = alpha(len(y_pred[i]), max_n)
          elif (j in y_pred[i]) and len(y_pred[i]) == 1:
            temp_value = 0
          else:
            temp_value = epsilon[j]
          
          if temp_value < min_value:
            min_value = temp_value
        scores_valid[i] = min_value

    scores_test = np.zeros((X_test.shape[0], max_n))
    y_pred = tw_layer(clf, X_test, epsilon)
    for i in range(X_test.shape[0]):
      for j in range(max_n):
        if j in y_pred[i]:
          if len(y_pred[i]) == 1:
            scores_test[i,j] = 0
          else:
            scores_test[i,j] = alpha(len(y_pred[i]), max_n)
        else:
          scores_test[i,j] = epsilon[j]
    
    pvalues = np.zeros((X_test.shape[0],max_n))
    for i in range(scores_test.shape[0]):
      for j in range(scores_test.shape[1]):
        pvalues[i,j] = (len(scores_valid[scores_valid >= scores_test[i, j]]) + 1)/(X_valid.shape[0] + 1)

    return pvalues, scores_test

def conformal_correction_alt(clf, X, y, X_test, max_n, superset=False):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.90, random_state=0)
    clf.fit(X_train, y_train)

    scores_valid = np.zeros(X_valid.shape[0])
    y_pred = clf.predict_proba(X_valid)
    for i in range(X_valid.shape[0]):
      if not superset:
        scores_valid[i] = np.max(y_pred[i,:]) - y_pred[i,y[i]]
      else:
        min_value = 1.0
        vals = np.where(y_valid[i,:] > 0)[0]
        for val in vals:
          temp_val = np.max(y_pred[i,:]) - y_pred[i,val]
          if temp_val < min_value:
            min_value = temp_val
        scores_valid[i] = min_value


    scores_test = np.zeros((X_test.shape[0], max_n))
    y_pred = clf.predict_proba(X_test)
    for i in range(X_test.shape[0]):
      for j in range(max_n):
        scores_test[i,j] = np.max(y_pred[i,:]) - y_pred[i,j]
    
    pvalues = np.zeros((X_test.shape[0],max_n))
    for i in range(scores_test.shape[0]):
      for j in range(scores_test.shape[1]):
        pvalues[i,j] = (len(scores_valid[scores_valid >= scores_test[i, j]]) + 1)/(X_valid.shape[0] + 1)

    return pvalues, scores_test

def conformal_to_tw(pvalues, epsilon):
  y_pred = []
  for i in range(pvalues.shape[0]):
    ps = np.zeros(pvalues.shape[1])
    decs = []
    risks = np.zeros(pvalues.shape[1])

    for j in range(pvalues.shape[1]):
      ps[j] = pvalues[i,j]
      decs.append(tuple(np.where(pvalues[i,:] >= ps[j])[0]))
      if len(decs[j]) == pvalues.shape[1]:
        risks[j] = alpha(pvalues.shape[1], pvalues.shape[1])
      else:
        max_risk = 0
        prob = 0
        for k in range(pvalues.shape[1]):
          if (k not in decs[j]):
            if epsilon[k] > max_risk:
              max_risk = epsilon[k]
            if pvalues[i,k] > prob:
              prob = pvalues[i,k]
        risks[j] = (1-prob)*alpha(len(decs[j]), pvalues.shape[1]) + prob*max_risk

    l = np.argmin(risks)
    y_pred.append(decs[l])
  return y_pred

def conformal_to_tw_alt(pvalues_arg, epsilon):
  y_pred = []
  pvalues = pvalues_arg.copy()
  for i in range(pvalues.shape[0]):
    probs = np.zeros(pvalues.shape[1])

    unique_values = np.unique(pvalues[i,:])
    unique_values.sort()

    max_in_row = np.max(pvalues[i,:])
    pvalues[i, pvalues[i,:] == max_in_row] = 1
    for j in range(len(unique_values)-1):
      val = unique_values[j]
      pvalues[i, pvalues[i,:] == val] = 1 - unique_values[j+1]
      

    for j in unique_values:
      alpha_cut = np.where(pvalues[i,:] >= j)[0]
      for k in alpha_cut:
        probs[k] += j/len(alpha_cut)

    y_pred.append(tw_layer_single(probs, epsilon))
  return y_pred




  