import numpy as np
from itertools import chain, combinations


def powerset(iterable):
    s = list(iterable)
    sets = chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    sets = [frozenset(elem) for elem in sets]
    return sets

def to_pairwise(assignment):
  relation = np.zeros((len(assignment), len(assignment), 4))

  for i in range(len(assignment)):
    for j in range(len(assignment)):
      if i == j:
        relation[i,j,1] = 1
      else:
        relation[i,j,0] = assignment[i][frozenset()] + assignment[j][frozenset()] - assignment[i][frozenset()]*assignment[j][frozenset()]
        for s_i in assignment[i]:
          for s_j in assignment[j]:
            if len(s_i) == 1 and s_i == s_j:
              relation[i,j,1] += assignment[i][s_i]*assignment[j][s_j]
              relation[i,j,3] += assignment[i][s_i]*assignment[j][s_j]
            elif s_i.intersection(s_j) == frozenset():
              relation[i,j,2] += assignment[i][s_i]*assignment[j][s_j]
            else:
              relation[i,j,3] += assignment[i][s_i]*assignment[j][s_j]
      
        relation[i,j,2] -= relation[i,j,0]
        relation[i,j,3] -= relation[i,j,1]

  return relation


def rough_to_evidential(assignment, n_clusters):
  print("Rough")
  m = []
  pset = powerset(range(n_clusters))

  for i in range(assignment.shape[0]):
    temp = frozenset(np.where(assignment[i] > 0)[0])
    m.append({})
    for s in pset:
      m[i][s] = 0.0
    m[i][temp] = 1.0

  return m

def fuzzy_to_evidential(assignment, n_clusters):
  print("Fuzzy")
  m = []
  pset = powerset(range(n_clusters))

  for i in range(assignment.shape[0]):
    temp = np.where(assignment[i] > 0)[0]
    m.append({})
    for s in pset:
      m[i][s] = 0.0
    for j in temp:
      m[i][frozenset([j])] = assignment[i,j]

  return m

def possibility_to_evidential(assignment, n_clusters):
  print("Possibility")
  m = []
  pset = powerset(range(n_clusters))

  for i in range(assignment.shape[0]):
    m.append({})
    for s in pset:
      m[i][s] = 0.0

    vals = np.sort(np.unique(assignment[i]))[::-1]
    vals = list(vals)
    if vals[-1] != 0.0:
      vals.append(0.0)
    for v in range(len(vals) - 1):
      temp = np.where(assignment[i] >= vals[v])[0]
      m[i][frozenset(temp)] = vals[v] - vals[v+1]
  return m

def hard_to_evidential(assignment, n_clusters):
  print("Hard")
  m = []
  pset = powerset(range(n_clusters))

  for i in range(assignment.shape[0]):
    m.append({})
    for s in pset:
      m[i][s] = 0.0
    m[i][frozenset([assignment[i]])] = 1.0
  return m


def to_evidential(assignment, n_clusters, cluster_type='rough'):
  print("Main")

  if cluster_type == 'rough':
    return rough_to_evidential(assignment, n_clusters) 
  elif cluster_type == 'fuzzy':
    return fuzzy_to_evidential(assignment, n_clusters)
  elif cluster_type == 'possibility':
    return possibility_to_evidential(assignment, n_clusters)
  else:
    return hard_to_evidential(assignment, n_clusters)

def evid_to_distrib(m):
  def evid_to_distrib_inner(m, i, arrs, vals, complete = False):
    if i == len(m):
      return arrs, vals
    inner_arrs = []
    inner_vals = []
    for s in m[i]:
      if m[i][s] > 0 or (complete and s != frozenset()):
        for a in range(len(arrs)):
          inner_arrs.append(arrs[a] + [s])
          inner_vals.append(vals[a]*m[i][s])
    
    out_arrs, out_vals = evid_to_distrib_inner(m, i+1, inner_arrs, inner_vals)
    return out_arrs, out_vals

  return evid_to_distrib_inner(m, 0, [[]], [1])

def rough_to_distrib(r):
  def rough_to_distrib_inner(m, i, arrs):
    if i == len(r):
      return arrs
    inner_arrs = []
    for s in r[i]:
      if arrs != []:
        for a in range(len(arrs)):
          inner_arrs.append(arrs[a] + [s])
      else:
          inner_arrs.append([s])
    
    out_arrs = rough_to_distrib_inner(m, i+1, inner_arrs)
    return out_arrs

  return rough_to_distrib_inner(r, 0, [])