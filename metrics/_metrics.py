from sklearn.metrics import rand_score
import numpy as np
from scipy.optimize import linprog
from scipy.optimize import linear_sum_assignment

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


def dual_rand(c1, c2):
  return 1 - rand_score(c1,c2)

def sample_rough(r, samples=100):
  arrs = []
  for i in range(samples):
    arr = []
    for j in range(len(r)):
      arr.append(np.random.choice(list(r[j])))
    arrs.append(arr)
  return np.unique(arrs,axis=0)

def sample_evid(m, samples=100):
  arrs = set()
  counts = {}
  for i in range(samples):
    arr = []
    for j in range(len(m)):
      vals = []
      probs = []
      for s in m[j]:
        if s == frozenset():
          vals.append(frozenset([-j]))
        else:
          vals.append(s)
        probs.append(m[j][s])
      arr.append(np.random.choice(vals, p=probs))
    arr_t = tuple(arr)
    if arr_t in arrs:
      counts[arr_t] += 1
    else:
      arrs.add(arr_t)
      counts[arr_t] = 1


  out_arrs = [list(arr) for arr in arrs]
  out_vals = np.array([counts[arr] for arr in arrs])
  return out_arrs, out_vals/np.sum(out_vals)


def rough_distrib_distance(R1, R2, metric=dual_rand):
  values = set()
  for c1 in R1:
    for c2 in R2:
      values.add(metric(c1,c2))

  return values

def evid_distrib_distances(m1, m2, metric=dual_rand, approximate=False, samples_outer=100, samples_inner=100):
  M1 = vals1 = 0
  if approximate:
    M1, vals1 = sample_evid(m1, samples_outer)
  else:
    M1, vals1 = evid_to_distrib(m1)

  M2 = vals2 = 0
  if approximate:
    M2, vals2 = sample_evid(m2, samples_outer)
  else:
    M2, vals2 = evid_to_distrib(m2)

  RS1 = []
  if approximate:
    RS1 = [sample_rough(r, samples_inner) for r in M1]
  else:
    RS1 = [rough_to_distrib(r) for r in M1]

  RS2 = []
  if approximate:
    RS2 = [sample_rough(r, samples_inner) for r in M2]
  else:
    RS2 = [rough_to_distrib(r) for r in M2]

  values = {}
  k = 0
  for r1 in range(len(RS1)):
    for r2 in range(len(RS2)):
      R1 = RS1[r1]
      R2 = RS2[r2]
      val = frozenset(rough_distrib_distance(R1, R2, metric))
      if val == frozenset():
        k += vals1[r1]*vals2[r2]
      else:
        if val not in values:
          values[val] = 0.0
        values[val] += vals1[r1]*vals2[r2]
  for val in values:
    values[val] /= (1 - k)
  return values

def summary_distrib(vals):
  lower = 0
  upper = 0
  for val in vals:
    tl = np.min(list(val))
    tu = np.max(list(val))
    lower += tl*vals[val]
    upper += tu*vals[val]
  return lower, upper

def transport_rough(R1, R2, metric=dual_rand, alpha=0):
  v1 = 0
  lower = np.inf
  for c1 in R1:
    min_val = np.inf
    for c2 in R2:
      val = metric(c1,c2)
      if val < min_val:
        min_val = val
    if min_val > v1:
      v1 = min_val
    if min_val < lower:
      lower = min_val

  v2 = 0
  for c2 in R2:
    min_val = np.inf
    for c1 in R1:
      val = metric(c1,c2)
      if val < min_val:
        min_val = val
    if min_val > v2:
      v2 = min_val

  upper = np.max([v1,v2])
  return alpha*upper + (1-alpha)*lower

def transport_evid(m1, m2, metric=dual_rand, alpha = 0):
  M1, vals1 = evid_to_distrib(m1)
  M2, vals2 = evid_to_distrib(m2)
  RS1 = [rough_to_distrib(r) for r in M1]
  RS2 = [rough_to_distrib(r) for r in M2]
  c = np.zeros((len(M1),len(M2)))
  for R1 in range(len(RS1)):
    for R2 in range(len(RS2)):
      c[R1,R2] = transport_rough(RS1[R1],RS2[R2],metric,alpha)
  c = c.reshape(len(M1)*len(M2))
  
  A_eq = np.zeros((len(M1)+len(M2)+1, len(M1)*len(M2)))
  b_eq = np.zeros(len(M1)+len(M2)+1)
  for R1 in range(len(RS1)):
    b_eq[R1] = vals1[R1]
    for R2 in range(len(RS2)):
      A_eq[R1, len(RS2)*R1 + R2] = 1

  for R2 in range(len(RS2)):
    b_eq[len(RS1) + R2] = vals2[R2]
    for R1 in range(len(RS1)):
      A_eq[len(RS1) + R2, len(RS2)*R1 + R2] = 1

  A_eq[-1,:] = 1
  b_eq[-1] = 1

  bounds = [(0,1)]*len(c)
  return linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='interior-point').fun


def approx_rand(m1, m2, alpha=0.0):
  mr1 = to_pairwise(m1)
  mr2 = to_pairwise(m2)

  def pair_distance(mxy1, mxy2, alpha=0.0):
    #e s ns a
    c = [0, 1, 1, 1,
         1, 0, 1, alpha,
         1, 1, 0, alpha,
         1, alpha, alpha, 0]
    
    A_eq = [[1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1],
            [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0],
            [0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0],
            [0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0],
            [0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]

    bounds = [(0,1)]*16
    return linprog(c=c, A_eq=A_eq, b_eq=list(mxy1)+list(mxy2)+[1], bounds=bounds, method='interior-point').fun

  res = 0
  for x in range(mr1.shape[0]):
    for y in range(mr1.shape[1]):
      val = 1 - pair_distance(mr1[x,y], mr2[x,y], alpha)
      res += val
  return res/(mr1.shape[0]*mr1.shape[1])

def partition_distance(C1, C2):
  clusters1 = np.unique(C1)
  clusters2 = np.unique(C2)

  dists = np.zeros((len(clusters1), len(clusters2)))
  for c1 in range(len(clusters1)):
    idx1 = set(np.where(C1 == clusters1[c1])[0])
    for c2 in range(len(clusters2)):
      idx2 = set(np.where(C2 == clusters2[c2])[0])
      idx = (idx1 - idx2).union(idx2 - idx1)
      dists[c1,c2] = len(idx)
  print(dists)
  
  row_ind, col_ind = linear_sum_assignment(dists)
  print(row_ind, col_ind)
  return dists[row_ind, col_ind].sum()/(2*len(C1))


def to_cluster_indicator(m, c):
  mass = np.zeros(4)

  mass[0] = m[frozenset()]
  mass[1] = m[frozenset([c])]
  for s in m:
    if c not in s and s != frozenset():
      mass[2] += m[s]
    elif c in s and s != frozenset([c]):
      mass[3] += m[s]

  return mass

def evid_partition_distance(M1, M2, n_clusters1, n_clusters2, alpha=0.0):
  dists = np.zeros((n_clusters1, n_clusters2))
  for c1 in range(n_clusters1):
    for c2 in range(n_clusters2):
      for i in range(len(M1)):
         m1 = to_cluster_indicator(M1[i], c1)
         m2 = to_cluster_indicator(M2[i], c2)
         dists[c1,c2] += pair_distance(m1, m2, alpha)

  row_ind, col_ind = linear_sum_assignment(dists)
  return dists[row_ind, col_ind].sum()/(2*len(M1))
