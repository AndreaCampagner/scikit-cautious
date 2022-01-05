from sklearn.metrics import rand_score
import numpy as np
from scipy.optimize import linprog

def dual_rand(c1, c2):
  return 1 - rand_score(c1,c2)


def rough_distrib_distance(R1, R2, metric=dual_rand):
  values = set()
  for c1 in R1:
    for c2 in R2:
      values.add(metric(c1,c2))

  return values

def distrib_distance(m1, m2, metric=dual_rand):
  M1, vals1 = evid_to_distrib(m1)
  M2, vals2 = evid_to_distrib(m2)
  RS1 = [rough_to_distrib(r) for r in M1]
  RS2 = [rough_to_distrib(r) for r in M2]
  values = {}
  for r1 in range(len(RS1)):
    for r2 in range(len(RS2)):
      R1 = RS1[r1]
      R2 = RS2[r2]
      val = frozenset(rough_distrib_distance(R1, R2, metric))
      if val not in values:
        values[val] = 0.0
      values[val] += vals1[r1]*vals2[r2]
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
