import numpy as np
from scipy.spatial import KDTree
from sklearn.base import BaseEstimator, ClassifierMixin

class IFSKNeighborsClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, k=3, k_range=0, m=2, m_range=0, shadowed=None):
        self.k = k
        self.k_range = k_range
        self.m = m
        self.m_range = m_range
        self.shadowed = shadowed
        
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.k_sup = self.k + self.k_range + 1
        y = np.array(y)
        self.tree = KDTree(X)
        
        self.ifs_classes = {}
        
        for i in range(X.shape[0]):
            cur_class = y[i]
            if cur_class not in self.ifs_classes:
                self.ifs_classes[cur_class] = []
        
        for key in self.ifs_classes:
            for i in range(X.shape[0]):
                u_min = 1
                u_max = 0
                for k_ind in range(self.k, self.k_sup):
                    _, indices = self.tree.query(X[i, :], k_ind)
                    indices = indices[1:]
                    neighbors = y[indices]
                    nnc = len(neighbors[neighbors == key])
                    u = 0
                    if key == y[i]:
                        u = 0.51 + (nnc/k_ind)*0.49
                    else:
                        u = (nnc/k_ind)*0.49
                    if u > u_max:
                        u_max = u
                    if u < u_min:
                        u_min = u
                self.ifs_classes[key].append((u_min, u_max))
                
        if self.shadowed == 'fuzzy':
            self.to_fuzzy_shadowed()
        elif self.shadowed == 'knowledge':
            self.to_knowledge_shadowed()
        elif self.shadowed == 'total':
            self.to_total_shadowed()
                
    def to_fuzzy_shadowed(self):
        entropy = 0
        fs = []
        for key in self.ifs_classes:
            for pair in self.ifs_classes[key]:
                fs.append(1 - np.abs(pair[0] - 0.5) - np.abs(1 - pair[1] - 0.5))
                entropy += fs[-1]
            boundary_size = int(np.ceil(entropy))
            indices = np.argsort(fs)[::-1]
            indices = indices[:boundary_size]
            for i in range(len(self.ifs_classes[key])):
                if i in indices:
                    self.ifs_classes[key][i] = (0.5,0.5)
                elif self.ifs_classes[key][i][0] >= 1 - self.ifs_classes[key][i][1]:
                    self.ifs_classes[key][i] = (1,1)
                else:
                    self.ifs_classes[key][i] = (0,0)
                    
    def to_knowledge_shadowed(self):
        entropy = 0
        fs = []
        for key in self.ifs_classes: 
            for pair in self.ifs_classes[key]:
                fs.append(pair[1] - pair[0])
                entropy += fs[-1]
            boundary_size = int(np.ceil(entropy))
            indices = np.argsort(fs)[::-1]
            indices = indices[:boundary_size]
            for i in range(len(self.ifs_classes[key])):
                if i in indices:
                    self.ifs_classes[key][i] = (0,1)
                elif self.ifs_classes[key][i][0] >= 1 - self.ifs_classes[key][i][1]:
                    self.ifs_classes[key][i] = (1,1)
                else:
                    self.ifs_classes[key][i] = (0,0)
                    
    def to_total_shadowed(self):
        entropy = 0
        fs = []
        for key in self.ifs_classes:
            for pair in self.ifs_classes[key]:
                f = 1 - np.abs(pair[0] - 0.5) - np.abs(1 - pair[1] - 0.5)
                k = pair[1] - pair[0]
                fs.append(np.max([f,k]))
                entropy += fs[-1]
            boundary_size = int(np.ceil(entropy))
            indices = np.argsort(fs)[::-1]
            indices = indices[:boundary_size]
            for i in range(len(self.ifs_classes[key])):
                
                if self.ifs_classes[key][i][0] >= 1 - self.ifs_classes[key][i][1]:
                    if i in indices:
                        self.ifs_classes[key][i] = (0.5,1)
                    else:
                        self.ifs_classes[key][i] = (1,1)
                    
                else:
                    if i in indices:
                        self.ifs_classes[key][i] = (0,0.5)
                    else:
                        self.ifs_classes[key][i] = (0,0)
    
    def predict(self, X):
        y_pred = []
        for x in X:
            distances, indices = self.tree.query(x, self.k)
            classes = {}
            for j in range(len(indices)):
                i = indices[j]
                for response in range(len(self.ifs_classes)):
                    if self.shadowed is None:
                        dist_m = []
                        for m in range(self.m, self.m + self.m_range + 1):
                            d = (1/(distances[j]**(2/m-1))/(np.sum(distances**(-2/m-1))))
                            dist_m.append(d)
                        d = (np.amin(dist_m), np.amax(dist_m))
                        #print(self.ifs_classes[response])
                        resp_min = self.ifs_classes[response][i][0]
                        resp_max = self.ifs_classes[response][i][1]
                        vote_min = np.amin([resp_min*d[0], resp_min*d[1], resp_max*d[0], resp_max*d[1]])
                        vote_max = np.amax([resp_min*d[0], resp_min*d[1], resp_max*d[0], resp_max*d[1]])
                        if response in classes:
                            classes[response] += (vote_min + vote_max)/2
                        else:
                            classes[response] = (vote_min + vote_max)/2
                    else:
                        d = (1/(distances[j]**(2/self.m-1))/(np.sum(distances**(-2/self.m-1))))
                        temp = self.ifs_classes[response][i][0]*d
                        temp += self.ifs_classes[response][i][1]*d
                        if response in classes:
                            classes[response] += temp/2
                        else:
                            classes[response] = temp/2

            sorted_classes = sorted(classes.items(), key=lambda x: x[1], reverse=True)
            y_pred.append(sorted_classes[0][0])
        return y_pred