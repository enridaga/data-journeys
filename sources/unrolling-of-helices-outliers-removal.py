
import numpy as np
x = np.array([(62,397,103),(82,347,107),(93,288,120),
     (94,266,128),(65,163,169),(12,102,198),
     (48,138,180),(77,187,157),(85,209,149),(89,316,112)])
xm = np.mean(x,axis=0)
print(xm.shape)
x = x - xm
print(x.shape)
v, s, t = np.linalg.svd(x,full_matrices=True)
sigma1 = s[0]
sigma2 = s[1]
sigma3 = s[2]
v1 = t[0]
v2 = t[1]
v3 = t[2]
Z = np.zeros((x.shape[0],10), np.float32)
Z[:,0] = x[:,0]**2
Z[:,1] = 2*x[:,0]*x[:,1]
Z[:,2] = 2*x[:,0]*x[:,2]
Z[:,3] = 2*x[:,0]
Z[:,4] = x[:,1]**2
Z[:,5] = 2*x[:,1]*x[:,2]
Z[:,6] = 2*x[:,1]
Z[:,7] = x[:,2]**2
Z[:,8] = 2*x[:,2]
Z[:,9] = 1
v, s, t = np.linalg.svd(Z,full_matrices=True)
smallest_value = np.min(np.array(s))
smallest_index = np.argmin(np.array(s))
T = np.array(t)
T = T[smallest_index,:]
S = np.zeros((4,4),np.float32)
S[0,0] = T[0]
S[0,1] = S[1,0] = T[1]
S[0,2] = S[2,0] = T[2]
S[0,3] = S[3,0] = T[3]
S[1,1] = T[4]
S[1,2] = S[2,1] = T[5]
S[1,3] = S[3,1] = T[6]
S[2,2] = T[7]
S[2,3] = S[3,2] = T[8]
S[3,3] = T[9]
norm = np.linalg.norm(np.dot(Z,T), ord=2)**2
print(norm)
### matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from trackml.dataset import load_event, load_dataset
from trackml.score import score_event
# Change this according to your directory preferred setting
path_to_train = "../input/train_1"
# This event is in Train_1
event_prefix = "event000001000"
hits, cells, particles, truth = load_event(os.path.join(path_to_train, event_prefix))
from sklearn.preprocessing import StandardScaler
import hdbscan
from scipy import stats
from tqdm import tqdm
from sklearn.cluster import DBSCAN

class Clusterer(object):
    def __init__(self,rz_scales=[0.65, 0.965, 1.528]):                        
        self.rz_scales=rz_scales
    
    def _eliminate_outliers(self,labels,M):
        norms=np.zeros((len(labels)),np.float32)
        indices=np.zeros((len(labels)),np.float32)
        for i, cluster in tqdm(enumerate(labels),total=len(labels)):
            if cluster == 0:
                continue
            index = np.argwhere(self.clusters==cluster)
            index = np.reshape(index,(index.shape[0]))
            indices[i] = len(index)
            x = M[index]
            norms[i] = self._test_quadric(x)
        threshold1 = np.percentile(norms,90)*5
        threshold2 = 25
        threshold3 = 6
        for i, cluster in enumerate(labels):
            if norms[i] > threshold1 or indices[i] > threshold2 or indices[i] < threshold3:
                self.clusters[self.clusters==cluster]=0   
    def _test_quadric(self,x):
        if x.size == 0 or len(x.shape)<2:
            return 0
        xm = np.mean(x,axis=0)
        x = x - xm
        Z = np.zeros((x.shape[0],10), np.float32)
        Z[:,0] = x[:,0]**2
        Z[:,1] = 2*x[:,0]*x[:,1]
        Z[:,2] = 2*x[:,0]*x[:,2]
        Z[:,3] = 2*x[:,0]
        Z[:,4] = x[:,1]**2
        Z[:,5] = 2*x[:,1]*x[:,2]
        Z[:,6] = 2*x[:,1]
        Z[:,7] = x[:,2]**2
        Z[:,8] = 2*x[:,2]
        Z[:,9] = 1
        v, s, t = np.linalg.svd(Z,full_matrices=False)        
        smallest_index = np.argmin(np.array(s))
        T = np.array(t)
        T = T[smallest_index,:]        
        norm = np.linalg.norm(np.dot(Z,T), ord=2)**2
        return norm

    def _preprocess(self, hits):
        
        x = hits.x.values
        y = hits.y.values
        z = hits.z.values

        r = np.sqrt(x**2 + y**2 + z**2)
        hits['x2'] = x/r
        hits['y2'] = y/r

        r = np.sqrt(x**2 + y**2)
        hits['z2'] = z/r

        ss = StandardScaler()
        X = ss.fit_transform(hits[['x2', 'y2', 'z2']].values)
        for i, rz_scale in enumerate(self.rz_scales):
            X[:,i] = X[:,i] * rz_scale
       
        return X
    def _init(self, dfh, w1, w2, w3, w4, w5, w6, w7, epsilon, Niter):
        dfh['r'] = np.sqrt(dfh['x'].values ** 2 + dfh['y'].values ** 2 + dfh['z'].values ** 2)
        dfh['rt'] = np.sqrt(dfh['x'].values ** 2 + dfh['y'].values ** 2)
        dfh['a0'] = np.arctan2(dfh['y'].values, dfh['x'].values)
        dfh['z1'] = dfh['z'].values / dfh['rt'].values
        dfh['z2'] = dfh['z'].values / dfh['r'].values
        dfh['s1'] = dfh['hit_id']
        dfh['N1'] = 1
        dfh['z1'] = dfh['z'].values / dfh['rt'].values
        dfh['z2'] = dfh['z'].values / dfh['r'].values
        dfh['x1'] = dfh['x'].values / dfh['y'].values
        dfh['x2'] = dfh['x'].values / dfh['r'].values
        dfh['x3'] = dfh['y'].values / dfh['r'].values
        dfh['x4'] = dfh['rt'].values / dfh['r'].values
        mm = 1
        for ii in tqdm(range(int(Niter))):
            mm = mm * (-1)
            dfh['a1'] = dfh['a0'].values + mm * (dfh['rt'].values + 0.000005
                                                 * dfh['rt'].values ** 2) / 1000 * (ii / 2) / 180 * np.pi
            dfh['sina1'] = np.sin(dfh['a1'].values)
            dfh['cosa1'] = np.cos(dfh['a1'].values)
            ss = StandardScaler()
            dfs = ss.fit_transform(dfh[['sina1', 'cosa1', 'z1', 'z2','x1','x2','x3','x4']].values)
            cx = np.array([w1, w1, w2, w3, w4, w5, w6, w7])
            dfs = np.multiply(dfs, cx)
            clusters = DBSCAN(eps=epsilon, min_samples=1, metric="euclidean", n_jobs=32).fit(dfs).labels_
            dfh['s2'] = clusters
            dfh['N2'] = dfh.groupby('s2')['s2'].transform('count')
            maxs1 = dfh['s1'].max()
            dfh.loc[(dfh['N2'] > dfh['N1']) & (dfh['N2'] < 20),'s1'] = dfh['s2'] + maxs1
            dfh['N1'] = dfh.groupby('s1')['s1'].transform('count')
        return dfh['s1'].values
    def predict(self, hits):         
        self.clusters = self._init(hits,2.7474448671796874,1.3649721713529086,0.7034918842926337,
                                        0.0005549122352940002,0.023096034747190672,0.04619756315527515,
                                        0.2437077420144654,0.009750302717746615,338)
        X = self._preprocess(hits) 
        cl = hdbscan.HDBSCAN(min_samples=1,min_cluster_size=7,
                             metric='braycurtis',cluster_selection_method='leaf',algorithm='best', leaf_size=50)
        labels = np.unique(self.clusters)
        self._eliminate_outliers(labels,X)          
        max_len = np.max(self.clusters)
        mask = self.clusters == 0
        self.clusters[mask] = cl.fit_predict(X[mask])+max_len
        return self.clusters
model = Clusterer()
labels = model.predict(hits)
def create_one_event_submission(event_id, hits, labels):
    sub_data = np.column_stack(([event_id]*len(hits), hits.hit_id.values, labels))
    submission = pd.DataFrame(data=sub_data, columns=["event_id", "hit_id", "track_id"]).astype(int)
    return submission

submission = create_one_event_submission(0, hits, labels)
score = score_event(truth, submission)
print("Your score: ", score)
path_to_test = "../input/test"
test_dataset_submissions = []

create_submission = False # True for submission 
if create_submission:
    for event_id, hits, cells in load_dataset(path_to_test, parts=['hits', 'cells']):

        # Track pattern recognition 
        model = Clusterer()
        labels = model.predict(hits)

        # Prepare submission for an event
        one_submission = create_one_event_submission(event_id, hits, labels)
        test_dataset_submissions.append(one_submission)
        
        print('Event ID: ', event_id)

    # Create submission file
    submission = pd.concat(test_dataset_submissions, axis=0)
    submission.to_csv('submission.csv', index=False)
