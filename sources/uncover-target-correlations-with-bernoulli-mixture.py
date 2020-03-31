
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
### matplotlib inline
from PIL import Image
from scipy.misc import imread

import tensorflow as tf


import os
print(os.listdir("../input"))

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
train_labels = pd.read_csv("../input/train.csv")
label_names = {
    0:  "Nucleoplasmn",  
    1:  "Nuclear membrane",   
    2:  "Nucleoli",   
    3:  "Nucleoli fibrillar center",   
    4:  "Nuclear speckles",
    5:  "Nuclear bodies",   
    6:  "Endoplasmic reticulum",   
    7:  "Golgi apparatus",   
    8:  "Peroxisomes",   
    9:  "Endosomes",   
    10:  "Lysosomes", 
    11:  "Intermediate filaments",   
    12:  "Actin filaments",   
    13:  "Focal adhesion sites",   
    14:  "Microtubules",   
    15:  "Microtubule ends",   
    16:  "Cytokinetic bridge",   
    17:  "Mitotic spindle",   
    18:  "Microtubule organizing center",   
    19:  "Centrosome",   
    20:  "Lipid droplets",   
    21:  "Plasma membrane",   
    22:  "Cell junctions",   
    23:  "Mitochondria",   
    24:  "Aggresome",   
    25:  "Cytosol",   
    26:  "Cytoplasmic bodies",   
    27:  "Rods & rings"
}

reverse_train_labels = dict((v,k) for k,v in label_names.items())

def fill_targets(row):
    row.Target = np.array(row.Target.split(" ")).astype(np.int)
    for num in row.Target:
        name = label_names[int(num)]
        row.loc[name] = 1
    return row
for key in label_names.keys():
    train_labels[label_names[key]] = 0

train_labels = train_labels.apply(fill_targets, axis=1)
train_labels.head()
target_counts = train_labels.drop(["Id", "Target"],axis=1).sum(axis=0).sort_values(ascending=False)
plt.figure(figsize=(15,15))
sns.barplot(y=target_counts.index.values, x=target_counts.values, order=target_counts.index);
train_labels["number_of_targets"] = train_labels.drop(["Id", "Target"],axis=1).sum(axis=1)

def find_counts(special_target, labels):
    counts = labels[labels[special_target] == 1].drop(
        ["Id", "Target", "number_of_targets"],axis=1
    ).sum(axis=0)
    counts = counts[counts > 0]
    counts = counts.sort_values()
    return counts

lyso_endo_counts = find_counts("Lysosomes", train_labels)

plt.figure(figsize=(15,5))
sns.barplot(x=lyso_endo_counts.index.values, y=lyso_endo_counts.values, palette="Blues");
count_perc = np.round(100 * train_labels["number_of_targets"].value_counts() / train_labels.shape[0], 2)
plt.figure(figsize=(20,5))
sns.barplot(x=count_perc.index.values, y=count_perc.values, palette="Reds")
plt.xlabel("Number of targets per image")
plt.ylabel("% of data");
targets = train_labels.drop(["Id", "Target", "number_of_targets"], axis=1)
from scipy.special import logsumexp

class BernoulliMixture:
    
    def __init__(self, n_components, max_iter, tol=1e-3):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
    
    def fit(self,x):
        self.x = x
        self.init_params()
        log_bernoullis = self.get_log_bernoullis(self.x)
        self.old_logL = self.get_log_likelihood(log_bernoullis)
        for step in range(self.max_iter):
            if step > 0:
                self.old_logL = self.logL
            # E-Step
            self.gamma = self.get_responsibilities(log_bernoullis)
            self.remember_params()
            # M-Step
            self.get_Neff()
            self.get_mu()
            self.get_pi()
            # Compute new log_likelihood:
            log_bernoullis = self.get_log_bernoullis(self.x)
            self.logL = self.get_log_likelihood(log_bernoullis)
            if np.isnan(self.logL):
                self.reset_params()
                print(self.logL)
                break

    def reset_params(self):
        self.mu = self.old_mu.copy()
        self.pi = self.old_pi.copy()
        self.gamma = self.old_gamma.copy()
        self.get_Neff()
        log_bernoullis = self.get_log_bernoullis(self.x)
        self.logL = self.get_log_likelihood(log_bernoullis)
        
    def remember_params(self):
        self.old_mu = self.mu.copy()
        self.old_pi = self.pi.copy()
        self.old_gamma = self.gamma.copy()
    
    def init_params(self):
        self.n_samples = self.x.shape[0]
        self.n_features = self.x.shape[1]
        #self.gamma = np.zeros(shape=(self.n_samples, self.n_components))
        self.pi = 1/self.n_components * np.ones(self.n_components)
        self.mu = np.random.RandomState(seed=0).uniform(low=0.25, high=0.75, size=(self.n_components, self.n_features))
        self.normalize_mu()
    
    def normalize_mu(self):
        sum_over_features = np.sum(self.mu, axis=1)
        for k in range(self.n_components):
            self.mu[k,:] /= sum_over_features[k]
            
    def get_responsibilities(self, log_bernoullis):
        gamma = np.zeros(shape=(log_bernoullis.shape[0], self.n_components))
        Z =  logsumexp(np.log(self.pi[None,:]) + log_bernoullis, axis=1)
        for k in range(self.n_components):
            gamma[:, k] = np.exp(np.log(self.pi[k]) + log_bernoullis[:,k] - Z)
        return gamma
        
    def get_log_bernoullis(self, x):
        log_bernoullis = self.get_save_single(x, self.mu)
        log_bernoullis += self.get_save_single(1-x, 1-self.mu)
        return log_bernoullis
    
    def get_save_single(self, x, mu):
        mu_place = np.where(np.max(mu, axis=0) <= 1e-15, 1e-15, mu)
        return np.tensordot(x, np.log(mu_place), (1,1))
        
    def get_Neff(self):
        self.Neff = np.sum(self.gamma, axis=0)
    
    def get_mu(self):
        self.mu = np.einsum('ik,id -> kd', self.gamma, self.x) / self.Neff[:,None] 
        
    def get_pi(self):
        self.pi = self.Neff / self.n_samples
    
    def predict(self, x):
        log_bernoullis = self.get_log_bernoullis(x)
        gamma = self.get_responsibilities(log_bernoullis)
        return np.argmax(gamma, axis=1)
        
    def get_sample_log_likelihood(self, log_bernoullis):
        return logsumexp(np.log(self.pi[None,:]) + log_bernoullis, axis=1)
    
    def get_log_likelihood(self, log_bernoullis):
        return np.mean(self.get_sample_log_likelihood(log_bernoullis))
        
    def score(self, x):
        log_bernoullis = self.get_log_bernoullis(x)
        return self.get_log_likelihood(log_bernoullis)
    
    def score_samples(self, x):
        log_bernoullis = self.get_log_bernoullis(x)
        return self.get_sample_log_likelihood(log_bernoullis)
from sklearn.model_selection import train_test_split

X = targets.values
x_train, x_test = train_test_split(X, shuffle=True, random_state=0)
components_to_test = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
scores = []


for n in range(len(components_to_test)):
    if n > 0:
        old_score = score
    model = BernoulliMixture(components_to_test[n], 200)
    model.fit(x_train)
    score = model.score(x_test)
    scores.append(score)
    if n > 0: 
        if score < old_score:
            estimated_components = components_to_test[n-1]
            break
        
estimated_components
model = BernoulliMixture(estimated_components, 200)
model.fit(X)
results = targets.copy()
results["cluster"] = np.argmax(model.gamma, axis=1)
grouped_targets = results.groupby("cluster").sum() / results.drop("cluster", axis=1).sum(axis=0) * 100
grouped_targets = grouped_targets.apply(np.round).astype(np.int32)

plt.figure(figsize=(20,15))
sns.heatmap(grouped_targets, cmap="Blues", annot=True, fmt="g", cbar=False);
plt.title("How are specific proteins distributed over clusters in percent?");
cluster_names = {
    0: "Actin filaments & Focal adhesion sites",
    1: "Aggresomes",
    2: "Microtubules, Mitotic spindle, Cytokinetic Bridge",
    3: "RodsRings, Microtubule ends, Nuclear bodies",
    4: "Some nuclear membranes",
    5: "various - RodsRings",
    6: "various - Mitotic spindle, Organizing center, Cytosol",
    7: "Nuclear bodies & Aggresomes",
    8: "Nuclear speckles",
    9: "Nuclear membrane & Actin filaments & Focal adhesion sites",
    10: "Endoplasmatic reticulum & Endosomes & Lysosomes",
    11: "Low dense 1",
    12: "Mitotic spindle, Organizing center",
    13: "Plasma membrane & various",
    14: "Nucleoli fibrillar center & Peroxisomes",
    15: "Low dense 2",
    16: "Nucleoli fibrillar center & Cytoplasmic bodies",
    17: "Nucleoli & Microtubule ends & Peroxisomes & Rods Rings",
    18: "Mitochondria & Lipid droplets & RodsRings & Nucleoli",
    19: "Low dense 3",
    20: "Golgi apparatus",
    21: "Intermediate filaments",
    22: "Centrosome",
    23: "Cytoplasmic bodies & Aggresomes",
    24: "Lipid droplets & Peroxisomes & Cell junctions"
}
cluster_size = results.groupby("cluster").Nucleoplasmn.count()
cluster_composition = results.groupby("cluster").sum().apply(lambda l: l/cluster_size, axis=0) * 100
cluster_composition = cluster_composition.apply(np.round).astype(np.int)

cluster_composition = cluster_composition.reset_index()
cluster_composition.cluster = cluster_composition.cluster.apply(lambda l: cluster_names[l])
cluster_composition = cluster_composition.set_index("cluster")

plt.figure(figsize=(20,20))
sns.heatmap(cluster_composition, cmap="Oranges", annot=True, fmt="g", cbar=False);
plt.title("How present alias hot are specific targets within one cluster?");
plt.ylabel("");
results["cluster_names"] = results.cluster.apply(lambda l: cluster_names[l])
cluster_ids = np.arange(0, estimated_components)
names = [cluster_names[l] for l in cluster_ids]

pi = pd.Series(data=model.pi, index=names).sort_values(ascending=False)
plt.figure(figsize=(20,5))
sns.barplot(x=pi.index, y=pi.values, palette="Reds_r", order=pi.index)
plt.xticks(rotation=90);
model.mu.shape
mu = pd.DataFrame(data=model.mu * 100, index=names, columns=results.drop(["cluster", "cluster_names"], axis=1).columns.values)
mu = mu.apply(np.round)

plt.figure(figsize=(20,20))
sns.heatmap(mu, cmap="Purples", annot=True, fmt="g", cbar=False)
cluster_counts = results.groupby("cluster").cluster.count()
cluster_counts = cluster_counts.sort_values()
names = [cluster_names[num] for num in cluster_counts.index]

plt.figure(figsize=(20,5))
sns.barplot(x=names, y=cluster_counts.values, order=names)
plt.xticks(rotation=90);
results["number_of_targets"] = results.drop("cluster", axis=1).sum(axis=1)

multilabel_stats = results.groupby("cluster_names").number_of_targets.value_counts() 
multilabel_stats /= results.groupby("cluster_names").number_of_targets.count()
multilabel_stats = multilabel_stats.unstack()
multilabel_stats.fillna(0, inplace=True)
multilabel_stats = 100 * multilabel_stats
multilabel_stats = multilabel_stats.apply(np.round)
multilabel_stats = multilabel_stats.astype(np.int)

plt.figure(figsize=(20,5))
sns.heatmap(multilabel_stats.transpose(),
            square=True,
            cbar=False,
            cmap="Greens",
            annot=True);
sample_logLs = model.score_samples(X)
my_threshold = np.quantile(sample_logLs, 0.05)
plt.figure(figsize=(20,5))
sns.distplot(sample_logLs)
plt.axvline(my_threshold, color="Red")
plt.xlabel("Sample log likelihood of bernoulli mixture")
plt.title("Choosing a threshold to detect anomalies")
plt.ylabel("Density")
results["anomaly"] = np.where(sample_logLs <= my_threshold, 1, 0)
anomalies = results.groupby("cluster_names").anomaly.value_counts() / results.groupby("cluster_names").cluster.count() * 100
anomalies = anomalies.unstack()
anomalies.fillna(0, inplace=True)
anomalies = anomalies.apply(np.round)
anomalies = anomalies.astype(np.int)

plt.figure(figsize=(20,5))
sns.heatmap(anomalies.transpose(), cmap="Reds", annot=True, square=True, cbar=False);
results["certainty"] = np.sort(model.gamma, axis=1)[:,-1]
certainties = results.certainty.values

plt.figure(figsize=(20,5))
sns.distplot(certainties, color="Orange")
plt.xlabel("Certainty of cluster assignment")
plt.ylabel("Density")
plt.title("How sure was the model in predicting the winner?");
plt.figure(figsize=(20,5))
sns.boxplot(x="cluster_names", y="certainty", data=results)
plt.ylim([0,1])
plt.xticks(rotation=90)
plt.xlabel("");
aggresome_cluster = results.loc[results.cluster==1].drop(["cluster",
                                 "cluster_names",
                                 "number_of_targets", 
                                  "anomaly", 
                                 "certainty"], axis=1).copy()

counts = aggresome_cluster.sum()
columns_of_interest = list(counts[counts>0].index.values)
aggresome_cluster = aggresome_cluster.loc[:,columns_of_interest]

aggresome_combinations = pd.DataFrame(index=aggresome_cluster.columns.values,
                                      columns=aggresome_cluster.columns.values)

for col in aggresome_combinations.columns:
    aggresome_combinations.loc[col,:] = aggresome_cluster[aggresome_cluster[col] == 1].sum()

mask = np.zeros_like(aggresome_combinations, dtype=np.bool)
mask[np.triu_indices_from(mask, k=1)] = True


plt.figure(figsize=(8,8))
sns.set(style="white")
sns.heatmap(aggresome_combinations, mask=mask, cmap="Reds",
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, vmin=0, vmax=50, annot=True, fmt="g")
aggresome_cluster.shape
af_cluster = results.loc[results.cluster==0].drop(["cluster",
                                 "cluster_names",
                                 "number_of_targets", 
                                  "anomaly", 
                                 "certainty"], axis=1).copy()

counts = af_cluster.sum()
columns_of_interest = list(counts[counts>0].index.values)
af_cluster = af_cluster.loc[:,columns_of_interest]

af_combinations = pd.DataFrame(index=af_cluster.columns.values,
                               columns=af_cluster.columns.values)

for col in af_combinations.columns:
    af_combinations.loc[col,:] = af_cluster[af_cluster[col] == 1].sum()

mask = np.zeros_like(af_combinations, dtype=np.bool)
mask[np.triu_indices_from(mask, k=1)] = True


plt.figure(figsize=(8,8))
sns.set(style="white")
sns.heatmap(af_combinations, mask=mask, cmap="Reds",
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, vmin=0, vmax=50, annot=True, fmt="g")
results["alternative_cluster"] = np.argsort(model.gamma, axis=1)[:,-2]
results["alternative_names"] = results.alternative_cluster.apply(lambda l: cluster_names[l])
results["alternative_certainties"] = np.sort(model.gamma, axis=1)[:,-2]
competition = np.round(100 * results.groupby(
    "cluster_names").alternative_names.value_counts() / results.groupby(
    "cluster_names").alternative_names.count())
competition = competition.unstack()
competition.fillna(0, inplace=True)
plt.figure(figsize=(20,15))
sns.heatmap(competition, cmap="Greens", annot=True, fmt="g", square=True, cbar=False)
results.to_csv("target_group_analysis.csv")