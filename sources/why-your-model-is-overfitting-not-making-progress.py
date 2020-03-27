
#import libraries
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#import datasts
train_df = pd.read_csv('../input/santander-customer-transaction-prediction/train.csv')
test_df = pd.read_csv('../input/santander-customer-transaction-prediction/test.csv')
full = pd.concat([train_df, test_df], sort = False)
feat1, feat2 = 'var_81', 'var_139'

fig = plt.subplots(figsize=(15, 5))

#plot pdf feat 1
plt.subplot(1, 2, 1)
sns.kdeplot(train_df[feat1][train_df['target'] == 0], shade=True, color="b", label = 'target = 0')
sns.kdeplot(train_df[feat1][train_df['target'] == 1], shade=True, color="r", label = 'target = 1')
plt.title(feat1)
plt.xlabel('Feature Values')
plt.ylabel('Probability')

#plot pdf feat 2
plt.subplot(1, 2, 2)
sns.kdeplot(train_df[feat2][train_df['target'] == 0], shade=True, color="b", label = 'target = 0')
sns.kdeplot(train_df[feat2][train_df['target'] == 1], shade=True, color="r", label = 'target = 1')
plt.title(feat2)
plt.xlabel('Feature Values')
plt.ylabel('Probability')

plt.show()
from sklearn.neighbors import KernelDensity
from operator import itemgetter

feat = 'var_81'

def calculate_pdf_difference(feat, df_feature, df_target, IQR_multiplier, bin_bandwidth_multiplier, print_number_bins):
    #Agreggating feature values in bin format using the Freedman-Diaconis rule
    IQR = df_feature[feat].quantile([0.75]).values - df_feature[feat].quantile([0.25]).values #Interquartile range (IQR)
    n = len(df_feature[feat])
    bin_size = IQR_multiplier*IQR/n**(1/3)
    bin_number = int(np.round((df_feature[feat].max() - df_feature[feat].min())/bin_size))
    binvalues = pd.cut(df_feature[feat], bins = bin_number, labels = range(bin_number)).astype('float')
    
    if print_number_bins:
        print('There are {} bins in the feature {}'.format(bin_number, feat))

    #Calculate the PDFs using the df_target
    pdf_0 = KernelDensity(kernel='gaussian', bandwidth=bin_size*bin_bandwidth_multiplier)
    pdf_0.fit(np.array(df_target[feat][df_target['target'] == 0]).reshape(-1,1))

    pdf_1 = KernelDensity(kernel='gaussian', bandwidth=bin_size*bin_bandwidth_multiplier)
    pdf_1.fit(np.array(df_target[feat][df_target['target'] == 1]).reshape(-1,1))

    #Creates an X array with the average feature value for each bin
    x = np.array(np.arange(min(df_feature[feat]) + bin_size/2 ,max(df_feature[feat]), bin_size)).reshape(-1,1)

    #gets the pdf values based on the X array
    log_pdf_0 = np.exp(pdf_0.score_samples(x))
    log_pdf_1 = np.exp(pdf_1.score_samples(x))

    #creates a dictionary that links the bin number with the PDFs value difference
    pdf_dict = dict()
    for i in range(bin_number):
        pdf_dict[i] = log_pdf_1[i] - log_pdf_0[i] 

    #gets the PDF difference for each row of the dataset based on its equivalent bin.
    bin_pdf_values = np.array(itemgetter(*list(binvalues))(pdf_dict))

    return bin_pdf_values, x, log_pdf_0, log_pdf_1

feat1, feat2 = 'var_81', 'var_139'

fig = plt.subplots(figsize=(15, 5))

#plot pdf feat 1
bin_pdf_values, x, log_pdf_0, log_pdf_1 = calculate_pdf_difference(feat = feat1, df_feature = full, df_target = train_df, IQR_multiplier = 2, bin_bandwidth_multiplier = 1.5, print_number_bins = True)

plt.subplot(1, 2, 1)

sns.kdeplot(train_df[feat1][train_df['target'] == 0], shade=False, color="b", label = 'target = 0')
sns.kdeplot(train_df[feat1][train_df['target'] == 1], shade=False, color="r", label = 'target = 1')
plt.plot(x, log_pdf_0)
plt.plot(x, log_pdf_1) 
plt.title(feat1)
plt.xlabel('Feature Values')
plt.ylabel('Probability')

#plot pdf feat 2
bin_pdf_values, x, log_pdf_0, log_pdf_1 = calculate_pdf_difference(feat = feat2, df_feature = full, df_target = train_df, IQR_multiplier = 2, bin_bandwidth_multiplier = 1.5, print_number_bins = True)

plt.subplot(1, 2, 2)
sns.kdeplot(train_df[feat2][train_df['target'] == 0], shade=False, color="b", label = 'target = 0')
sns.kdeplot(train_df[feat2][train_df['target'] == 1], shade=False, color="r", label = 'target = 1')
plt.plot(x, log_pdf_0)
plt.plot(x, log_pdf_1) 
plt.title(feat2)
plt.xlabel('Feature Values')
plt.ylabel('Probability')

plt.show()

feat1, feat2 = 'var_81', 'var_139'



print('-----------------------------------------------')
IQR_multiplier = 0.5
bin_bandwidth_multiplier = 0.2
print('IQR_multiplier', IQR_multiplier)
print('bin_bandwidth_multiplier', bin_bandwidth_multiplier)
print('-----------------------------------------------')
fig = plt.subplots(figsize=(15, 5))
#plot pdf feat 1
bin_pdf_values, x, log_pdf_0, log_pdf_1 = calculate_pdf_difference(feat = feat1, df_feature = full, df_target = train_df, IQR_multiplier = IQR_multiplier, bin_bandwidth_multiplier = bin_bandwidth_multiplier, print_number_bins = True)

plt.subplot(1, 2, 1)

sns.kdeplot(train_df[feat1][train_df['target'] == 0], shade=False, color="b", label = 'target = 0')
sns.kdeplot(train_df[feat1][train_df['target'] == 1], shade=False, color="r", label = 'target = 1')
plt.plot(x, log_pdf_0)
plt.plot(x, log_pdf_1) 
plt.title(feat1)
plt.xlabel('Feature Values')
plt.ylabel('Probability')

#plot pdf feat 2
bin_pdf_values, x, log_pdf_0, log_pdf_1 = calculate_pdf_difference(feat = feat2, df_feature = full, df_target = train_df, IQR_multiplier = IQR_multiplier, bin_bandwidth_multiplier = bin_bandwidth_multiplier, print_number_bins = True)

plt.subplot(1, 2, 2)
sns.kdeplot(train_df[feat2][train_df['target'] == 0], shade=False, color="b", label = 'target = 0')
sns.kdeplot(train_df[feat2][train_df['target'] == 1], shade=False, color="r", label = 'target = 1')
plt.plot(x, log_pdf_0)
plt.plot(x, log_pdf_1) 
plt.title(feat2)
plt.xlabel('Feature Values')
plt.ylabel('Probability')

plt.show()

print('-----------------------------------------------')
IQR_multiplier = 5
bin_bandwidth_multiplier = 0.2
print('IQR_multiplier', IQR_multiplier)
print('bin_bandwidth_multiplier', bin_bandwidth_multiplier)
print('-----------------------------------------------')
fig = plt.subplots(figsize=(15, 5))
#plot pdf feat 1
bin_pdf_values, x, log_pdf_0, log_pdf_1 = calculate_pdf_difference(feat = feat1, df_feature = full, df_target = train_df, IQR_multiplier = IQR_multiplier, bin_bandwidth_multiplier = bin_bandwidth_multiplier, print_number_bins = True)

plt.subplot(1, 2, 1)

sns.kdeplot(train_df[feat1][train_df['target'] == 0], shade=False, color="b", label = 'target = 0')
sns.kdeplot(train_df[feat1][train_df['target'] == 1], shade=False, color="r", label = 'target = 1')
plt.plot(x, log_pdf_0)
plt.plot(x, log_pdf_1) 
plt.title(feat1)
plt.xlabel('Feature Values')
plt.ylabel('Probability')

#plot pdf feat 2
bin_pdf_values, x, log_pdf_0, log_pdf_1 = calculate_pdf_difference(feat = feat2, df_feature = full, df_target = train_df, IQR_multiplier = IQR_multiplier, bin_bandwidth_multiplier = bin_bandwidth_multiplier, print_number_bins = True)

plt.subplot(1, 2, 2)
sns.kdeplot(train_df[feat2][train_df['target'] == 0], shade=False, color="b", label = 'target = 0')
sns.kdeplot(train_df[feat2][train_df['target'] == 1], shade=False, color="r", label = 'target = 1')
plt.plot(x, log_pdf_0)
plt.plot(x, log_pdf_1) 
plt.title(feat2)
plt.xlabel('Feature Values')
plt.ylabel('Probability')

plt.show()

print('-----------------------------------------------')
IQR_multiplier = 3
bin_bandwidth_multiplier = 0.2
print('IQR_multiplier', IQR_multiplier)
print('bin_bandwidth_multiplier', bin_bandwidth_multiplier)
print('-----------------------------------------------')

fig = plt.subplots(figsize=(15, 5))

#plot pdf feat 1
bin_pdf_values, x, log_pdf_0, log_pdf_1 = calculate_pdf_difference(feat = feat1, df_feature = full, df_target = train_df, IQR_multiplier = IQR_multiplier, bin_bandwidth_multiplier = bin_bandwidth_multiplier, print_number_bins = True)

plt.subplot(1, 2, 1)

sns.kdeplot(train_df[feat1][train_df['target'] == 0], shade=False, color="b", label = 'target = 0')
sns.kdeplot(train_df[feat1][train_df['target'] == 1], shade=False, color="r", label = 'target = 1')
plt.plot(x, log_pdf_0)
plt.plot(x, log_pdf_1) 
plt.title(feat1)
plt.xlabel('Feature Values')
plt.ylabel('Probability')

#plot pdf feat 2
bin_pdf_values, x, log_pdf_0, log_pdf_1 = calculate_pdf_difference(feat = feat2, df_feature = full, df_target = train_df, IQR_multiplier = IQR_multiplier, bin_bandwidth_multiplier = bin_bandwidth_multiplier, print_number_bins = True)

plt.subplot(1, 2, 2)
sns.kdeplot(train_df[feat2][train_df['target'] == 0], shade=False, color="b", label = 'target = 0')
sns.kdeplot(train_df[feat2][train_df['target'] == 1], shade=False, color="r", label = 'target = 1')
plt.plot(x, log_pdf_0)
plt.plot(x, log_pdf_1) 
plt.title(feat2)
plt.xlabel('Feature Values')
plt.ylabel('Probability')

plt.show()

print('-----------------------------------------------')
IQR_multiplier = 3
bin_bandwidth_multiplier = 3
print('IQR_multiplier', IQR_multiplier)
print('bin_bandwidth_multiplier', bin_bandwidth_multiplier)
print('-----------------------------------------------')

fig = plt.subplots(figsize=(15, 5))


#plot pdf feat 1
bin_pdf_values, x, log_pdf_0, log_pdf_1 = calculate_pdf_difference(feat = feat1, df_feature = full, df_target = train_df, IQR_multiplier = IQR_multiplier, bin_bandwidth_multiplier = bin_bandwidth_multiplier, print_number_bins = True)

plt.subplot(1, 2, 1)

sns.kdeplot(train_df[feat1][train_df['target'] == 0], shade=False, color="b", label = 'target = 0')
sns.kdeplot(train_df[feat1][train_df['target'] == 1], shade=False, color="r", label = 'target = 1')
plt.plot(x, log_pdf_0)
plt.plot(x, log_pdf_1) 
plt.title(feat1)
plt.xlabel('Feature Values')
plt.ylabel('Probability')

#plot pdf feat 2
bin_pdf_values, x, log_pdf_0, log_pdf_1 = calculate_pdf_difference(feat = feat2, df_feature = full, df_target = train_df, IQR_multiplier = IQR_multiplier, bin_bandwidth_multiplier = bin_bandwidth_multiplier, print_number_bins = True)

plt.subplot(1, 2, 2)
sns.kdeplot(train_df[feat2][train_df['target'] == 0], shade=False, color="b", label = 'target = 0')
sns.kdeplot(train_df[feat2][train_df['target'] == 1], shade=False, color="r", label = 'target = 1')
plt.plot(x, log_pdf_0)
plt.plot(x, log_pdf_1) 
plt.title(feat2)
plt.xlabel('Feature Values')
plt.ylabel('Probability')

plt.show()

#defines the test_df target columns based on a submission of score 0.901
pd.options.mode.chained_assignment = None  # disables copy warning from pandas library. default='warn'
submission = pd.read_csv('../input/submission-0901/submission_0901.csv')
test_df['target'] = submission['target']
test_df.sort_values('target', ascending = False, inplace = True)
test_df['target'].iloc[:20098] = 1
test_df['target'].iloc[20098:] = 0
feat1, feat2 = 'var_81', 'var_139'

fig = plt.subplots(figsize=(15, 5))

#plot pdf feat 1
plt.subplot(1, 2, 1)
sns.kdeplot(train_df[feat1][train_df['target'] == 0], shade=False, color="b", label = 'train target = 0')
sns.kdeplot(test_df[feat1][test_df['target'] == 0], shade=False, color="g", label = 'test target = 0')
sns.kdeplot(train_df[feat1][train_df['target'] == 1], shade=False, color="r", label = 'train target = 1')
sns.kdeplot(test_df[feat1][test_df['target'] == 1], shade=False, color="k", label = 'test target = 1')
plt.title(feat1)
plt.xlabel('Feature Values')
plt.ylabel('Probability')

#plot pdf feat 2
plt.subplot(1, 2, 2)
sns.kdeplot(train_df[feat2][train_df['target'] == 0], shade=False, color="b", label = 'train target = 0')
sns.kdeplot(test_df[feat2][test_df['target'] == 0], shade=False, color="g", label = 'test target = 0')
sns.kdeplot(train_df[feat2][train_df['target'] == 1], shade=False, color="r", label = 'train target = 1')
sns.kdeplot(test_df[feat2][test_df['target'] == 1], shade=False, color="k", label = 'test target = 1')
plt.title(feat2)
plt.xlabel('Feature Values')
plt.ylabel('Probability')

plt.show()
print('-----------------------------------------------')
IQR_multiplier = 3
bin_bandwidth_multiplier = 0.2
print('IQR_multiplier', IQR_multiplier)
print('bin_bandwidth_multiplier', bin_bandwidth_multiplier)
print('-----------------------------------------------')

fig = plt.subplots(figsize=(15, 5))


#plot pdf feat 1


plt.subplot(1, 2, 1)

bin_pdf_values, x, log_pdf_0, log_pdf_1 = calculate_pdf_difference(feat = feat1, df_feature = full, df_target = train_df, IQR_multiplier = IQR_multiplier, bin_bandwidth_multiplier = bin_bandwidth_multiplier, print_number_bins = True)
plt.plot(x, log_pdf_0, 'b', label = 'train target = 0')
plt.plot(x, log_pdf_1, 'r', label = 'train target = 1')
bin_pdf_values, x, log_pdf_0, log_pdf_1 = calculate_pdf_difference(feat = feat1, df_feature = full, df_target = test_df, IQR_multiplier = IQR_multiplier, bin_bandwidth_multiplier = bin_bandwidth_multiplier, print_number_bins = False)
plt.plot(x, log_pdf_0, 'g', label = 'test target = 0')
plt.plot(x, log_pdf_1, 'k', label = 'test target = 1')
plt.title(feat1)
plt.xlabel('Feature Values')
plt.ylabel('Probability')

#plot pdf feat 2
plt.subplot(1, 2, 2)
bin_pdf_values, x, log_pdf_0, log_pdf_1 = calculate_pdf_difference(feat = feat2, df_feature = full, df_target = train_df, IQR_multiplier = IQR_multiplier, bin_bandwidth_multiplier = bin_bandwidth_multiplier, print_number_bins = True)
plt.plot(x, log_pdf_0, 'b', label = 'train target = 0')
plt.plot(x, log_pdf_1, 'r', label = 'train target = 1')
bin_pdf_values, x, log_pdf_0, log_pdf_1 = calculate_pdf_difference(feat = feat2, df_feature = full, df_target = test_df, IQR_multiplier = IQR_multiplier, bin_bandwidth_multiplier = bin_bandwidth_multiplier, print_number_bins = False)
plt.plot(x, log_pdf_0, 'g', label = 'test target = 0')
plt.plot(x, log_pdf_1, 'k', label = 'test target = 1')
plt.title(feat1)
plt.xlabel('Feature Values')
plt.ylabel('Probability')

plt.show()
from sklearn.metrics import roc_auc_score
from sklearn import tree
import graphviz

cols = [c for c in train_df.columns if (c not in ['ID_code', 'target'])]

valid = train_df.sample(frac=0.2, random_state=42)
train = train_df[ ~train_df.index.isin(valid.index) ].sample(frac=1)

#model = tree.DecisionTreeClassifier(max_leaf_nodes=4)
model = tree.DecisionTreeRegressor(max_leaf_nodes=4)
model.fit(train[cols], train['target'])
#pred_val = model.predict_proba(valid[cols])[:,1]
pred_val = model.predict(valid[cols])

print('AUC = ',round( roc_auc_score(valid['target'],pred_val),4 ) )
tree_graph = tree.export_graphviz(model, out_file=None, max_depth = 10,
    impurity = False, feature_names = cols, class_names = ['0', '1'],
    rounded = True, filled= True )
graphviz.Source(tree_graph)  