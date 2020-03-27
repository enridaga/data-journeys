
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
dataset= [11,10,12,14,12,15,14,13,15,102,12,14,17,19,107, 10,13,12,14,12,108,12,11,14,13,15,10,15,12,10,14,13,15,10]
outlier=[]
def detect_outlier(data):
    
    threshold=3
    mean = np.mean(data)
    std =np.std(data)
    
    
    for i in data:
        z_score= (i - mean)/std 
        if np.abs(z_score) > threshold:
            outlier.append(i)
    return outlier
outlier_data=detect_outlier(dataset)
outlier_data
## data sorting
dataset= sorted(dataset)
dataset = pd.DataFrame(dataset)
quantile1 = dataset.quantile(0.25)
quantile3 = dataset.quantile(0.75)
print(quantile1.values,quantile3.values)
## Find the IQR
iqrValue=quantile3-quantile1
print(iqrValue)
## Find the lower bound value and the higher bound value
lower_bound_val = quantile1 -(1.5 * iqrValue) 
upper_bound_val = quantile3 +(1.5 * iqrValue) 
# Anything that lies outside of lower and upper bound is an outlier
print(lower_bound_val.values,upper_bound_val.values)
import pandas as pd
from sklearn.datasets import load_boston
boston = load_boston()
x = boston.data
y = boston.target
columns = boston.feature_names
#create the dataframe
boston_df = pd.DataFrame(boston.data)
boston_df.columns = columns
boston_df.head(2)
import seaborn as sns
sns.boxplot(dataset);
sns.boxplot(boston_df['DIS']);
fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(boston_df['INDUS'], boston_df['TAX'])
ax.set_xlabel('Proportion of non-retail business acres per town')
ax.set_ylabel('Full-value property-tax rate per $10,000')
plt.show()
