
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# load Haberman dataset into pandas dataframe
haber = pd.read_csv("/kaggle/input/habermans-survival-data-set/haberman.csv", names = ["age", "year", "nodes", "status"])
haber
# number of data points and features
haber.shape
# number of classes and data points for each class
haber["status"].value_counts()
# high level statistics
haber.describe()
# Histogram and PDF for age
sns.FacetGrid(haber, hue = "status", height = 5)\
   .map(sns.distplot, "age")\
   .add_legend();
plt.title("Histogram for Age")
plt.ylabel("density")
plt.show();
# Histogram and PDF for year
sns.FacetGrid(haber, hue = "status", height = 5)\
   .map(sns.distplot, "year")\
   .add_legend();
plt.title("Histogram for Year")
plt.ylabel("density")
plt.show();
# Histogram and PDF for nodes
sns.FacetGrid(haber, hue = "status", height = 5)\
   .map(sns.distplot, "nodes")\
   .add_legend();
plt.title("Histogram for Nodes")
plt.ylabel("density")
plt.show();
# 1.1 Computing PDF and CDF for "nodes" feature, status = 1
haber_1 = haber.loc[haber["status"] == 1]
counts, bin_edges = np.histogram(haber_1["nodes"], bins = 23, density = True)
print(bin_edges)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
print(pdf)
print(cdf)
plt.plot(bin_edges[1:], pdf, label = "pdf")
plt.plot(bin_edges[1:], cdf, label = "cdf")
plt.xlabel("nodes of status 1")
plt.ylabel("probability of # of patients")
plt.legend();
plt.title("PDF and CDF for nodes with status 1")
plt.show;
# 1.2 Computing PDF and CDF for "nodes" feature, status = 2
haber_2 = haber.loc[haber["status"] == 2]
counts, bin_edges = np.histogram(haber_2["nodes"], bins = 26, density = False)
print(bin_edges)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
print(pdf)
print(cdf)
plt.plot(bin_edges[1:], pdf, label = "pdf")
plt.plot(bin_edges[1:], cdf, label = "cdf")
plt.xlabel("nodes of status 2")
plt.ylabel("probability of # of patients")
plt.legend();
plt.title("PDF and CDF for nodes with status 2")
plt.show;
# 1.3 Computing PDF and CDF for "nodes" feature
counts, bin_edges = np.histogram(haber["nodes"], bins = 26, density = False)
print(bin_edges)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
print(pdf)
print(cdf)
plt.plot(bin_edges[1:], pdf, label = "pdf")
plt.plot(bin_edges[1:], cdf, label = "cdf")
plt.xlabel("nodes of status 1 & 2 combined")
plt.ylabel("probability of # of patients")
plt.legend();
plt.title("PDF and CDF for nodes with status 1 & 2")
plt.show;
# 2.1 Computing PDF and CDF for "age" feature
counts, bin_edges = np.histogram(haber["age"], bins = 10, density = False)
print(bin_edges)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
print(pdf)
print(cdf)
plt.plot(bin_edges[1:], pdf, label = "pdf")
plt.plot(bin_edges[1:], cdf, label = "cdf")
plt.xlabel("age")
plt.ylabel("probability of # of patients")
plt.legend();
plt.title("PDF and CDF for Age")
plt.show;
# 2.2 Computing PDF and CDF for "year" feature
counts, bin_edges = np.histogram(haber["year"], bins = 10, density = False)
print(bin_edges)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
print(pdf)
print(cdf)
plt.plot(bin_edges[1:], pdf, label = "pdf")
plt.plot(bin_edges[1:], cdf, label = "cdf")
plt.xlabel("year")
plt.ylabel("probability of # of patients")
plt.legend();
plt.title("PDF and CDF for Year")
plt.show;
#Mean and standard deviation
print("Mean:")
print(np.mean(haber_1["nodes"]))
print(np.mean(haber_2["nodes"]))

print(np.mean(haber["age"]))


print("\nStandard Deviation:")
print(np.std(haber_1["nodes"]))
print(np.std(haber_2["nodes"]))
# Median, Percentile, quantile, MAD
print("Median:")
print(np.median(haber_1["nodes"]))
print(np.median(haber_2["nodes"]))

print("\nQuantiles:")
print(np.percentile(haber_1["nodes"],np.arange(0, 100, 25)))
print(np.percentile(haber_2["nodes"],np.arange(0, 100, 25)))

print("\n20th Percentile range")
print(np.percentile(haber_1["nodes"],np.arange(0, 100, 20)))
print(np.percentile(haber_2["nodes"],np.arange(0, 100, 20)))

from statsmodels import robust
print("\nMedian Absolute Deviation:")
print(robust.mad(haber_1["nodes"]))
print(robust.mad(haber_2["nodes"]))
# Box plot and Whiskers

# Setting handles for the legend.
import matplotlib.patches as mpatches
blue_patch = mpatches.Patch(color = "steelblue", label = "1")
orange_patch = mpatches.Patch(color = "orange", label = "2")

# Box plot and whiskers for nodes
sns.boxplot(x = "status", y = "nodes", data = haber)
plt.title("Box plot for Nodes")
plt.legend(title = "status", handles = [blue_patch, orange_patch])
plt.show();
# Box plot and whiskers for age
sns.boxplot(x = "status", y = "age", data = haber)
plt.title("Box plot for Age")
plt.legend(title = "status", handles = [blue_patch, orange_patch])
plt.show();
# Box plot and whiskers for year
sns.boxplot(x = "status", y = "year", data = haber)
plt.title("Box plot for Year")
plt.legend(title = "status", handles = [blue_patch, orange_patch])
plt.show();
#Violin plots
import matplotlib.patches as mpatches
blue_patch = mpatches.Patch(color = "steelblue", label = "1")
orange_patch = mpatches.Patch(color = "orange", label = "2")

# Violin plot for nodes
sns.violinplot(x = "status", y = "nodes", data = haber)
plt.title("Violin plot for Nodes")
plt.legend(title = "status", handles = [blue_patch, orange_patch])
plt.show();
# Violin plot for age 
sns.violinplot(x = "status", y = "age", data = haber)
plt.title("Violin plot for Age")
plt.legend(title = "status", handles = [blue_patch, orange_patch])
plt.show();
# Violin plot for year
sns.violinplot(x = "status", y = "year", data = haber)
plt.title("Violin plot for Year")
plt.legend(title = "status", handles = [blue_patch, orange_patch])
plt.show();
# Scatter plot of nodes and age
sns.set_style("whitegrid")
sns.FacetGrid(haber, hue = "status", height = 5)\
   .map(plt.scatter, "nodes", "age")\
   .add_legend()
plt.title("Scatter plot of nodes and age")
plt.show();
# Pair plots
plt.close();
#plt.title("Pair plots for 3 features in the dataset")
sns.set_style("whitegrid");
sns.pairplot(haber, hue = "status", vars = ["age", "year", "nodes"], height = 3);
plt.suptitle("Pair plots for the 3 features in the dataset")
plt.show()