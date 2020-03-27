
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))
import warnings
warnings.filterwarnings("ignore")
data=pd.read_csv('../input/StudentsPerformance.csv')
#read csv for analysis
#we'll see the first five lines.
data.head()
#we'll see the last five lines.
data.tail()
#random data 
data.sample(5)
data.sample(frac=0.1)
#it is a process that shows the property value in the data set and shows the numbers in the register values.
data.info()
data.iloc[:,0:3].dtypes
#It is a function that shows the analysis of numerical values.
data.describe()
#It shows the data types in the data set.
data.dtypes
#It is a function that shows the analysis of proximity values between data.
data.corr()
data.iloc[:,1:].corr()
#control data
data.isnull().values.any()
#all data control for null values
data.isnull().sum()
#show columns
for i,col in enumerate(data.columns):
    print(i+1,". column is ",col)
#rename columns
data.rename(columns=({'gender':'Gender','race/ethnicity':'Race/Ethnicity'
                     ,'parental level of education':'Parental_Level_of_Education'
                     ,'lunch':'Lunch','test preparation course':'Test_Preparation_Course'
                      ,'math score':'Math_Score','reading score':'Reading_Score'
                     ,'writing score':'Writing_Score'}),inplace=True)
#show columns
for i,col in enumerate(data.columns):
    print(i+1,". column is ",col)
#show count Gender
data['Gender'].value_counts()
#show Gender's unique
data['Gender'].unique()
#Gender show bar plot
sns.set(style='whitegrid')
ax=sns.barplot(x=data['Gender'].value_counts().index,y=data['Gender'].value_counts().values,palette="Blues_d",hue=['female','male'])
plt.legend(loc=8)
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.title('Show of Gender Bar Plot')
plt.show()
sns.barplot(x=data['Gender'].value_counts().index,y=data['Gender'].value_counts().values)
plt.title('Genders other rate')
plt.ylabel('Rates')
plt.legend(loc=0)
plt.show()
plt.figure(figsize=(7,7))
sns.barplot(x=data['Race/Ethnicity'].value_counts().index,
              y=data['Race/Ethnicity'].value_counts().values)
plt.xlabel('Race/Ethnicity')
plt.ylabel('Frequency')
plt.title('Show of Race/Ethnicity Bar Plot')
plt.show()
data.head()
plt.figure(figsize=(10,7))
sns.barplot(x = "Parental_Level_of_Education", y = "Writing_Score", hue = "Gender", data = data)
plt.xticks(rotation=45)
plt.show()
sns.barplot(x = "Parental_Level_of_Education", y = "Reading_Score", hue = "Gender", data = data)
plt.xticks(rotation=45)
plt.show()
sns.barplot(x = "Parental_Level_of_Education", y = "Math_Score", hue = "Gender", data = data)
plt.xticks(rotation=45)
plt.show()
plt.figure(figsize=(12,7))
sns.catplot(y="Gender", x="Math_Score",
                 hue="Parental_Level_of_Education",
                 data=data, kind="bar")
plt.title('for Parental Level Of Education Gender & Math_Score')
plt.show()
plt.figure(figsize=(10,10))
sns.catplot(x="Gender", y="Math_Score",
                 hue="Test_Preparation_Course",
                 data=data, kind="bar")
plt.title('for Test Preparation Course Gender & Math_Score')
plt.show()
ax = sns.barplot("Parental_Level_of_Education", "Writing_Score", data=data,
                  linewidth=2.5, facecolor=(1, 1, 1, 0),
                  errcolor=".2", edgecolor=".2")
plt.xticks(rotation=90)
plt.show()
# Draw a nested barplot to show survival for class and sex
data.head()
data['Test_Preparation_Course'].unique()
data_lunch_score=data[data['Test_Preparation_Course']=="completed"].groupby(data['Lunch']).Writing_Score.sum()
plt.title("Lunch - Free/reduced & standard")
sns.barplot(x=data_lunch_score.index,y=data_lunch_score.values)
plt.show()
f,ax=plt.subplots(figsize=(9,10))
sns.barplot(x=data['Gender'].value_counts().values,y=data['Gender'].value_counts().index,alpha=0.5,color='red',label='Gender')
sns.barplot(x=data['Race/Ethnicity'].value_counts().values,y=data['Race/Ethnicity'].value_counts().index,color='blue',alpha=0.7,label='Race/Ethnicity')
ax.legend(loc='upper right',frameon=True)
ax.set(xlabel='Gender , Race/Ethnicity',ylabel='Groups',title="Gender vs Race/Ethnicity ")
plt.show()
#Gender show point plot
data['Race/Ethnicity'].unique()
len(data[(data['Race/Ethnicity']=='group B')].Math_Score)
f,ax1=plt.subplots(figsize=(25,10))
sns.pointplot(x=np.arange(1,191),y=data[(data['Race/Ethnicity']=='group B')].Math_Score,color='lime',alpha=0.8)
sns.pointplot(x=np.arange(1,191),y=data[(data['Race/Ethnicity']=='group B')].Reading_Score,color='red',alpha=0.5)
#sns.pointplot(x=np.arange(1,191),y=data[(data['Race/Ethnicity']=='group B')].Math_Score,color='lime',alpha=0.8)
plt.xlabel('Group B index State')
plt.ylabel('Frequency')
plt.title('Group B Math Score & Reading_Score')
plt.xticks(rotation=90)
plt.grid()
plt.show()
plt.figure(figsi)
ax = sns.pointplot(x="Reading_Score", y="Math_Score", hue="Gender",data=data)
plt.xticks(rotation=90)
plt.show()
ax = sns.pointplot(x="Reading_Score", y="Writing_Score", hue="Gender",data=data,markers=["o", "x"],linestyles=["-", "--"])
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize=(10,10))
sns.jointplot(x=np.arange(1,191),y=data[(data['Race/Ethnicity']=='group B')].Math_Score,color='lime',alpha=0.8)
plt.xlabel('Group B index State')
plt.ylabel('Frequency')
plt.title('Group B Frequency Race/Ethnicity')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
plt.figure(figsize=(10,10))
sns.jointplot(x=np.arange(1,191),y=data[(data['Race/Ethnicity']=='group B')].Math_Score,color='lime',kind='hex',alpha=0.8)
plt.xlabel('Group B index State')
plt.ylabel('Frequency')
plt.title('Group B Frequency Race/Ethnicity')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
plt.figure(figsize=(10,10))
sns.jointplot(x=np.arange(1,191),y=data[(data['Race/Ethnicity']=='group B')].Math_Score,color='lime',space=0,kind='kde')
plt.xlabel('Group B index State')
plt.ylabel('Frequency')
plt.title('Group B Frequency Race/Ethnicity')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
#Gender show point plot
data['Race/Ethnicity'].unique()
len(data[(data['Race/Ethnicity']=='group B')].Math_Score)
plt.figure(figsize=(10,10))
sns.jointplot(x=np.arange(1,191),y=data[(data['Race/Ethnicity']=='group B')].Reading_Score,color='k').plot_joint(sns.kdeplot, zorder=0, n_levels=6)
plt.xlabel('Group B index State')
plt.ylabel('Frequency')
plt.title('Group B Math Score & Reading_Score')
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize=(10,10))
sns.jointplot(x=np.arange(1,191),y=data[(data['Race/Ethnicity']=='group B')].Reading_Score,color='lime',alpha=0.8)
plt.xlabel('Group B index State')
plt.ylabel('Frequency')
plt.title('Group B Frequency Race/Ethnicity')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
labels=data['Race/Ethnicity'].value_counts().index
colors=['blue','red','yellow','green','brown']
explode=[0,0,0.1,0,0]
values=data['Race/Ethnicity'].value_counts().values

#visualization
plt.figure(figsize=(7,7))
plt.pie(values,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%')
plt.title('Race/Ethnicity According Analysis',color='black',fontsize=10)
plt.show()

plt.figure(figsize=(4,4))
labels=['Math Score', 'Reading Score', 'Writing Score']
colors=['blue','red','yellow']
explode=[0,0,0.1]
values=[data.Math_Score.mean(),data.Reading_Score.mean(),data.Writing_Score.mean()]

plt.pie(values,labels=labels,colors=colors,explode=explode,autopct='%1.1f%%',shadow=True)
plt.legend(['Math Score', 'Reading Score', 'Writing Score'] , loc=3)
plt.axis('equal')
plt.tight_layout()
plt.show()
data.groupby('Race/Ethnicity')['Reading_Score'].mean()
# Data to plot
labels = 'group A', 'group B', 'group C', 'group D','group E'
sizes = data.groupby('Race/Ethnicity')['Reading_Score'].mean().values
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0.1, 0, 0, 0,0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Reading Score for Every Race/Ethnicity Mean')
plt.axis('equal')
plt.show()
sns.lmplot(x='Math_Score',y='Reading_Score',data=data)
plt.xlabel('Math Score')
plt.ylabel('Reading Score')
plt.title('Math Score vs Reading Score')
plt.show()
sns.lmplot(x='Math_Score',y='Writing_Score',hue='Gender',data=data)
plt.xlabel('Math Score')
plt.ylabel('Writing Score')
plt.title('Math Score vs Writing Score')
plt.show()
sns.lmplot(x='Math_Score',y='Writing_Score',hue='Gender',data=data,markers=['x','o'])
plt.xlabel('Math Score')
plt.ylabel('Writing Score')
plt.title('Math Score vs Writing Score')
plt.show()
sns.set(style="dark")
rs = np.random.RandomState(50)

# Set up the matplotlib figure
f, axes = plt.subplots(3, 3, figsize=(9, 9), sharex=True, sharey=True)

# Rotate the starting point around the cubehelix hue circle
for ax, s in zip(axes.flat, np.linspace(0, 3, 10)):

    # Create a cubehelix colormap to use with kdeplot
    cmap = sns.cubehelix_palette(start=s, light=1, as_cmap=True)

    # Generate and plot a random bivariate dataset
    x, y = rs.randn(2, 50)
    sns.kdeplot(x, y, cmap=cmap, shade=True, cut=5, ax=ax)
    ax.set(xlim=(-3, 3), ylim=(-3, 3))

f.tight_layout()
plt.show()
sns.kdeplot(data['Math_Score'])
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Math Score Kde Plot System Analysis')
plt.show()
sns.kdeplot(data['Reading_Score'],shade=True,color='r')
sns.kdeplot(data['Writing_Score'],shade=True,color='b')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Reading Score vs Writing Score Kde Plot System Analysis')
plt.show()
sns.kdeplot(data['Reading_Score'],data['Writing_Score'])
plt.show()
sns.kdeplot(data['Reading_Score'],data['Writing_Score'],shade=True)
plt.show()
sns.kdeplot(data['Math_Score'],bw=.15)
plt.xlabel('Math Score')
plt.ylabel('Frequency')
plt.title('Math Score Show Kde Plot')
plt.show()
sns.kdeplot(data['Reading_Score'],data['Writing_Score'],cmap='Reds',shade=True,shade_lowest=False)
sns.kdeplot(data['Writing_Score'],data['Reading_Score'],cmap='Blues',shade=True,shade_lowest=False)
plt.show()
sns.violinplot(data['Math_Score'])
plt.xlabel('Math Score')
plt.ylabel('Frequency')
plt.title('Violin Math Score Show')
plt.show()
sns.violinplot(x=data['Race/Ethnicity'],y=data['Math_Score'])
plt.show()
sns.violinplot(data['Gender'],y=data['Reading_Score'],hue=data['Race/Ethnicity'],palette='muted')
plt.legend(loc=10)
plt.show()
sns.violinplot(data['Race/Ethnicity'],data['Writing_Score'],
               hue=data['Gender'],palette='muted',split=True)
plt.legend(loc=8)
plt.show()
sns.violinplot(data['Parental_Level_of_Education'],data['Math_Score'],hue=data['Gender'],dodge=False)
plt.xticks(rotation=90)
plt.show()
sns.heatmap(data.corr())
plt.show()
sns.heatmap(data.corr(),vmin=0,vmax=1)
plt.show()
sns.heatmap(data.corr(),annot=True)
plt.show()
# Compute the correlation matrix
corr = data.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
sns.heatmap(data.corr(),cmap='YlGnBu')
plt.show()
sns.axes_style("white")
mask = np.zeros_like(data.corr())
mask[np.triu_indices_from(mask)] = True
sns.heatmap(data.corr(),vmax=.3,mask=mask,square=True)
plt.show()
sns.set(style='whitegrid')
sns.boxplot(data['Math_Score'])
plt.show()
sns.boxplot(x=data['Gender'],y=data['Math_Score'])
plt.show()
sns.boxplot(x=data['Race/Ethnicity'],y=data['Writing_Score'],hue=data['Gender'],palette="Set3")
plt.show()
sns.boxplot(data['Math_Score'],orient='h',palette='Set2')
plt.show()
sns.boxenplot(x="Race/Ethnicity", y="Writing_Score",
              color="b",
              scale="linear", data=data)
plt.show()
sns.boxplot(x=data['Race/Ethnicity'],y=data['Writing_Score'],hue=data['Gender'],dodge=False)
plt.show()
sns.boxplot(x=data['Parental_Level_of_Education'],y=data['Math_Score'])
plt.xticks(rotation=90)
sns.swarmplot(x=data['Parental_Level_of_Education'],y=data['Math_Score'],color=".25")
plt.xticks(rotation=90)
plt.show()
sns.set(style='whitegrid')
sns.swarmplot(x=data['Math_Score'])
plt.show()
sns.set(style="whitegrid")

sns.swarmplot(y=data["Writing_Score"],color='red')
sns.swarmplot(y=data["Reading_Score"],color='blue')
plt.title('Writing & Reading Scores')
plt.show()
sns.swarmplot(x=data['Lunch'],y=data['Reading_Score'])
plt.show()
sns.swarmplot(x=data['Test_Preparation_Course'],y=data['Math_Score'],hue=data['Gender'])
plt.show()
sns.swarmplot(x=data['Test_Preparation_Course'],y=data['Writing_Score'],hue=data['Race/Ethnicity'],palette='Set2',dodge=True)
plt.show()
sns.boxplot(x=data['Lunch'],y=data['Math_Score'],whis=np.inf)
sns.swarmplot(x=data['Lunch'],y=data['Math_Score'],color='.2')
plt.show()
sns.violinplot(x=data['Test_Preparation_Course'],y=data['Reading_Score'],inner=None)
sns.swarmplot(x=data['Test_Preparation_Course'],y=data['Reading_Score'],color='white',edgecolor='gray')
plt.show()
sns.pairplot(data)
plt.show()
sns.pairplot(data,diag_kind='kde')
plt.show()
sns.pairplot(data,kind='reg')
plt.show()
sns.pairplot(data, diag_kind="kde", markers="+",
                  plot_kws=dict(s=50, edgecolor="b", linewidth=1),
                  diag_kws=dict(shade=True))
plt.show()
sns.pairplot(data, hue="Reading_Score")
plt.show()
data.columns
sns.countplot(data['Race/Ethnicity'])
plt.show()
sns.countplot(data['Gender'])
plt.show()
sns.countplot(data['Race/Ethnicity'],hue=data['Gender'])
plt.show()
sns.countplot(y=data['Parental_Level_of_Education'],palette="Set3",hue=data['Gender'])
plt.legend(loc=4)
plt.show()
sns.countplot(x=data['Lunch'],facecolor=(0,0,0,0),linewidth=5,edgecolor=sns.color_palette('dark',3))
plt.show()
sns.countplot(x="Parental_Level_of_Education", hue="Lunch",
                 data=data)
plt.xticks(rotation=45)
plt.show()
sns.FacetGrid(data,col='Gender',row='Gender')
plt.tight_layout()
plt.show()
g=sns.FacetGrid(data,col='Race/Ethnicity',row='Race/Ethnicity')
g=g.map(plt.hist,"Math_Score",bins=np.arange(0,65,3),color='r')
plt.show()
g=sns.FacetGrid(data,col='Lunch',row='Lunch')
g=(g.map(plt.scatter,"Reading_Score",'Writing_Score',edgecolor='w').add_legend())
plt.tight_layout()
plt.show()
g = sns.FacetGrid(data, col="Parental_Level_of_Education", col_wrap=3)
g = g.map(plt.plot, "Reading_Score", "Writing_Score", marker=".")
plt.show()
sns.set()

# Generate an example radial datast
r = np.linspace(0, 10, num=100)
df = pd.DataFrame({'r': r, 'slow': r, 'medium': 2 * r, 'fast': 4 * r})

# Convert the dataframe to long-form or "tidy" format
df = pd.melt(df, id_vars=['r'], var_name='speed', value_name='theta')

# Set up a grid of axes with a polar projection
g = sns.FacetGrid(df, col="speed", hue="speed",
                  subplot_kws=dict(projection='polar'), height=4.5,
                  sharex=False, sharey=False, despine=False)

# Draw a scatterplot onto each axes in the grid
g.map(sns.scatterplot, "theta", "r")
plt.show()
pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
g = sns.FacetGrid(data, row="Writing_Score", hue="Reading_Score", aspect=15, height=.5, palette=pal)

# Draw the densities in a few steps
g.map(sns.kdeplot, "Writing_Score", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)
g.map(sns.kdeplot, "Reading_Score", clip_on=False, color="w", lw=2, bw=.2)
g.map(plt.axhline, y=0, lw=2, clip_on=False)
plt.show()
sns.stripplot(x=data['Reading_Score'])
plt.show()
sns.stripplot(x="Parental_Level_of_Education",y='Writing_Score',data=data)
plt.xticks(rotation=45)
plt.show()
sns.stripplot(x="Gender",y='Writing_Score',jitter=True,data=data)
plt.xticks(rotation=45)
plt.show()
sns.stripplot(x="Lunch",y='Reading_Score',jitter=0.05,data=data)
plt.xticks(rotation=45)
plt.show()
sns.stripplot(x='Test_Preparation_Course',y='Reading_Score',hue='Gender',jitter=True,data=data)
plt.show()
sns.stripplot(x='Race/Ethnicity',y='Math_Score',hue='Lunch',jitter=True,dodge=True,palette="Set2",data=data)
plt.show()
sns.stripplot(x='Lunch',y='Math_Score',hue='Lunch',jitter=True,dodge=True,size=20,marker='D',edgecolor='gray',alpha=.25,palette="Set2",data=data)
plt.legend(loc=10)
plt.show()
sns.factorplot(x="Lunch", y="Math_Score", hue="Gender", data=data)
plt.show()
sns.factorplot(x="Gender", y="Reading_Score", hue="Lunch", kind='violin',data=data)
plt.show()
sns.factorplot(x="Race/Ethnicity", y="Math_Score", hue="Gender",col='Lunch',data=data)
plt.show()
g=sns.factorplot(x="Parental_Level_of_Education", y="Writing_Score", hue="Lunch",
                col="Gender", data=data)
plt.tight_layout()
plt.show()
ax = sns.distplot(data['Reading_Score'], rug=True, hist=False)
plt.show()
ax = sns.distplot(data['Writing_Score'], vertical=True)
plt.show()
ax = sns.distplot(data['Math_Score'])
plt.show()
ax = sns.distplot(data['Reading_Score'], color="y")
plt.show()
sns.set(style="white", palette="muted", color_codes=True)
rs = np.random.RandomState(10)

f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)
sns.despine(left=True)

# Generate a random univariate dataset
d = rs.normal(size=100)

# Plot a simple histogram with binsize determined automatically
sns.distplot(d, kde=False, color="b", ax=axes[0, 0])

# Plot a kernel density estimate and rug plot
sns.distplot(d, hist=False, rug=True, color="r", ax=axes[0, 1])

# Plot a filled kernel density estimate
sns.distplot(d, hist=False, color="g", kde_kws={"shade": True}, ax=axes[1, 0])

# Plot a historgram and kernel density estimate
sns.distplot(d, color="m", ax=axes[1, 1])

plt.setp(axes, yticks=[])
plt.tight_layout()
plt.show()
data.columns
data[data['Gender']=='male']['Math_Score'].value_counts().sort_index().plot.line(color='b')
data[data['Gender']=='female']['Math_Score'].value_counts().sort_index().plot.line(color='r')
plt.xlabel('Math_Score')
plt.ylabel('Frequency')
plt.title('Math_Score vs Frequency')
plt.show()
sns.lineplot(x='Math_Score',y='Reading_Score',data=data)
plt.show()
sns.lineplot(x='Reading_Score',y='Writing_Score',hue='Lunch',data=data)
plt.show()
sns.lineplot(x='Writing_Score',y='Reading_Score',data=data,hue='Lunch',
            style='Gender')
plt.show()
female_filter=data[data['Gender']=='female']
sns.lineplot(x='Reading_Score',y='Writing_Score',data=female_filter,
            hue='Lunch',style='Test_Preparation_Course',dashes=False)
plt.show()
sns.lineplot(x="Math_Score", y="Writing_Score", hue="Lunch",err_style="bars", ci=68, data=data)
plt.show()
ax = sns.lineplot(x="Math_Score", y="Reading_Score", hue="Test_Preparation_Course",
                   units="Lunch", estimator=None, lw=1,
                   data=data.query("Gender == 'male'"))
ax = sns.lineplot(x="Math_Score", y="Writing_Score",
                   hue="Lunch", style="Gender",
                   data=data)
plt.show()
data.groupby('Gender')[['Writing_Score','Reading_Score']].mean()
x=data[data.Parental_Level_of_Education=='bachelor\'s degree'].groupby('Race/Ethnicity')['Math_Score'].count()
x
sns.lineplot(data=x,color='coral',label='Race/Ethnicity')
plt.show()
f, ax = plt.subplots(figsize=(6.5, 6.5))
sns.despine(f, left=True, bottom=True)
sns.scatterplot(x="Math_Score", y="Reading_Score",
                hue="Gender", size="Gender",data=data)
plt.show()
data.head()

# Draw the full plot
sns.clustermap(data.corr(), center=0, cmap="vlag",
               linewidths=.75, figsize=(13, 13))
sns.set(style="white")
# Plot miles per gallon against horsepower with other semantics
sns.relplot(x="Reading_Score",y="Math_Score",hue="Gender",
            sizes=(40, 400), alpha=.5, palette="muted",
            height=6, data=data)
plt.show()
f, ax = plt.subplots(figsize=(6.5, 6.5))
sns.despine(f, left=True, bottom=True)
sns.scatterplot(x="Reading_Score", y="Writing_Score",
                hue="Lunch", size="Gender",data=data)
plt.show()
data.head()
f, ax = plt.subplots(figsize=(6.5, 6.5))
sns.despine(f, left=True, bottom=True)
sns.scatterplot(x="Reading_Score", y="Writing_Score",
                hue="Lunch", size="Gender",data=data[data['Parental_Level_of_Education']=="some college"])
plt.show()
data[np.logical_and(data['Race/Ethnicity']=='group A',data['Parental_Level_of_Education']=='some college')].head()
f, ax = plt.subplots(figsize=(6.5, 6.5))
sns.despine(f, left=True, bottom=True)
sns.scatterplot(x="Reading_Score", y="Writing_Score",
                hue="Lunch", size="Gender",data=data[np.logical_and(data['Race/Ethnicity']=='group C',data['Parental_Level_of_Education']=='some college')])
plt.show()