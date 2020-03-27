
'''Ignore deprecation and future warnings.'''
def suppress_warnings():
    import warnings
    warnings.filterwarnings('ignore', category = DeprecationWarning) 
    warnings.filterwarnings('ignore', category = FutureWarning) 

'''Import required modules.'''
import numpy as np               # For linear algebra
import pandas as pd              # For data manipulation
import matplotlib.pyplot as plt  # For 2D visualization
import seaborn as sns            
from scipy import stats          # For statistics

'''Plotly visualization .'''
import plotly.offline as py
from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go
init_notebook_mode(connected = True) # Required to use plotly offline in jupyter notebook
'''Customize visualization.'''
plt.style.use('bmh')                    # Use bmh's style for plotting
sns.set_style({'axes.grid':False})      # Remove gridlines

'''Display markdown formatted output like bold, italic bold etc.'''
from IPython.display import Markdown
def bold(string):
    display(Markdown(string))
'''Read and preview the train data from csv file.'''
train = pd.read_csv('../input/train.csv')
bold('**Preview of Train Data:**')
display(train.head(2))

'''Read and preview the test from csv file.'''
test = pd.read_csv('../input/test.csv')
bold('**Preview of Test Data:**')
display(test.head(2))
'''Merge train and test data together. This eliminates the hassle of handling train and test data seperately for various analysis.'''
merged = pd.concat([train, test], sort = False)
bold('**Preview of Merged Data:**')
display(merged.head(5))
'''Shape of the combined data'''
bold('**Shape of the Merged Data:**')
display(merged.shape)

'''Variables in the combined data'''
bold('**Name of the Variables in merged data:**')
display(merged.columns)
'''Pandas data types for our different variables.'''
bold('**Data Types of Our Variables:**')
display(merged.dtypes)
''' #1.Function for displaying bar labels in absolute scale.'''
def abs_bar_labels():
    font_size = 15
    plt.ylabel('Absolute Frequency', fontsize = font_size)
    plt.xticks(rotation = 0, fontsize = font_size)
    plt.yticks([])
    
    # Set individual bar lebels in absolute number
    for x in ax.patches:
        ax.annotate(x.get_height(), 
        (x.get_x() + x.get_width()/2., x.get_height()), ha = 'center', va = 'center', xytext = (0, 7), 
        textcoords = 'offset points', fontsize = font_size, color = 'black')
    
'''#2.Function for displaying bar lebels in relative scale.'''
def pct_bar_labels():
    font_size = 15
    plt.ylabel('Relative Frequency (%)', fontsize = font_size)
    plt.xticks(rotation = 0, fontsize = font_size)
    plt.yticks([]) 
    
    # Set individual bar lebels in proportional scale
    for x in ax1.patches:
        ax1.annotate(str(x.get_height()) + '%', 
        (x.get_x() + x.get_width()/2., x.get_height()), ha = 'center', va = 'center', xytext = (0, 7), 
        textcoords = 'offset points', fontsize = font_size, color = 'black')
         
'''#3.Function to create a dataframe of absolute and relative frequency of each variable. And plot absolute and relative frequency.'''
def absolute_and_relative_freq(variable):
    global  ax, ax1 
    # Dataframe of absolute and relative frequency
    absolute_frequency = variable.value_counts()
    relative_frequency = round(variable.value_counts(normalize = True)*100, 2)
    # Was multiplied by 100 and rounded to 2 decimal points for percentage.
    df = pd.DataFrame({'Absolute Frequency':absolute_frequency, 'Relative Frequency(%)':relative_frequency})
    print('Absolute & Relative Frequency of',variable.name,':')
    display(df)
    
    # This portion plots absolute frequency with bar labeled.
    fig_size = (18,5)
    font_size = 15
    title_size = 18
    ax =  absolute_frequency.plot.bar(title = 'Absolute Frequency of %s' %variable.name, figsize = fig_size)
    ax.title.set_size(title_size)
    abs_bar_labels()  # Displays bar labels in abs scale.
    plt.show()
    
    # This portion plots relative frequency with bar labeled.
    ax1 = relative_frequency.plot.bar(title = 'Relative Frequency of %s' %variable.name, figsize = fig_size)
    ax1.title.set_size(title_size)
    pct_bar_labels() # Displays bar labels in relative scale.
    plt.show()
'''Plot and count the number of survivors and victims in absolute and relative scale in the tragedy.'''
absolute_and_relative_freq(merged.Survived)
'''Plot and count the absolute and relative frequency of Sex.'''
absolute_and_relative_freq(merged.Sex)
'''Plot and count the absolute and relative frequency of Pclass.'''
absolute_and_relative_freq(merged.Pclass)
'''Plot and count the absolute and relative frequency of Embarked.'''
absolute_and_relative_freq(merged.Embarked)
'''Absolute frequency of Cabin.'''
abs_freq_cabin = merged.Cabin.value_counts(dropna = False)
bold('**Categories of Cabin:**')
display(abs_freq_cabin.head())

"""As frequency of Cabin isn't what we expected, let's count total categories in Cabin."""
bold('**Total Categories in Cabin:**')
display(abs_freq_cabin.count())
'''Finally preview the variable Cabin to see what is causing the irregularity.'''
bold('**Preview of Cabin:**')
display(merged.Cabin.head(7))
'''Count total categories in Name.'''
bold('**Total Categories in Name:**')
display(merged.Name.value_counts().count())
"""Let's finally check the what's inside the variable Name."""
bold('**Preview Name:**')
display(merged.Name.head())
'''Count total groups in variable Ticket.'''
bold('**Total Groups in Ticket:**')
display(merged.Ticket.value_counts().count())
'''Lets investigate Ticket.'''
bold('**Preview of Ticket:**')
display(merged.Ticket.head())
'''Plot and count the absolute and relative frequency of SibSp.'''
absolute_and_relative_freq(merged.SibSp)
'''Plot and count the absolute and relative frequency of Parch.'''
absolute_and_relative_freq(merged.Parch)
'''#1.Plot histogram.'''
def histogram(variable):
    global ax
    font_size = 15
    fig_size = (18, 7)
    title_size = 18
    ax = variable.plot.hist(figsize = fig_size, color = 'teal')
    plt.xlabel('%s' %variable.name, fontsize = font_size)
    plt.xticks(fontsize = font_size)
    plt.title('%s' %variable.name + ' Distribution with Histogram', fontsize = title_size)
    abs_bar_labels()
    plt.show()
    
'''#2.Plot density plot .'''
def density_plot(variable):
    fig_size = (18, 7)
    font_size = 15
    title_size = 18
    plt.figure(figsize = fig_size)
    variable.plot.hist(density = True, color = 'orchid')
    variable.plot.kde(style = 'k--')
    plt.xlabel('%s'%variable.name, fontsize = font_size)
    plt.ylabel('Density', fontsize = font_size)
    plt.xticks(fontsize = font_size)
    plt.yticks(fontsize = font_size)
    plt.title('%s ' %variable.name + 'Distribution with Density Plot & Histogram', fontsize = title_size)
    plt.show()
    
'''#3.Calculate descriptive statistics.'''
def summary_stats(variable):
    stats = variable.describe()
    skew = pd.Series(variable.skew(), index = ['skewness'])
    df_stats = pd.DataFrame(pd.concat([skew, stats], sort = False), columns = [variable.name])
    df_stats.index.name = 'Stats'
    display(df_stats)
'''Plot histogram of Fare.'''
histogram(merged.Fare)
'''Plot density plot of Fare.'''
density_plot(merged.Fare)
'''Calculate summary statistics of Fare.'''
bold('**Summary Stats of Fare:**')
summary_stats(merged.Fare)
'''Plot histogram of Age.'''
histogram(merged.Age)
'''Plot density plot of Age and calculate summary statistics.'''
density_plot(merged.Age)
bold('**Summary Stats of Age:**')
summary_stats(merged.Age)
'''What does passengerId contain?'''
display(merged.PassengerId.head())
"""Let's preview the Cabin again."""
bold('**Preview of Cabin:**')
display(merged.Cabin.head())

"""It seems Cabin contains some missing values. Let's count them."""
bold('**Missing Values in Cabin:**')
display(merged.Cabin.isnull().sum())
'''Total categories in Cabin before processing.'''
bold('**Total Categories in Cabin before Processing:**')
display(merged.Cabin.value_counts(dropna = False).count())
"""Flag all the NaNs of Cabin as 'X'."""
merged.Cabin.fillna(value = 'X', inplace = True)
'''Keep only the 1st character where Cabin is alphanumerical.'''
merged.Cabin = merged.Cabin.apply( lambda x : x[0])
bold('**Cabin Categories after Processing:**')
display(merged.Cabin.value_counts())
'''After processing, we can visualize the absolute and relative frequency of newly transformed Cabin variable.'''
absolute_and_relative_freq(merged.Cabin)
"""Lets see what's inside the Name."""
display(merged.Name.head(8))
'''Create a new variable Title that extracts titles from Name.'''
merged['Title'] = merged.Name.str.extract('([A-Za-z]+)\.')
'''See how it works.'''
display(merged.Title.head())
'''Count the extracted categories of Title from Name.'''
display(merged.Title.value_counts())
'''Create a bucket Officer and put Dr, Rev, Col, Major, Capt titles into it.'''
merged.Title.replace(to_replace = ['Dr', 'Rev', 'Col', 'Major', 'Capt'], value = 'Officer', inplace = True)

'''Put Dona, Jonkheer, Countess, Sir, Lady, Don in bucket Aristocrat.'''
merged.Title.replace(to_replace = ['Dona', 'Jonkheer', 'Countess', 'Sir', 'Lady', 'Don'], value = 'Aristocrat', inplace = True)

'''Finally Replace Mlle and Ms with Miss. And Mme with Mrs.'''
merged.Title.replace({'Mlle':'Miss', 'Ms':'Miss', 'Mme':'Mrs'}, inplace = True)
"""let's see how Title looks now"""
display(merged.Title.head())
'''After processing, visualise and count absolute and relative frequency of transformed Title.'''
absolute_and_relative_freq(merged.Title)
'''Merge SibSp and Parch to create a variable Family_size.'''
merged['Family_size'] = merged.SibSp + merged.Parch + 1  # Adding 1 for single person
bold('**Categoiries in Family_size:**')
display(merged.Family_size.value_counts())
'''Create buckets of single, small, medium, and large and then put respective values into them.'''
merged.Family_size.replace(to_replace = [1], value = 'single', inplace = True)
merged.Family_size.replace(to_replace = [2,3], value = 'small', inplace = True)
merged.Family_size.replace(to_replace = [4,5], value = 'medium', inplace = True)
merged.Family_size.replace(to_replace = [6, 7, 8, 11], value = 'large', inplace = True)
'''After processing, visualise and count the absolute and relative frequency of engineered Family_size.'''
absolute_and_relative_freq(merged.Family_size)
"""Let's preview the variable Ticket first."""
display(merged.Ticket.head())
'''Assign N if there is only number and no character. If there is a character, extract the character only.'''
ticket = []
for x in list(merged.Ticket):
    if x.isdigit():
        ticket.append('N')
    else:
        ticket.append(x.replace('.','').replace('/','').strip().split(' ')[0])
        
'''Swap values'''
merged.Ticket = ticket
'''Now Ccunt the categories in Ticket.'''
bold('**Categories of Ticket:**')
display(merged.Ticket.value_counts())
'''Keep only the 1st character of Ticket to further reduce the Ticket categories.'''
merged.Ticket = merged.Ticket.apply(lambda x : x[0])
bold('**Ticket after Processing:**')
display(merged.Ticket.value_counts())
'''After processing, visualise and count the absolute and relative frequency of updated Ticket.'''
absolute_and_relative_freq(merged.Ticket)
'''Create a function to count total outliers. And plot variables with and without outliers.'''
def outliers(variable):
    global filtered
    # Calculate 1st, 3rd quartiles and iqr.
    q1, q3 = variable.quantile(0.25), variable.quantile(0.75)
    iqr = q3 - q1
    
    # Calculate lower fence and upper fence for outliers
    l_fence, u_fence = q1 - 1.5*iqr , q3 + 1.5*iqr   # Any values less than l_fence and greater than u_fence are outliers.
    
    # Observations that are outliers
    outliers = variable[(variable<l_fence) | (variable>u_fence)]
    print('Total Outliers of', variable.name,':', outliers.count())
    
    # Drop obsevations that are outliers
    filtered = variable.drop(outliers.index, axis = 0)

    # Create subplots
    out_variables = [variable, filtered]
    out_titles = [' Distribution with Outliers', ' Distribution Without Outliers']
    title_size = 25
    font_size = 18
    plt.figure(figsize = (25, 15))
    for ax, outlier, title in zip(range(1,3), out_variables, out_titles):
        plt.subplot(2, 1, ax)
        sns.boxplot(outlier).set_title('%s' %outlier.name + title, fontsize = title_size)
        plt.xticks(fontsize = font_size)
        plt.xlabel('%s' %outlier.name, fontsize = font_size)
'''Count total outliers of Age. Plot Age with and without outliers.'''
outliers(merged.Age)
'''Count total outliers of Fare. Plot Fare with and without outliers.'''
outliers(merged.Fare)
'''Function to plot scatter plot'''
def scatter_plot(x, y, title, yaxis, size, c_scale):
    trace = go.Scatter(
    x = x,
    y = y,
    mode = 'markers',
    marker = dict(color = y, size = size, showscale = True, colorscale = c_scale))
    layout = go.Layout(hovermode= 'closest', title = title, yaxis = dict(title = yaxis))
    fig = go.Figure(data = [trace], layout = layout)
    return iplot(fig)
'''Get and plot only the missing columns with their missing values.'''
missing_columns = len(merged) - merged.loc[:, np.sum(merged.isnull())>0].count()
x = missing_columns.index
y = missing_columns
title = 'Features with Missing Values'
scatter_plot(x, y, title, 'Missing Values', 30, 'Rainbow')
"""Let's count the missing values for each variable."""
bold('**Missing Values of Each Feature:**')
display(merged.isnull().sum())
'''Impute missing values of Embarked. Embarked is a categorical variable where S is the most frequent.'''
merged.Embarked.fillna(value = 'S', inplace = True)

'''Impute missing values of Fare. Fare is a numerical variable with outliers. Hence it will be imputed by median.'''
merged.Fare.fillna(value = merged.Fare.median(), inplace = True)
"""Create a boxplot to view the variables correlated with Age. First extract the variables we're interested in."""
correlation = merged.loc[:, ['Sex', 'Pclass', 'Embarked', 'Title', 'Family_size', 'Parch', 'SibSp', 'Cabin', 'Ticket']]
fig, axes = plt.subplots(nrows = 3, ncols = 3, figsize = (25,25))
for ax, column in zip(axes.flatten(), correlation.columns):
    sns.boxplot(x = correlation[column], y =  merged.Age, ax = ax)
    ax.set_title(column, fontsize = 23)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 20)
    ax.tick_params(axis = 'both', which = 'minor', labelsize = 20)
    ax.set_ylabel('Age', fontsize = 20)
    ax.set_xlabel('')
fig.suptitle('Variables Associated with Age', fontsize = 30)
fig.tight_layout(rect = [0, 0.03, 1, 0.95])
"""Let's plot correlation heatmap to see which variable is highly correlated with Age and if our boxplot interpretation holds true. We need to convert categorical variable into numerical to plot correlation heatmap. So convert categorical variables into numerical."""
from sklearn.preprocessing import LabelEncoder
correlation = correlation.agg(LabelEncoder().fit_transform)
correlation['Age'] = merged.Age # Inserting Age in variable correlation.
correlation = correlation.set_index('Age').reset_index() # Move Age at index 0.
# Now create the heatmap correlation
plt.figure(figsize=(15,6))
sns.heatmap(correlation.corr(), cmap ='BrBG',annot = True)
plt.title('Variables correlated with Age')
plt.show()
'''Impute Age with median of respective columns (i.e., Title and Pclass).'''
merged.Age = merged.groupby(['Title', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
'''So by now we should have variables with no missing values.'''
bold('**Missing Values after Imputation:**')
display(merged.isnull().sum())
"""Let's split the train and test data for bivariate analysis since test data has no Survived values. We need our target variable without missing values to conduct the association test with predictor variables."""
df_train = merged.iloc[:891, :]
df_test = merged.iloc[891:, :]
df_test = df_test.drop(columns = ['Survived'], axis = 1)

'''#1.Create a function that creates boxplot between categorical and numerical variables and calculates biserial correlation.'''
def boxplot_and_correlation(cat,num):
    '''cat = categorical variable, and num = numerical variable.'''
    plt.figure(figsize = (18,7))
    title_size = 18
    font_size = 15
    ax = sns.boxplot(x = cat, y = num)
    
    # Select boxes to change the color
    box = ax.artists[0]
    box1 = ax.artists[1]
    
    # Change the appearance of that box
    box.set_facecolor('red')
    box1.set_facecolor('green')
    plt.title('Association between Survived & %s' %num.name, fontsize = title_size)
    plt.xlabel('%s' %cat.name, fontsize = font_size)
    plt.ylabel('%s' %num.name, fontsize = font_size)
    plt.xticks(fontsize = font_size)
    plt.yticks(fontsize = font_size)
    plt.show()
    print('Correlation between', num.name, 'and', cat.name,':', stats.pointbiserialr(num, cat))

'''#2.Create another function to calculate mean when grouped by categorical variable. And also plot the grouped mean.'''
def nume_grouped_by_cat(num, cat):
    global ax
    font_size = 15
    title_size = 18
    grouped_by_cat = num.groupby(cat).mean().sort_values( ascending = False)
    grouped_by_cat.rename ({1:'survived', 0:'died'}, axis = 'rows', inplace = True) # Renaming index
    grouped_by_cat = round(grouped_by_cat, 2)
    ax = grouped_by_cat.plot.bar(figsize = (18,5)) 
    abs_bar_labels()
    plt.title('Mean %s ' %num.name + ' of Survivors vs Victims', fontsize = title_size)
    plt.ylabel('Mean ' + '%s' %num.name, fontsize = font_size)
    plt.xlabel('%s' %cat.name, fontsize = font_size)
    plt.xticks(fontsize = font_size)
    plt.yticks(fontsize = font_size)
    plt.show()

'''#3.This function plots histogram of numerical variable for every class of categorical variable.'''
def num_hist_by_cat(num,cat):
    font_size = 15
    title_size = 18
    plt.figure(figsize = (18,7))
    num[cat == 1].hist(color = ['g'], label = 'Survived', grid = False)
    num[cat == 0].hist(color = ['r'], label = 'Died', grid = False)
    plt.yticks([])
    plt.xticks(fontsize = font_size)
    plt.xlabel('%s' %num.name, fontsize = font_size)
    plt.title('%s ' %num.name + ' Distribution of Survivors vs Victims', fontsize = title_size)
    plt.legend()
    plt.show()
   
'''#4.Create a function to calculate anova between numerical and categorical variable.'''
def anova(num, cat):
    from scipy import stats
    grp_num_by_cat_1 = num[cat == 1] # Group our numerical variable by categorical variable(1). Group Fair by survivors
    grp_num_by_cat_0 = num[cat == 0] # Group our numerical variable by categorical variable(0). Group Fare by victims
    f_val, p_val = stats.f_oneway(grp_num_by_cat_1, grp_num_by_cat_0) # Calculate f statistics and p value
    print('Anova Result between ' + num.name, ' & '+ cat.name, ':' , f_val, p_val)  
'''Create a boxplot to visualize the strength of association of Survived with Fare. Also calculate biserial correlation.'''
boxplot_and_correlation(df_train.Survived, df_train.Fare)
'''So the mean fare of survivors should be much more (positive correlation or boxplot interpretation) than those who died. Calculate mean fare paid by the survivors as well as by the victims.'''
nume_grouped_by_cat(df_train.Fare, df_train.Survived)
"""Plot histogram of survivor's vs victims fare."""
num_hist_by_cat(df_train.Fare, df_train.Survived)
"""Let's perform ANOVA between Fare and Survived. One can omit this step. I perform just to show how anova is performed if there were more than two groups in our categorical variable."""
anova(df_train.Fare, df_train.Survived)
"""Let's create a box plot between Age and Survived to have an idea by how much Age is associated with Survived. Also find point biserial correlation between them."""
boxplot_and_correlation(df_train.Survived, df_train.Age)
'''So the mean age of survivors should be just less than those who died (small negative correlation and reading boxplot). Calculate the mean age of survivors and victims.'''
nume_grouped_by_cat(df_train.Age, df_train.Survived)
'''Histogram of survivors vs victims age.'''
num_hist_by_cat(df_train.Age, df_train.Survived)
'''Perform ANOVA between all the levels of Survived (i.e.., 0 and 1) and Age.'''
anova(df_train.Age, df_train.Survived)
'''#1.Create a function that calculates absolute and relative frequency of Survived variable by a categorical variable. And then plots the absolute and relative frequency of Survived by a categorical variable.'''
def crosstab(cat, cat_target):
    '''cat = categorical variable, cat_target = our target categorical variable.'''
    global ax, ax1
    fig_size = (18, 5)
    title_size = 18
    font_size = 15
    cat_grouped_by_cat_target = pd.crosstab(index = cat, columns = cat_target)
    cat_grouped_by_cat_target.rename({0:'Victims', 1:'Survivors'}, axis = 'columns', inplace = True)  # Renaming the columns
    pct_cat_grouped_by_cat_target = round(pd.crosstab(index = cat, columns = cat_target, normalize = 'index')*100, 2)
    pct_cat_grouped_by_cat_target.rename({0:'Victims(%)', 1:'Survivors(%)'}, axis = 'columns', inplace = True)
    
    # Plot absolute frequency of Survived by a categorical variable
    ax =  cat_grouped_by_cat_target.plot.bar(color = ['r', 'g'], title = 'Absolute Count of Survival and Death by %s' %cat.name, figsize = fig_size)
    ax.title.set_size(fontsize = title_size)
    abs_bar_labels()
    plt.xlabel(cat.name, fontsize = font_size)
    plt.show()
    
    # Plot relative frequrncy of Survived by a categorical variable
    ax1 = pct_cat_grouped_by_cat_target.plot.bar(color = ['r', 'g'], title = 'Percentage Count of Survival and Death by %s' %cat.name, figsize = fig_size)
    ax1.title.set_size(fontsize = title_size)
    pct_bar_labels()
    plt.xlabel(cat.name, fontsize = font_size)
    plt.show()
    
'''#2.Create a function to calculate chi_square test between a categorical and target categorical variable.'''
def chi_square(cat, cat_target):
    cat_grouped_by_cat_target = pd.crosstab(index = cat, columns = cat_target)
    test_result = stats.chi2_contingency (cat_grouped_by_cat_target)
    print('Chi Square Test Result between Survived & %s' %cat.name + ':')
    display(test_result)

'''#3.Finally create another function to calculate Bonferroni-adjusted pvalue for a categorical and target categorical variable.'''
def bonferroni_adjusted(cat, cat_target):
    dummies = pd.get_dummies(cat)
    for columns in dummies:
        crosstab = pd.crosstab(dummies[columns], cat_target)
        print(stats.chi2_contingency(crosstab))
    print('\nColumns:', dummies.columns)
'''Plot the no of passergers who survived and died due to their sex in absolute and relative scale.'''
crosstab(df_train.Sex, df_train.Survived)
'''Perform chi-square test of independence between Survived and Sex.'''
chi_square(df_train.Sex, df_train.Survived)
'''Plot the number of passengers who survived and died due to their pclass in absolute and relative scale.'''
crosstab(df_train.Pclass, df_train.Survived)
'''Perform chi-square test of independence between Survived and Pclass.'''
chi_square(df_train.Pclass, df_train.Survived)
'''Calculate Bonferroni-adjusted pvalue for Pclass (1,2,3) and Survived.'''
bonferroni_adjusted(df_train.Pclass, df_train.Survived)
'''Count and plot the survivors and victims by place of embarkation in absolute and relative scale.'''
crosstab(df_train.Embarked, df_train.Survived)
'''Now perform chi-square test to find the association between Embarked and Survived.'''
chi_square(df_train.Embarked, df_train.Survived)
'''Calculate Bonferroni-adjusted pvalue  between Embarked (C,Q,S one by one) and Survived.'''
bonferroni_adjusted(df_train.Embarked, df_train.Survived)
'''Count and plot absolute and relative number of survivors and victims due to SibSp.'''
crosstab(df_train.SibSp, df_train.Survived)
'''Chi-square test between SibSp and Survived.'''
chi_square(df_train.SibSp, df_train.Survived)
'''Count and visualize absolute and relative number of survivors and victims by Parch.'''
crosstab(df_train.Parch, df_train.Survived)
'''Perform Chi-square test of independence between Parch and Survived.'''
chi_square(df_train.Parch, df_train.Survived)
'''Count and visualize absolute and relative number of survivors and victims by Title.'''
crosstab(df_train.Title, df_train.Survived)
'''Perform Chi-square test of independence between Title and Survived.'''
chi_square(df_train.Title, df_train.Survived)
'''Survivors and victims count and percentage count by Family_size. Also plot the absolute and percentage count.'''
crosstab(df_train.Family_size, df_train.Survived)
'''Perform Chi-square test of independence between Family_size and Survived.'''
chi_square(df_train.Family_size, df_train.Survived)
'''Calculate Bonferroni-adjusted pvalue  between Family_size and Survived.'''
bonferroni_adjusted(df_train.Family_size, df_train.Survived)
'''Count and plot absolute and relative number of survivors and victims due to Cabin possession.'''
crosstab(df_train.Cabin, df_train.Survived)
"""Perform Chi-square test of independence between Cabin and Survived."""
chi_square(df_train.Cabin, df_train.Survived)
'''Count and plot absolute and relative number of survivors and victims due to Ticket category.'''
crosstab(df_train.Ticket, df_train.Survived)
'''Perform Chi-square test of independence between Ticket and Survived.'''
chi_square(df_train.Ticket, df_train.Survived)
'''Create a function that plots the impact of 3 predictor variables at a time on a target variable.'''
def multivariate_analysis(cat1, cat2, cat3, cat_target):
    font_size = 15
    grouped = round(pd.crosstab(index = [cat1, cat2, cat3], columns = cat_target, normalize = 'index')*100, 2)
    grouped.rename({0:'Died%', 1:'Survived%'}, axis = 1, inplace = True)
    grouped.plot.bar(color = ['r', 'g'], figsize = (18,5))
    plt.xlabel(cat1.name + ',' + cat2.name + ',' + cat3.name, fontsize = font_size)
    plt.ylabel('Relative Frequency (%)', fontsize = font_size)
    plt.xticks(fontsize = font_size)
    plt.yticks(fontsize = font_size)
    plt.legend(loc = 'best')
    plt.show()
'''Proportion of survivors and victims due to pclass, sex, and cabin.'''
multivariate_analysis(df_train.Pclass, df_train.Sex, df_train.Cabin, df_train.Survived)
bold('**Findings: Sex male seems to be deciding factor for death.**')
'''Proportion of survivors and victims due to pclass, sex, and embarked.'''
multivariate_analysis(df_train.Pclass, df_train.Sex, df_train.Embarked, df_train.Survived)
bold('**Findings: Again Sex male seems to be deciding factor for death and female for survival.**')
'''Proportion of survivors and victims due to pclass, sex, and SibSp.'''
multivariate_analysis(df_train.Pclass, df_train.Sex, df_train.SibSp, df_train.Survived)
bold('**Findings: Bigger SibSp and male is responsible more for death.**')
'''Proportion of survivors and victims due to pclass, sex, and Parch.'''
multivariate_analysis(df_train.Pclass, df_train.Sex, df_train.Parch, df_train.Survived)
bold('**Findings: Bigger Parch and Sex male is responsible more for death.**')
'''Proportion of survivors and victims due to pclass, sex, and title.'''
multivariate_analysis(df_train.Pclass, df_train.Sex, df_train.Title, df_train.Survived)
bold('**Findings: Findings: Passengers with sex male and title mr mostly died.**')
'''Proportion of survivors and victims due to pclass, sex, and family_size.'''
multivariate_analysis(df_train.Pclass, df_train.Sex, df_train.Family_size, df_train.Survived)
bold('**Findings: Sex male, family_size single and large greatly influence the death ratio.**')
'''Proportion of survivors and victims due to pclass, sex, and Ticket category.'''
multivariate_analysis(df_train.Pclass, df_train.Sex, df_train.Ticket, df_train.Survived)
bold('**Findings: Sex female, ticket p and w mostly survived.**')
'''Proportion of survivors and victims due to pclass, title, and cabin.'''
multivariate_analysis(df_train.Pclass, df_train.Title, df_train.Cabin, df_train.Survived)
bold('**Findings: Title mrs, master and cabin x had best survival ratio.**')
'''Proportion of survivors and victims due to family_size, sex, and cabin.'''
multivariate_analysis(df_train.Family_size, df_train.Sex, df_train.Cabin, df_train.Survived)
bold('**Findings: Family_size small, medium and sex female had best survival chance.**')
'''Proportion of survivors and victims due to sex, title, and family_size.'''
multivariate_analysis(df_train.Sex, df_train.Title, df_train.Family_size, df_train.Survived)
bold('**Findings: Title aristocrat, sex female and family_size small mostly survived.**')
'''Proportion of survivors and victims due to sex, title, and cabin.'''
multivariate_analysis(df_train.Sex, df_train.Title, df_train.Cabin, df_train.Survived)
bold('**Findings: findings: Title aristocrat, miss, mrs and sex female mostly survived.**')
'''Proportion of survivors and victims due to sex, title, and embarked.'''
multivariate_analysis(df_train.Sex, df_train.Title, df_train.Embarked, df_train.Survived)
bold('**Findings: Embarked c, sex female and title master and aristocrat had best survival rate.**')
"""Proportion of survivors and victims due to sex, title, and Ticket."""
multivariate_analysis(df_train.Sex, df_train.Title, df_train.Ticket, df_train.Survived)
bold('**Findings: Ticker n, w and sex male and title mr mostly died.**')
'''Create bin categories for Age.'''
label_names = ['infant','child','teenager','young_adult','adult','aged']

'''Create range for each bin categories of Age.'''
cut_points = [0,5,12,18,35,60,81]

'''Create and view categorized Age with original Age.'''
merged['Age_binned'] = pd.cut(merged.Age, cut_points, labels = label_names)
bold('**Age with Categorized Age:**')
display(merged[['Age', 'Age_binned']].head(2))
'''Create bin categories for Fare.'''
groups = ['low','medium','high','very_high']

'''Create range for each bin categories of Fare.'''
cut_points = [-1, 130, 260, 390, 520]

'''Create and view categorized Fare with original Fare.'''
merged['Fare_binned'] = pd.cut(merged.Fare, cut_points, labels = groups)
bold('**Fare with Categorized Fare:**')
display(merged[['Fare', 'Fare_binned']].head(2))
"""Let's see all the variables we currently have with their category."""
display(merged.head(2))

'''Drop the features that would not be useful anymore.'''
merged.drop(columns = ['Name', 'Age', 'Fare'], inplace = True, axis = 1)

'''Features after dropping.'''
bold('**Features Remaining after Dropping:**')
display(merged.columns)
'''Checking current data types.'''
bold('**Current Variable Data Types:**')
display(merged.dtypes)
'''Correcting data types, converting into categorical variables.'''
merged.loc[:, ['Pclass', 'Sex', 'Embarked', 'Cabin', 'Title', 'Family_size', 'Ticket']] = merged.loc[:, ['Pclass', 'Sex', 'Embarked', 'Cabin', 'Title', 'Family_size', 'Ticket']].astype('category')

'''Due to merging there are NaN values in Survived for test set observations.'''
merged.Survived = merged.Survived.dropna().astype('int')#Converting without dropping NaN throws an error.
'''Check if data types have been corrected.'''
bold('**Data Types after Correction:**')
display(merged.dtypes)
'''Convert categorical data into numeric to feed our machine learning model.'''
merged = pd.get_dummies(merged)

"""Let's visualize the updated dataset that would be fed to our machine learning algorithms."""
bold('**Preview of Processed Data:**')
display(merged.head(2))
'''Set a seed for reproducibility'''
seed = 43

"""Let's split the train and test set to feed machine learning algorithm."""
df_train = merged.iloc[:891, :]
df_test  = merged.iloc[891:, :]
'''Drop passengerid from train set and Survived from test set.'''
df_train = df_train.drop(columns = ['PassengerId'], axis = 1)
df_test = df_test.drop(columns = ['Survived'], axis = 1)
'''Extract data sets as input and output for machine learning models.'''
X_train = df_train.drop(columns = ['Survived'], axis = 1) # Input matrix as pandas dataframe (dim:891*47).
y_train = df_train['Survived'] # Output vector as pandas series (dim:891*1)

"""Extract test set"""
X_test  = df_test.drop("PassengerId", axis = 1).copy()
'''See the dimensions of input and output data set.'''
print('Input Matrix Dimension:  ', X_train.shape)
print('Output Vector Dimension: ', y_train.shape)
print('Test Data Dimension:     ', X_test.shape)
"""Building machine learning models: 
We will try 10 different classifiers to find the best classifier after tunning model's hyperparameters that will best generalize the unseen(test) data."""

'''Now initialize all the classifiers object.'''
'''#1.Logistic Regression'''
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

'''#2.Support Vector Machines'''
from sklearn.svm import SVC
svc = SVC(gamma = 'auto')

'''#3.Random Forest Classifier'''
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state = seed, n_estimators = 100)

'''#4.KNN'''
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

'''#5.Gaussian Naive Bayes'''
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

'''#6.Decision Tree Classifier'''
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state = seed)

'''#7.Gradient Boosting Classifier'''
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(random_state = seed)

'''#8.Adaboost Classifier'''
from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier(random_state = seed)

'''#9.ExtraTrees Classifier'''
from sklearn.ensemble import ExtraTreesClassifier
etc = ExtraTreesClassifier(random_state = seed)

'''#10.Extreme Gradient Boosting'''
from xgboost import XGBClassifier
xgbc = XGBClassifier(random_state = seed)
'''Create a function that returns train accuracy of different models.'''
def train_accuracy(model):
    model.fit(X_train, y_train)
    train_accuracy = model.score(X_train, y_train)
    train_accuracy = np.round(train_accuracy*100, 2)
    return train_accuracy


'''Models with best training accuracy:'''
train_accuracy = pd.DataFrame({'Train_accuracy(%)':[train_accuracy(lr), train_accuracy(svc), train_accuracy(rf), train_accuracy(knn), train_accuracy(gnb), train_accuracy(dt), train_accuracy(gbc), train_accuracy(abc), train_accuracy(etc), train_accuracy(xgbc)]})
train_accuracy.index = ['LR', 'SVC', 'RF', 'KNN', 'GNB', 'DT', 'GBC', 'ABC', 'ETC', 'XGBC']
sorted_train_accuracy = train_accuracy.sort_values(by = 'Train_accuracy(%)', ascending = False)
bold('**Training Accuracy of the Classifiers:**')
display(sorted_train_accuracy)
suppress_warnings()
'''Create a function that returns mean cross validation score for different models.'''
def x_val_score(model):
    from sklearn.model_selection import cross_val_score
    x_val_score = cross_val_score(model, X_train, y_train, cv = 10, scoring = 'accuracy').mean()
    x_val_score = np.round(x_val_score*100, 2)
    return x_val_score

"""Let's perform k-fold (k=10) cross validation to find the classifier with the best cross validation accuracy."""
x_val_score = pd.DataFrame({'X_val_score(%)':[x_val_score(lr), x_val_score(svc), x_val_score(rf), x_val_score(knn), x_val_score(gnb), x_val_score(dt), x_val_score(gbc), x_val_score(abc), x_val_score(etc), x_val_score(xgbc)]})
x_val_score.index = ['LR', 'SVC', 'RF', 'KNN', 'GNB', 'DT', 'GBC', 'ABC', 'ETC', 'XGBC']
sorted_x_val_score = x_val_score.sort_values(by = 'X_val_score(%)', ascending = False) 
bold('**Models 10-fold Cross Validation Score:**')
display(sorted_x_val_score)
suppress_warnings()
"""Define all the models' hyperparameters one by one first::"""

'''Define hyperparameters the logistic regression will be tuned with. For LR, the following hyperparameters are usually tunned.'''
lr_params = {'penalty':['l1', 'l2'],
             'C': np.logspace(0, 4, 10)}

'''For GBC, the following hyperparameters are usually tunned.'''
gbc_params = {'learning_rate': [0.01, 0.02, 0.05, 0.01],
              'max_depth': [4, 6, 8],
              'max_features': [1.0, 0.3, 0.1], 
              'min_samples_split': [ 2, 3, 4],
              'random_state':[seed]}

'''For SVC, the following hyperparameters are usually tunned.'''
svc_params = {'C': [6, 7, 8, 9, 10, 11, 12], 
              'kernel': ['linear','rbf'],
              'gamma': [0.5, 0.2, 0.1, 0.001, 0.0001]}

'''For DT, the following hyperparameters are usually tunned.'''
dt_params = {'max_features': ['auto', 'sqrt', 'log2'],
             'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 
             'min_samples_leaf':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
             'random_state':[seed]}

'''For RF, the following hyperparameters are usually tunned.'''
rf_params = {'criterion':['gini','entropy'],
             'n_estimators':[10, 15, 20, 25, 30],
             'min_samples_leaf':[1, 2, 3],
             'min_samples_split':[3, 4, 5, 6, 7], 
             'max_features':['sqrt', 'auto', 'log2'],
             'random_state':[44]}

'''For KNN, the following hyperparameters are usually tunned.'''
knn_params = {'n_neighbors':[3, 4, 5, 6, 7, 8],
              'leaf_size':[1, 2, 3, 5],
              'weights':['uniform', 'distance'],
              'algorithm':['auto', 'ball_tree','kd_tree','brute']}

'''For ABC, the following hyperparameters are usually tunned.'''
abc_params = {'n_estimators':[1, 5, 10, 15, 20, 25, 40, 50, 60, 80, 100, 130, 160, 200, 250, 300],
              'learning_rate':[0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5],
              'random_state':[seed]}

'''For ETC, the following hyperparameters are usually tunned.'''
etc_params = {'max_depth':[None],
              'max_features':[1, 3, 10],
              'min_samples_split':[2, 3, 10],
              'min_samples_leaf':[1, 3, 10],
              'bootstrap':[False],
              'n_estimators':[100, 300],
              'criterion':["gini"], 
              'random_state':[seed]}

'''For XGBC, the following hyperparameters are usually tunned.'''
xgbc_params = {'n_estimators': (150, 250, 350,450,550,650, 700, 800, 850, 1000),
              'learning_rate': (0.01, 0.6),
              'subsample': (0.3, 0.9),
              'max_depth': [3, 4, 5, 6, 7, 8, 9],
              'colsample_bytree': (0.5, 0.9),
              'min_child_weight': [1, 2, 3, 4],
              'random_state':[seed]}
'''Create a function to tune hyperparameters of the selected models.'''
def tune_hyperparameters(model, params):
    from sklearn.model_selection import GridSearchCV
    global best_params, best_score
    # Construct grid search object with 10 fold cross validation.
    grid = GridSearchCV(model, params, verbose = 0, cv = 10, scoring = 'accuracy', n_jobs = -1)
    # Fit using grid search.
    grid.fit(X_train, y_train)
    best_params, best_score = grid.best_params_, np.round(grid.best_score_*100, 2)
    return best_params, best_score
'''Tune LR hyperparameters.'''
tune_hyperparameters(lr, params = lr_params)
lr_best_params, lr_best_score = best_params, best_score
print('LR Best Score:', lr_best_score)
print('And Best Parameters:', lr_best_params)
"""Tune GBC's hyperparameters."""
tune_hyperparameters(gbc, params = gbc_params)
gbc_best_score, gbc_best_params = best_score, best_params
"""Tune SVC's hyperparameters."""
tune_hyperparameters(svc, params = svc_params)
svc_best_score, svc_best_params = best_score, best_params
"""Tune DT's hyperparameters."""
tune_hyperparameters(dt, params = dt_params)
dt_best_score, dt_best_params = best_score, best_params
"""Tune RF's hyperparameters."""
tune_hyperparameters(rf, params = rf_params)
rf_best_score, rf_best_params = best_score, best_params
"""Tune KNN's hyperparameters."""
tune_hyperparameters(knn, params = knn_params)
knn_best_score, knn_best_params = best_score, best_params
"""Tune ABC's hyperparameters."""
tune_hyperparameters(abc, params = abc_params)
abc_best_score, abc_best_params = best_score, best_params
"""Tune ETC's hyperparameters."""
tune_hyperparameters(etc, params = etc_params)
etc_best_score, etc_best_params = best_score, best_params
'''Create a dataframe of tunned scores and sort them in descending order.'''
tunned_scores = pd.DataFrame({'Tunned_accuracy(%)': [lr_best_score, gbc_best_score, svc_best_score, dt_best_score, rf_best_score, knn_best_score, abc_best_score, etc_best_score]})
tunned_scores.index = ['LR', 'GBC', 'SVC', 'DT', 'RF', 'KNN', 'ABC', 'ETC']
sorted_tunned_scores = tunned_scores.sort_values(by = 'Tunned_accuracy(%)', ascending = False)
bold('**Models Accuracy after Optimization:**')
display(sorted_tunned_scores)
'''#4.Create a function that compares cross validation scores with tunned scores for different models by plotting them.'''
def compare_scores(accuracy):
    global ax1   
    font_size = 15
    title_size = 18
    ax1 = accuracy.plot.bar(legend = False,  title = 'Models %s' % ''.join(list(accuracy.columns)), figsize = (18, 5), color = 'sandybrown')
    ax1.title.set_size(fontsize = title_size)
    # Removes square brackets and quotes from column name after to converting list.
    pct_bar_labels()
    plt.ylabel('% Accuracy', fontsize = font_size)
    plt.show()

'''Compare cross validation scores with tunned scores to find the best model.'''
bold('**Comparing Cross Validation Scores with Optimized Scores:**')
compare_scores(sorted_x_val_score)
compare_scores(sorted_tunned_scores)
'''Instantiate the models with optimized hyperparameters.'''
rf  = RandomForestClassifier(**rf_best_params)
gbc = GradientBoostingClassifier(**gbc_best_params)
svc = SVC(**svc_best_params)
knn = KNeighborsClassifier(**knn_best_params)
etc = ExtraTreesClassifier(**etc_best_params)
lr  = LogisticRegression(**lr_best_params)
dt  = DecisionTreeClassifier(**dt_best_params)
abc = AdaBoostClassifier(**abc_best_params)

'''Train all the models with optimised hyperparameters.'''
models = {'RF':rf, 'GBC':gbc, 'SVC':svc, 'KNN':knn, 'ETC':etc, 'LR':lr, 'DT':dt, 'ABC':abc}
bold('**10-fold Cross Validation after Optimization:**')
score = []
for x, (keys, items) in enumerate(models.items()):
    # Train the models with optimized parameters using cross validation.
    # No need to fit the data. cross_val_score does that for us.
    # But we need to fit train data for prediction in the follow session.
    from sklearn.model_selection import cross_val_score
    items.fit(X_train, y_train)
    scores = cross_val_score(items, X_train, y_train, cv = 10, scoring = 'accuracy')*100
    score.append(scores.mean())
    print('Mean Accuracy: %0.4f (+/- %0.4f) [%s]'  % (scores.mean(), scores.std(), keys))
'''Make prediction using all the trained models.'''
model_prediction = pd.DataFrame({'RF':rf.predict(X_test), 'GBC':gbc.predict(X_test), 'ABC':abc.predict(X_test),
                                 'ETC':etc.predict(X_test), 'DT':dt.predict(X_test), 'SVC':svc.predict(X_test), 
                                 'KNN':knn.predict(X_test), 'LR':lr.predict(X_test)})

"""Let's see how each model classifies a prticular class."""
bold('**All the Models Prediction:**')
display(model_prediction.head())
'''Create a function that plot feature importance by the selected tree based models.'''
def feature_importance(model):
    importance = pd.DataFrame({'Feature': X_train.columns,
                              'Importance': np.round(model.feature_importances_,3)})
    importance = importance.sort_values(by = 'Importance', ascending = False).set_index('Feature')
    return importance

'''Create subplots of feature impotance of rf, gbc, dt, etc, and abc.'''
fig, axes = plt.subplots(3,2, figsize = (20,40))
fig.suptitle('Tree Based Models Feature Importance', fontsize = 28)
tree_models = [rf, gbc, dt, etc, abc]
tree_names = ['RF', 'GBC', 'DT', 'ETC', 'ABC']

for ax, model, name in zip(axes.flatten(), tree_models, tree_names):
    feature_importance(model).plot.barh(ax = ax, title = name, fontsize = 16, color = 'green')
fig.delaxes(ax = axes[2,1]) # We don't need the last subplot.
fig.tight_layout(rect = [0, 0.03, 1, 0.97])
"""Let's plot feature importance of LR."""
coeff = pd.DataFrame({'Feature':X_train.columns,'Importance':np.transpose(lr.coef_[0])})
coeff.sort_values(by = 'Importance').set_index('Feature').plot.bar(title = 'Feature Importance of Linear Model (LR)', color = 'chocolate', figsize = (18,2.5))
plt.show()
'''Create a function that returns learning curves for different classifiers.'''
def plot_learning_curve(model):
    from sklearn.model_selection import learning_curve
    # Create feature matrix and target vector
    X, y = X_train, y_train
    # Create CV training and test scores for various training set sizes
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv = 10,
                                                    scoring='accuracy', n_jobs = -1, 
                                                    train_sizes = np.linspace(0.01, 1.0, 17), random_state = seed)
                                                    # 17 different sizes of the training set

    # Create means and standard deviations of training set scores
    train_mean = np.mean(train_scores, axis = 1)
    train_std = np.std(train_scores, axis = 1)

    # Create means and standard deviations of test set scores
    test_mean = np.mean(test_scores, axis = 1)
    test_std = np.std(test_scores, axis = 1)

    # Draw lines
    plt.plot(train_sizes, train_mean, 'o-', color = 'red',  label = 'Training score')
    plt.plot(train_sizes, test_mean, 'o-', color = 'green', label = 'Cross-validation score')
    
    # Draw bands
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha = 0.1, color = 'r') # Alpha controls band transparency.
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha = 0.1, color = 'g')

    # Create plot
    font_size = 15
    plt.xlabel('Training Set Size', fontsize = font_size)
    plt.ylabel('Accuracy Score', fontsize = font_size)
    plt.xticks(fontsize = font_size)
    plt.yticks(fontsize = font_size)
    plt.legend(loc = 'best')
    plt.grid()
'''Now plot learning curves of the optimized models in subplots.'''
plt.figure(figsize = (25,25))
lc_models = [rf, gbc, dt, etc, abc, knn, svc, lr]
lc_labels = ['RF', 'GBC', 'DT', 'ETC', 'ABC', 'KNN', 'SVC', 'LR']

for ax, models, labels in zip (range(1,9), lc_models, lc_labels):
    plt.subplot(4,2,ax)
    plot_learning_curve(models)
    plt.title(labels, fontsize = 18)
plt.suptitle('Learning Curves of Optimized Models', fontsize = 28)
plt.tight_layout(rect = [0, 0.03, 1, 0.97])
'''Return prediction to use it in another function.'''
def x_val_predict(model):
    from sklearn.model_selection import cross_val_predict
    predicted = cross_val_predict(model, X_train, y_train, cv = 10)
    return predicted # Now we can use it in another function by assigning the function to its return value.

'''Function to return confusion matrix.'''
def confusion_matrix(model):
    predicted = x_val_predict(model)
    confusion_matrix = pd.crosstab(y_train, predicted, rownames = ['Actual'], colnames = ['Predicted/Classified'], margins = True) # We use pandas crosstab
    return display(confusion_matrix)

'''Now calculate confusion matrix of rf and gbc.'''
bold('**RF Confusion Matrix:**')
confusion_matrix(rf)
bold('**GBC Confusion Matrix:**')
confusion_matrix(gbc)
'''Function to calculate precision score.'''
def precision_score(model):
    from sklearn.metrics import precision_score
    predicted = x_val_predict(model)
    precision_score = precision_score(y_train, predicted)
    return np.round(precision_score*100, 2)

'''Compute precision score for rf and gbc.'''
print('RF  Precision Score:', precision_score(rf))
print('GBC Precision Score:', precision_score(gbc))
'''Function to calculate recall score.'''
def recall_score(model):
    from sklearn.metrics import recall_score
    predicted = x_val_predict(model)
    recall_score = recall_score(y_train, predicted)
    return np.round(recall_score*100, 2)

'''Compute recall score for rf and gbc.'''
print('RF  Recall Score:', recall_score(rf))
print('GBC Recall Score:', recall_score(gbc))
'''Function for specificity score.'''
def specificity_score(model):
    from sklearn.metrics import confusion_matrix
    predicted = x_val_predict(model)
    tn, fp, fn, tp = confusion_matrix(y_train, predicted).ravel()
    specificity_score = tn / (tn + fp)
    return np.round(specificity_score*100, 2)

'''Calculate specificity score for rf and gbc.'''
print('RF  Specificity Score:', specificity_score(rf))
print('GBC Specificity Score:', specificity_score(gbc))
'''Function for F1 score.'''
def f1_score(model):
    from sklearn.metrics import f1_score
    predicted = x_val_predict(model)
    f1_score = f1_score(y_train, predicted)
    return np.round(f1_score*100, 2)

'''Calculate f1 score for rf and gbc.'''
print('RF  F1 Score:', f1_score(rf))
print('GBC F1 Score:', f1_score(gbc))
'''Function to compute classification report.'''
def classification_report(model):
    from sklearn.metrics import classification_report
    predicted = x_val_predict(model)
    classification_report = classification_report(y_train, predicted)
    return print(classification_report)

'''Now calculate classification report for rf and gbc.'''
bold('**RF Classification Report:**')
classification_report(rf)
bold('**GBC Classification Report:**')
classification_report(gbc)
'''#7Function for plotting precision-recall vs threshold curve.'''
def precision_recall_vs_threshold(model, title):
    from sklearn.metrics import precision_recall_curve
    probablity = model.predict_proba(X_train)[:, 1]
    plt.figure(figsize = (18, 5))
    precision, recall, threshold = precision_recall_curve(y_train, probablity)
    plt.plot(threshold, precision[:-1], 'b-', label = 'precision', lw = 3.7)
    plt.plot(threshold, recall[:-1], 'g', label = 'recall', lw = 3.7)
    plt.xlabel('Threshold')
    plt.legend(loc = 'best')
    plt.ylim([0, 1])
    plt.title(title)
    plt.show()

'''Now plot precision-recall vs threshold curve for rf and gbc.'''
precision_recall_vs_threshold(rf, title = 'RF Precision-Recall vs Threshold Curve' )
precision_recall_vs_threshold(gbc, title = 'GBC Precision-Recall vs Threshold Curve')
'''Function to plot recall vs precision curve.'''
def plot_precision_vs_recall(model, title):
    from sklearn.metrics import precision_recall_curve
    probablity = model.predict_proba(X_train)[:, 1]
    plt.figure(figsize = (18, 5))
    precision, recall, threshold = precision_recall_curve(y_train, probablity)
    plt.plot(recall, precision, 'r-', lw = 3.7)
    plt.ylabel('Recall')
    plt.xlabel('Precision')
    plt.axis([0, 1.5, 0, 1.5])
    plt.title(title)
    plt.show()

'''Now plot recall vs precision curve of rf and gbc.'''
plot_precision_vs_recall(rf, title = 'RF Precision-Recall Curve')
plot_precision_vs_recall(gbc, title = 'GBC Precision-Recall Curve')
'''Function to plot ROC curve with AUC score.'''
def plot_roc_and_auc_score(model, title):
    from sklearn.metrics import roc_curve, roc_auc_score
    probablity = model.predict_proba(X_train)[:, 1]
    plt.figure(figsize = (18, 5))
    false_positive_rate, true_positive_rate, threshold = roc_curve(y_train, probablity)
    auc_score = roc_auc_score(y_train, probablity)
    plt.plot(false_positive_rate, true_positive_rate, label = "ROC CURVE, AREA = "+ str(auc_score))
    plt.plot([0, 1], [0, 1], 'red', lw = 3.7)
    plt.xlabel('False Positive Rate (1-Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.axis([0, 1, 0, 1])
    plt.legend(loc = 4)
    plt.title(title)
    plt.show()

'''Plot roc curve and auc score for rf and gbc.'''
plot_roc_and_auc_score(rf, title = 'RF ROC Curve with AUC Score')
plot_roc_and_auc_score(gbc, title = 'GBC ROC Curve with AUC Score')
'''Submission with the most accurate random forest classifier.'''
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": rf.predict(X_test)})
submission.to_csv('submission_rf.csv', index = False)


'''Submission with the most accurate gradient boosting classifier.'''
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": gbc.predict(X_test)})
submission.to_csv('submission_gbc.csv', index = False)
bold('**How hard voting works:**')
data = [[1, 1, 1, 0, 1], 
        [0, 0, 0, 1, 0]]
display(pd.DataFrame(data, columns= ['Class','RF', 'LR', 'KNN', 'Hard_voting']).set_index('Class'))
'''Create a data frame to store base models prediction.
First four in the dataframe are tree based models. Then two are kernel based. And the last is a linear model.'''
base_prediction = model_prediction # We've a df of all the models prediction.

"""Let's see how each model classifies a prticular class."""
bold('**All the Base Models Prediction:**')
display(base_prediction.head())

"""Let's visualize the correlations among the predictions of base models."""
plt.figure(figsize = (20, 6))
sns.heatmap(base_prediction.corr(), annot = True)
plt.title('Prediction Correlation among the Base Models', fontsize = 18)
plt.show()
'''We will use mlxtend library to train, predict and plot decision regions of hard voting ensemble classifier.'''
'''Define base models for hard voting ensemble.'''
base_models = [rf, gbc, dt, knn, lr]

'''Import ensemble classifier from mlxtend.'''
from mlxtend.classifier import EnsembleVoteClassifier

'''Initialize hard voting ensemble.'''
hard_vct = EnsembleVoteClassifier(clfs = base_models, voting = 'hard')
print('Training Hard Voting Ensemble Classifier...')
display(hard_vct.fit(X_train, y_train))
print('Done.')

'''Predict with hard voting ensemble.'''
y_pred_hard_vct = pd.DataFrame(hard_vct.predict(X_test), columns = ['Hard_vct'])

'''Hard voting cross validation score.'''
print('\nComputing Hard Voting Cross Val Score...')
hard_x_val_score = cross_val_score(hard_vct, X_train, y_train, cv = 10, scoring = 'accuracy')
hard_x_val_score = np.round(hard_x_val_score.mean()*100, 2)
print('Done.')

'''Compare hard voting score with best base models scores.'''
hard_vs_base_score = pd.DataFrame({'Hard_vs_base_score(%)': [hard_x_val_score, rf_best_score, gbc_best_score, dt_best_score, knn_best_score, lr_best_score]})
'''So basically we're comparing hard voting x_val_score with base models's tunned score.'''
hard_vs_base_score.index = ['Hard_vct', 'RF', 'GBC', 'DT', 'KNN', 'LR']
bold('**Hard Voting vs Base Models Scores:**')
display(hard_vs_base_score)
suppress_warnings()
'''Perform Standarization:
Variables have very different ranges (diffenence between max and  min). The purpose of standarization is to reduce the dispersion of these variables.'''
from sklearn.preprocessing import StandardScaler
'''Initialize standard scaler object.'''
std_scaler = StandardScaler()
'''Fit standard scaler object to train data.'''
std_scaler.fit(X_train)
'''Apply the standard scaler to training set.'''
X_train_scaled = std_scaler.transform(X_train)


'''Perform PCA:'''
from sklearn.decomposition import PCA
'''Initialize pca object with two components. i.e., converting into 2d from 47d.'''
pca = PCA(n_components = 2) # Projection to 2d from 47d
'''Fit pca to scaled data.'''
pca.fit(X_train_scaled)
'''Apply pca to scaled data.'''
pca_train = pca.transform(X_train_scaled)
'''Create a data frame consisting of two pca.'''
df_train_pca = pd.DataFrame(data = pca_train, columns = ['pca-1', 'pca-2'])
bold('**Projection to 2D from 47D:**')
display(df_train_pca.head())

"""let's merge our two pca components with our target feature."""
finaldf = pd.concat([df_train_pca, y_train], axis = 1)
bold('**Target with 2-PCA Components:**')
display(finaldf.head())
'''Now calculate how much variance we get off these two components.'''
bold('**Total Variance Explained by 2 PCA Components:**')
display(np.round((pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1])*100, 2))
'''Visualize our newly transformed samples with class labels.'''
'''Visualize our newly transformed samples with class labels.'''
plt.figure(figsize = (18,6))
plt.xlabel('PCA_1', fontsize = 15)
plt.ylabel('PCA_2', fontsize = 15)
plt.title('2-Component PCA (2D-Transformed Samples)', fontsize = 18)
targets = [1, 0]
colors = ['g', 'r']
for target, color in zip(targets,colors):
    indices = finaldf['Survived'] == target
    plt.scatter(finaldf.loc[indices, 'pca-1'],
               finaldf.loc[indices, 'pca-2'],
               c = color, s = 37)
plt.legend(targets)
plt.show()
'''We will use mlxtend for plotting decision regions of base and ensemble models. Initialize base models and hard voting ensemble.'''
rf_pca = RandomForestClassifier(random_state = seed)
gbc_pca = GradientBoostingClassifier(random_state = seed)
dt_pca = DecisionTreeClassifier(random_state = seed)
knn_pca = KNeighborsClassifier()
lr_pca = LogisticRegression(random_state = seed)
base_model_pca = [rf_pca, gbc_pca, dt_pca, knn_pca, lr_pca]
hard_vct_pca = EnsembleVoteClassifier(clfs = base_model_pca, voting = 'hard')

'''Function to plot decision region.'''
def plot_decision_region(model):
    from mlxtend.plotting import plot_decision_regions
    '''Train models with data pca returned. Get the train data.'''
    X = df_train_pca.values # Must be converted into numpy array.
    y = y_train.values
    model.fit(X, y) 
    decision_region = plot_decision_regions(X = X, y = y, clf = model)
    plt.xlabel('PCA-1', fontsize = 15)
    plt.ylabel('PCA_2', fontsize = 15)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    return decision_region

'''Now plot decison regions for hard voting ensemble vs base models in subplots.'''
plt.figure(figsize = (25,25))
en_models = [hard_vct_pca, rf_pca, gbc_pca, dt_pca, knn_pca, lr_pca]
en_labels = ['Hard_vct', 'RF', 'GBC', 'DT', 'KNN', 'LR']
for ax, models, labels in zip(range(1,7), en_models, en_labels):
    plt.subplot(3,2,ax)
    plot_decision_region(models)
    plt.title(labels, fontsize = 18)
plt.suptitle('Hard Voting vs Base Models Decision Regions', fontsize = 28)
plt.tight_layout(rect = [0, 0.03, 1, 0.97])
suppress_warnings()
'''Create a data frame consisting of base models and hard voting ensemble predictions. Revised base models are now rf, gbc, dt, knn, lr without svc and etc.'''
base_prediction = base_prediction.drop(columns = ['ABC', 'SVC', 'ETC'], axis = 1)

'''See base models prediction with hard voting prediction.'''
df_hard_base = pd.concat([base_prediction,y_pred_hard_vct], sort = False, axis = 1)
display(df_hard_base.head(7))
bold('**How soft voting works:**')
data = [[0.49, 0.99, 0.49, 0.66, 1], 
        [0.51, 0.01, 0.51, 0.34, 0]]
display(pd.DataFrame(data, columns= ['RF', 'LR', 'KNN', 'Average', 'Soft_voting']))
'''Base models for soft voting is the base models of hard voting.'''
'''Initialize soft voting ensemble.'''
base_models = [rf, gbc, dt, knn, lr]
soft_vct = EnsembleVoteClassifier(clfs = base_models, voting = 'soft')
print('Fitting Soft Voting Ensemble...')
display(soft_vct.fit(X_train, y_train))
print('Done.')

'''Predict with soft voting ensemble.'''
y_pred_soft_vct = pd.DataFrame(soft_vct.predict(X_test), columns = ['Soft_vct'])

'''Hard voting cross validation score.'''
print('\nComputing Soft Voting X Val Score...')
soft_x_val_score = cross_val_score(soft_vct, X_train, y_train, cv = 10, scoring = 'accuracy')
soft_x_val_score = np.round(soft_x_val_score.mean()*100, 2)
print('Done.')

'''Compare hard voting score with best base models scores.'''
soft_vs_base_score = pd.DataFrame({'Soft_vs_base_score(%)': [soft_x_val_score, rf_best_score, gbc_best_score, dt_best_score, knn_best_score, lr_best_score]})
'''So basically we're comparing soft voting x_val_score with base models's tunned score.'''
soft_vs_base_score.index = ['Soft_vct', 'RF', 'GBC', 'DT', 'KNN', 'LR']
bold('**Soft Voting vs Base Models Scores:**')
display(soft_vs_base_score)
suppress_warnings()
'''We would use the same data to plot decision region we got analysing PCA.'''
base_model_pca = [rf_pca, gbc_pca, dt_pca, knn_pca, lr_pca]
soft_vct_pca = EnsembleVoteClassifier(clfs = base_model_pca, voting = 'soft')

'''Plot decision regions for soft voting ensemble vs base models in subplots.'''
plt.figure(figsize = (25,25))
en_models = [soft_vct_pca, rf_pca, gbc_pca, dt_pca, knn_pca, lr_pca]
en_labels = ['Soft_vct', 'RF', 'GBC', 'DT', 'KNN', 'LR']
for ax, models, labels in zip(range(1,7), en_models, en_labels):
    plt.subplot(3,2,ax)
    plot_decision_region(models)
    plt.title(labels, fontsize = 18)
plt.suptitle('Soft Voting vs Base Models Decision Regions', fontsize = 28)
plt.tight_layout(rect = [0, 0.03, 1, 0.97])
suppress_warnings()
'''See base models prediction with soft voting prediction.'''
df_soft_base = pd.concat([base_prediction,y_pred_soft_vct], sort = False, axis = 1)
bold('**Soft voting should have labeled the class 1 instead of 0.**')
display(df_soft_base.iloc[192, :])
bold('**Soft voting should have labeled the class 0 instead of 1.**')
display(df_soft_base.iloc[335, :])
bold('**Soft voting should have labeled the class 1 instead of 0.**')
display(df_soft_base.iloc[344, :])
'''Initialize bagging classifier.'''
from sklearn.ensemble import BaggingClassifier
bagg = BaggingClassifier(base_estimator = rf, verbose = 0, n_jobs = -1, random_state = seed)
'''We use rf as the base estimator for bagging technique.'''
print('Fitting Bagging Ensemble...')
display(bagg.fit(X_train, y_train))
print('Done.')

'''Bagging cross validation score.'''
print('\nComputing Bagging X Val Score..')
bagg_x_val_score = cross_val_score(bagg, X_train, y_train, cv = 10, scoring = 'accuracy')
bagg_x_val_score = np.round(bagg_x_val_score.mean()*100, 2)
print('Done.')

'''Compare bagging ensemble score with best base models scores.'''
bagg_vs_base_score = pd.DataFrame({'Bagging_vs_base_score(%)': [bagg_x_val_score, rf_best_score, gbc_best_score, dt_best_score, knn_best_score, lr_best_score]})
'''So basically we're comparing bagging x_val_score with base models's tunned score.'''
bagg_vs_base_score.index = ['Bagg', 'RF', 'GBC', 'DT', 'KNN', 'LR']
bold('**Bagging vs Base Models Scores:**')
display(bagg_vs_base_score)
'''We will use adaptive boosting, gradient boosting and extreme gradient boosting classifiers for boosting ensemble method.'''
'''Initialize boosting classifier. Base models for boosting:'''
boost_models = [abc, gbc, xgbc] # Unoptimized xgbc
boost = EnsembleVoteClassifier(clfs = boost_models, voting='hard')

'''Fitting boosting.'''
print('Fitting Boosting Ensemble...')
display(boost.fit(X_train, y_train))
print('Done.')

'''Boosting cross validation score.'''
print('\nCalculating Boosting X Val Score...')
boost_x_val_score = cross_val_score(boost, X_train, y_train, cv = 10, scoring = 'accuracy')
boost_x_val_score = np.round(boost_x_val_score.mean()*100, 2)
print('Done.')

'''Compare boosting ensemble score with best base models scores.'''
xgbc_best_score = 82.27  # xgbc's x_val_score.
boost_vs_base_score = pd.DataFrame({'Boosting_vs_base_score(%)': [boost_x_val_score, abc_best_score, gbc_best_score, xgbc_best_score]})
'''So basically we're comparing boosting x_val_score with base models's tunned score except xgbc.'''
boost_vs_base_score.index = ['Boost', 'ABC', 'GBC', 'XGBC']
bold('**Boosting vs Base Models Scores:**')
display(boost_vs_base_score)
suppress_warnings()
'''Perform blending in mlens.'''
from mlens.ensemble import BlendEnsemble
'''Initialize blend ensembler.'''
blend = BlendEnsemble(n_jobs = -1, test_size = 0.5, random_state = seed)
'''Base models for blending.'''
base_models = [gbc, rf, dt, knn, abc]
blend.add(base_models)
'''Meta learner for blending. We will use lr.'''
blend.add_meta(lr)
'''Train the blend ensemble.'''
print('Fitting Blending...')
display(blend.fit(X_train, y_train))
print('Done.')
'''Import stacking method from vecstack.'''
from vecstack import stacking
from sklearn.metrics import accuracy_score
'''Initialize base models. We will use the same base models as blending.'''
base_models = [rf, dt, gbc, abc, knn]
'''Perform stacking.'''
S_train, S_test = stacking(base_models,                # list of base models
                           X_train, y_train, X_test,   # data
                           regression = False,         # classification task (if you need 
                                                       # regression - set to True)
                           mode = 'oof_pred_bag',      # mode: oof for train set, predict test 
                                                       # set in each fold and vote
                           needs_proba = False,        # predict class labels (if you need 
                                                       # probabilities - set to True) 
                           save_dir = None,            # do not save result and log (to save 
                                                       # in current dir - set to '.')
                           metric = accuracy_score,    # metric: callable
                           n_folds = 10,               # number of folds
                           stratified = True,          # stratified split for folds
                           shuffle = True,             # shuffle the data
                           random_state=  seed,        # ensure reproducibility
                           verbose = 1)                # print progress
'''Input features for meta learner.'''
display(S_train[:5])
display(S_train.shape)
'''Test (prediction) set for meta learner.'''
display(S_test[:5].shape)
display(S_test.shape)
"""Initialize 1st level model that is our meta learner. We will use lr."""
super_learner = lr 
    
'''Fit meta learner on the output of base learners.'''
print('Fitting Stacking...')
super_learner.fit(S_train, y_train)
print('Done.')
'''Finally predict using super learner.'''
y_pred_super = super_learner.predict(S_test)
'''Predicting with different ensembles.'''

'''Hard voting.'''
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": hard_vct.predict(X_test)})
submission.to_csv('submission_hard_vct.csv', index = False)

'''Soft voting.'''
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": soft_vct.predict(X_test)})
submission.to_csv('submission_soft_vct.csv', index = False)

'''Bagging.'''
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": bagg.predict(X_test)})
submission.to_csv('submission_bagg.csv', index = False)

'''Boosting.'''
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": boost.predict(X_test)})
submission.to_csv('submission_boost.csv', index = False)

'''Blending.'''
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": blend.predict(X_test).astype(int)})
submission.to_csv('submission_blend.csv', index = False)

'''Stacking.'''
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_pred_super.astype(int)})
submission.to_csv('submission_super.csv', index = False)
suppress_warnings()
'''Create a df of different ensemble submission scores and base models.'''
submission_score = pd.DataFrame({'Ensembles or Base Models':['Bagging(en)', 'Boosting(en)', 'Blending(en)', 'Stacking(en)', 'Hard_voting(en)', 'Soft_voting(en)', 'RF(base)', 'GBC(base)'],
             'Scored_on_submission(%)':[80.861, 78.468, 79.425, 79.425, 78.947, 78.468, 79.425, 78.468 ]})
submission_score = submission_score.set_index('Ensembles or Base Models').sort_values(by = 'Scored_on_submission(%)', ascending = False)
bold("**Ensembles' vs Base Models' Scores on Submission:**")
display(submission_score)
"""Let's plot models' submission score for the last time.
We would use plotly with cufflinks (used to create plotly charts from pandas series or data frame)."""
import cufflinks as cf
cf.go_offline() # Required to use plotly offline with cufflinks.
py.init_notebook_mode() # Graphs charts inline (jupyter notebook).

'''Plot bar plot.'''
figure = submission_score.iplot(kind = 'bar', asFigure = True, title = 'Models Scored on Submission', theme = 'solar')
iplot(figure)
'''Edit the figure object.'''
figure
"""Let's update the figure."""
figure['data'][0]['text'] = submission_score.values.flatten().tolist()
figure['data'][0]['textposition'] = 'auto'
figure['data'][0]['textfont'] = dict(family = 'sans serif', size = 18, color = 'aquamarine')
figure['data'][0]['hoverinfo'] = 'none'
iplot(figure)