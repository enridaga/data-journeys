
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
import seaborn as sns
from matplotlib_venn import venn3, venn2
warnings.filterwarnings("ignore")
import matplotlib.pylab as plt
plt.style.use('fivethirtyeight')
#%matplotlib inline

# Read Multiple Choice Responses
mc = pd.read_csv('../input/multipleChoiceResponses.csv')
# Data Prep

# Pull just IDE Questions
ide_qs = mc[['Q13_Part_1','Q13_Part_2','Q13_Part_3','Q13_Part_4','Q13_Part_5',
             'Q13_Part_6','Q13_Part_7','Q13_Part_8','Q13_Part_9','Q13_Part_10',
             'Q13_Part_11','Q13_Part_12','Q13_Part_13','Q13_Part_14','Q13_Part_15']].drop(0)


# Rename Columns for IDE Type
column_rename = {'Q13_Part_1': 'Jupyter/IPython',
                 'Q13_Part_2': 'RStudio',
                'Q13_Part_3': 'PyCharm',
                'Q13_Part_4': 'Visual Studio Code',
                'Q13_Part_5': 'nteract',
                'Q13_Part_6': 'Atom',
                'Q13_Part_7': 'MATLAB',
                'Q13_Part_8': 'Visual Studio',
                'Q13_Part_9': 'Notepad++',
                'Q13_Part_10': 'Sublime Text',
                'Q13_Part_11': 'Vim',
                'Q13_Part_12': 'IntelliJ',
                'Q13_Part_13': 'Spyder',
                'Q13_Part_14': 'None',
                'Q13_Part_15': 'Other',
                }

# Make binary columns from IDE answers.
ide_qs_binary = ide_qs.rename(columns=column_rename).fillna(0).replace('[^\\d]',1, regex=True)
mc_and_ide = pd.concat([mc.drop(0), ide_qs_binary], axis=1)
color_pal = sns.color_palette("hls", 16)
ide_qs_binary_drop_noresponse = ide_qs_binary.copy()
ide_qs_binary_drop_noresponse['no reponse'] = ide_qs_binary_drop_noresponse.sum(axis=1).apply(lambda x: 1 if x == 0 else 0)
ide_qs_binary_drop_noresponse = ide_qs_binary_drop_noresponse.loc[ide_qs_binary_drop_noresponse['no reponse'] == 0].drop('no reponse', axis=1).copy()

plot_df = ((ide_qs_binary_drop_noresponse.sum() / ide_qs_binary_drop_noresponse.count()).sort_values() * 100 ).round(2)
ax = plot_df.plot(kind='barh', figsize=(10, 10),
          title='2018 Kaggle Survey IDE Use (Excluding Non-Respondents)',
          color=color_pal)
for i, (p, pr) in enumerate(zip(plot_df.index, plot_df.values)):
    #plt.text(s=p, x=1, y=i, color="w", verticalalignment="center", size=18)
    plt.text(s=str(pr)+"%", x=pr-5, y=i, color="w",
             verticalalignment="center", horizontalalignment="left", size=10)
ax.set_xlabel("% of Respondents")
plt.show()
# Results manually pulled from report found here: https://insights.stackoverflow.com/survey/2018/
stackoverflow_results = {'Visual Studio Code': 34.9,
                         'Visual Studio':  34.3,
                         'Notepad++': 34.2,
                         'Sublime Text': 28.9,
                         'Vim': 25.8,
                         'IntelliJ': 24.9,
                         'Android Studio': 19.3,
                         'Eclipse': 18.9,
                         'Atom': 18.0,
                         'PyCharm': 12.0,
                         'Xcode': 10.6,
                         'PHPStorm': 9.0,
                         'NetBeans': 8.2,
                         'IPython / Jupyter': 7.4,
                         'Emacs': 4.1,
                         'RStudio': 3.3,
                         'RubyMine': 1.6,
                         'TextMate': 1.1,
                         'Coda': 0.6,
                         'Komodo': 0.6,
                         'Zend': 0.4,
                         'Light Table': 0.2,
                         }
so_df = pd.DataFrame(stackoverflow_results, index=[0]).T.sort_values(0)

# Colors
color_pal = sns.color_palette("Set1", 5)
Jupyter_color = color_pal[0]
Rstudio_color = color_pal[1]
notepad_color = color_pal[2]
vs_code_color = color_pal[3]
vs_color = color_pal[4]
# Kaggle
fig = plt.figure(figsize=(18, 10))
plt.subplot(1, 2, 1)
plot_df = ((ide_qs_binary_drop_noresponse.sum() / ide_qs_binary_drop_noresponse.count()).sort_values() * 100 ).round(2)
ax1 = plot_df.plot(kind='barh',
          title='2018 Kaggle Survey',
          color=['k','k','k','k','k',
                 'k',vs_code_color,vs_color,'k','k',
                 'k','k',notepad_color,Rstudio_color,Jupyter_color])
for i, (p, pr) in enumerate(zip(plot_df.index, plot_df.values)):
   #plt.text(s=p, x=1, y=i, color="w", verticalalignment="center", size=18)
    if pr > 10:
        plt.text(s=str(pr)+"%", x=pr-8, y=i, color="w",
                 verticalalignment="center", horizontalalignment="left", size=12)
ax1.set_xlabel("% of Respondents")
# Stackoverflow
plt.subplot(1, 2, 2)

ax2 = so_df[0].plot(kind='barh',
          title='2018 Stackoverflow Survey IDE Use',
          color=['k','k','k','k','k',
                 'k',Rstudio_color,'k',Jupyter_color,'k',
                 'k','k','k','k','k',
                 'k','k','k','k',notepad_color,
                 vs_color,vs_code_color],
          legend=False,
          ax=plt.gca())
plt.set_cmap('Blues')
ax2.set_xlabel("% of Respondents")
for i, (p, pr) in enumerate(zip(so_df.index, so_df[0].values)):
    if pr > 3:
        plt.text(s=str(pr)+"%", x=pr-3, y=i, color="w",
             verticalalignment="center", horizontalalignment="left", size=12)
plt.suptitle('Kaggle vs Stackoverflow IDE Use (Top 3 of Each Colored)', fontsize = 20, y=1.02)
plt.tight_layout()
# plt.style.use('fivethirtyeight')
plt.show()
plt.figure(figsize=(15, 8))

venn3(subsets=(len(ide_qs_binary.loc[(ide_qs_binary['Jupyter/IPython'] == 1) & (ide_qs_binary['RStudio'] == 0) & (ide_qs_binary['Notepad++'] == 0)]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter/IPython'] == 0) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['Notepad++'] == 0)]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter/IPython'] == 1) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['Notepad++'] == 0)]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter/IPython'] == 0) & (ide_qs_binary['RStudio'] == 0) & (ide_qs_binary['Notepad++'] == 1)]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter/IPython'] == 1) & (ide_qs_binary['RStudio'] == 0) & (ide_qs_binary['Notepad++'] == 1)]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter/IPython'] == 0) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['Notepad++'] == 1)]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter/IPython'] == 1) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['Notepad++'] == 1)])),
      set_labels=('Jupyter', 'RStudio', 'Notepad++'))
plt.title('Jupyter vs RStudio vs Notepad++ (All users)')
plt.tight_layout()
plt.show()
# Venn Diagram of 
plt.figure(figsize=(15, 8))
plt.subplot(1, 2, 1)
venn3(subsets=(len(ide_qs_binary.loc[(ide_qs_binary['Jupyter/IPython'] == 1) & (ide_qs_binary['RStudio'] == 0) & (ide_qs_binary['Notepad++'] == 0 & (mc_and_ide['Q6'] == 'Student'))]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter/IPython'] == 0) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['Notepad++'] == 0)& (mc_and_ide['Q6'] == 'Student')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter/IPython'] == 1) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['Notepad++'] == 0)& (mc_and_ide['Q6'] == 'Student')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter/IPython'] == 0) & (ide_qs_binary['RStudio'] == 0) & (ide_qs_binary['Notepad++'] == 1)& (mc_and_ide['Q6'] == 'Student')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter/IPython'] == 1) & (ide_qs_binary['RStudio'] == 0) & (ide_qs_binary['Notepad++'] == 1)& (mc_and_ide['Q6'] == 'Student')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter/IPython'] == 0) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['Notepad++'] == 1)& (mc_and_ide['Q6'] == 'Student')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter/IPython'] == 1) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['Notepad++'] == 1& (mc_and_ide['Q6'] == 'Student'))])),
      set_labels=('Jupyter', 'RStudio', 'Notepad++'))
plt.title('Students IDE Use')
plt.subplot(1, 2, 2)
venn3(subsets=(len(ide_qs_binary.loc[(ide_qs_binary['Jupyter/IPython'] == 1) & (ide_qs_binary['RStudio'] == 0) & (ide_qs_binary['Notepad++'] == 0) & (mc_and_ide['Q6'] != 'Student')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter/IPython'] == 0) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['Notepad++'] == 0)& (mc_and_ide['Q6'] != 'Student')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter/IPython'] == 1) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['Notepad++'] == 0)& (mc_and_ide['Q6'] != 'Student')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter/IPython'] == 0) & (ide_qs_binary['RStudio'] == 0) & (ide_qs_binary['Notepad++'] == 1)& (mc_and_ide['Q6'] != 'Student')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter/IPython'] == 1) & (ide_qs_binary['RStudio'] == 0) & (ide_qs_binary['Notepad++'] == 1)& (mc_and_ide['Q6'] != 'Student')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter/IPython'] == 0) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['Notepad++'] == 1)& (mc_and_ide['Q6'] != 'Student')]),
               len(ide_qs_binary.loc[(ide_qs_binary['Jupyter/IPython'] == 1) & (ide_qs_binary['RStudio'] == 1) & (ide_qs_binary['Notepad++'] == 1& (mc_and_ide['Q6'] != 'Student'))])),
      set_labels=('Jupyter', 'RStudio', 'Notepad++'))
plt.title('Non-Students IDE Use')
plt.tight_layout()
plt.show()
plt.figure(figsize=(15, 8))
venn3(subsets=(len(ide_qs_binary.loc[(ide_qs_binary['Sublime Text'] == 1) & (ide_qs_binary['PyCharm'] == 0) & (ide_qs_binary['Notepad++'] == 0)]),
               len(ide_qs_binary.loc[(ide_qs_binary['Sublime Text'] == 0) & (ide_qs_binary['PyCharm'] == 1) & (ide_qs_binary['Notepad++'] == 0)]),
               len(ide_qs_binary.loc[(ide_qs_binary['Sublime Text'] == 1) & (ide_qs_binary['PyCharm'] == 1) & (ide_qs_binary['Notepad++'] == 0)]),
               len(ide_qs_binary.loc[(ide_qs_binary['Sublime Text'] == 0) & (ide_qs_binary['PyCharm'] == 0) & (ide_qs_binary['Notepad++'] == 1)]),
               len(ide_qs_binary.loc[(ide_qs_binary['Sublime Text'] == 1) & (ide_qs_binary['PyCharm'] == 0) & (ide_qs_binary['Notepad++'] == 1)]),
               len(ide_qs_binary.loc[(ide_qs_binary['Sublime Text'] == 0) & (ide_qs_binary['PyCharm'] == 1) & (ide_qs_binary['Notepad++'] == 1)]),
               len(ide_qs_binary.loc[(ide_qs_binary['Sublime Text'] == 1) & (ide_qs_binary['PyCharm'] == 1) & (ide_qs_binary['Notepad++'] == 1)])),
      set_labels=('Sublime Text', 'PyCharm', 'Notepad++'))
plt.title('Sublime Text vs PyCharm vs Notepad++ (All)')
plt.show()
# Pull just question6
ide_by_q6 = mc_and_ide \
    .rename(columns={'Q6':'Job Title'}) \
    .groupby('Job Title')['Jupyter/IPython','RStudio','PyCharm','Visual Studio Code',
                   'nteract','Atom','MATLAB','Visual Studio','Notepad++','Sublime Text',
                   'Vim','IntelliJ','Spyder','None','Other'] \
    .mean()

def magnify():
    return [dict(selector="th",
                 props=[("font-size", "8pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "9pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '9pt')])
]
np.random.seed(25)
cmap = cmap=sns.diverging_palette(5, 250, as_cmap=True)
#bigdf = pd.DataFrame(np.random.randn(20, 25)).cumsum()
ide_by_q6.T.sort_values('Data Analyst', ascending=False).T \
    .sort_values('RStudio', ascending=False) \
    .rename(columns={'Jupyter/IPython': 'Jupyter',
                     'Visual Studio':'VStudio',
                     'Visual Studio Code': "VSCode",
                     'Sublime Text': 'Sublime'}) \
    [['Jupyter','RStudio','Notepad++','PyCharm','Spyder','Sublime','MATLAB','VStudio','VSCode']] \
    .sort_index() \
    .style.background_gradient(cmap, axis=1)\
    .set_precision(2)\
    .format("{:.0%}")
# Make Salary into a categorical so it can be sorted
salary_ordered = ['0-10,000' ,
                    '10-20,000',
                    '20-30,000',
                    '30-40,000',
                    '40-50,000',
                    '50-60,000',
                    '60-70,000',
                    '70-80,000',
                    '80-90,000',
                    '90-100,000',
                    '100-125,000',
                    '125-150,000',
                    '150-200,000',
                    '200-250,000',
                    '250-300,000',
                    '300-400,000',
                    '400-500,000',
                    '500,000+',
                  #  'I do not wish to disclose my approximate yearly compensation'
                 ]
mc_and_ide['Salary'] = pd.Categorical(mc_and_ide['Q9'], salary_ordered)


ide_salary_breakdown = mc_and_ide.groupby('Salary')['Jupyter/IPython','RStudio','PyCharm','Visual Studio Code',
                   'nteract','Atom','MATLAB','Visual Studio','Notepad++','Sublime Text',
                   'Vim','IntelliJ','Spyder','None','Other'].mean().sort_index()

ide_salary_breakdown['Mean Salary'] = [5, 15, 25, 35, 45, 55, 65, 75,
                                       85, 95, 112.500, 137.500, 175.000, 225.000, 275.000,
                                       350.000, 450.000, 550.000]

# Make the plots
fig, axes = plt.subplots(nrows=5, ncols=3, sharex=True, sharey=True, figsize=(15, 15))
color_pal = sns.color_palette("husl", 16)
n = 1
for col in ide_salary_breakdown.set_index('Mean Salary').columns:
    #print(col)
    plt.subplot(5, 3, n)
    (ide_salary_breakdown.set_index('Mean Salary')[col] * 100) \
        .plot(title=col, xlim=(20,300), color=color_pal[n])
    plt.ylabel('% Use')
    plt.xlabel('Mean Salary ($1,000)')
    n += 1
plt.subplots_adjust(hspace = 0.5)
plt.suptitle('IDE Use (% of respondents) by Salary (\$20k-$300k)', size=20)
fig.tight_layout()
fig.subplots_adjust(top=0.93)
plt.show()
salary_ide_counts = mc_and_ide.groupby('Salary')['Jupyter/IPython', 'RStudio', 'PyCharm',
                             'Visual Studio Code', 'nteract', 'Atom',
                             'MATLAB', 'Visual Studio', 'Notepad++',
                             'Sublime Text', 'Vim', 'IntelliJ', 'Spyder',
                             'None', 'Other'] \
    .sum() \
    .T \
    .sort_values('90-100,000', ascending=False) \
    .T \
    .sort_index() \
    .T

salary_ide_counts.columns = [str(col) for col in salary_ide_counts.columns]
salary_ide_counts['\$250k+'] = salary_ide_counts[['250-300,000', '300-400,000', '400-500,000', '500,000+']].sum(axis=1)
salary_ide_counts['\$150k-$250k'] = salary_ide_counts[['150-200,000', '200-250,000']].sum(axis=1)
salary_ide_counts['\$100k-$150k'] = salary_ide_counts[['100-125,000', '125-150,000']].sum(axis=1)
salary_ide_counts['\$80k-$100k'] = salary_ide_counts[['80-90,000', '90-100,000']].sum(axis=1)
salary_ide_counts['\$60k-$80k'] = salary_ide_counts[['60-70,000', '70-80,000']].sum(axis=1)
salary_ide_counts['\$40k-$60k'] = salary_ide_counts[['40-50,000', '50-60,000']].sum(axis=1)
salary_ide_counts['\$20k-$40k'] = salary_ide_counts[['20-30,000', '30-40,000']].sum(axis=1)
salary_ide_counts['\$0-$20k'] = salary_ide_counts[['0-10,000', '10-20,000']].sum(axis=1)

salary_ide_counts[['\$0-$20k','\$20k-$40k','\$40k-$60k','\$60k-$80k','\$80k-$100k','\$100k-$150k','\$150k-$250k','\$250k+']] \
    .style.background_gradient(cmap)
# Make a sorted list of country by number of responses.
country_sorted = mc_and_ide.groupby('Q3').count().sort_values('Q1', ascending=False).index

country_ide_stats = mc_and_ide.groupby('Q3')['Jupyter/IPython','RStudio','PyCharm',
                                             'Visual Studio Code',
                   'nteract','Atom','MATLAB','Visual Studio','Notepad++','Sublime Text',
                   'Vim','IntelliJ','Spyder','None','Other'] \
    .mean()
country_ide_stats.index = pd.Categorical(country_ide_stats.index, country_sorted)
country_ide_stats.sort_index()[:10] \
    .rename({'United Kingdom of Great Britain and Northern Ireland':'GB/N.Ireland',
             'United States of America':'USA'}) \
    .T \
    .sort_values('USA', ascending=False) \
    .style.background_gradient(cmap, axis=0).format("{:.0%}")
(country_ide_stats.sort_index() \
     .rename(columns={'Visual Studio Code': 'VSCode', 'Jupyter/IPython' : 'Jupyter'})[:6].drop('Other',axis=1) \
    .rename({'United Kingdom of Great Britain and Northern Ireland':'GB/N.Ireland',
             'United States of America':'USA'}) \
    .T \
    .sort_values('USA', ascending=False) * 100) \
    .plot(kind='bar', figsize=(15, 5), title='Top 5 Responding Countries IDE Use', rot=40)
plt.tight_layout()
plt.ylabel('% Of Respondents')
plt.show()
age_ide_counts = mc_and_ide.groupby('Q2')['Jupyter/IPython','RStudio','PyCharm','Visual Studio Code',
                   'nteract','Atom','MATLAB','Visual Studio','Notepad++','Sublime Text',
                   'Vim','Spyder'] \
    .sum()

age_ide_counts.apply(lambda x: x / x.sum() * 100, axis=1) \
    .plot(kind='bar',
          stacked=True,
          figsize=(15, 5),
          title="IDE Use by Age",
          colormap=plt.get_cmap('tab20'))
plt.style.use('ggplot')
plt.ylabel('Percent total Use')
plt.xlabel('Age')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.tight_layout()
plt.show()
color_pal = sns.color_palette("husl", 16)
fig, axes = plt.subplots(nrows=4, ncols=3, sharex=True, sharey=True, figsize=(15, 10))
n = 1
for col in age_ide_counts.columns:
    plt.subplot(4, 3, n)
    age_ide_counts[col].plot.bar(title=col, color=color_pal[n])
    plt.xlabel('Age')
    plt.ylabel('Count')
    n += 1
plt.subplots_adjust(hspace = 0.9)
#plt.style.use('fivethirtyeight')
plt.show()
color_pal = sns.color_palette("Set2", 11)

notebook_cols = []
for x in mc.columns:
    if x[:3] == 'Q14':
        notebook_cols.append(x)

notebook_qs = mc[notebook_cols]
colname_replace = {}
for x in notebook_qs.columns:
    col_newname = notebook_qs[x][0].replace('Which of the following hosted notebooks have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - ','')
    colname_replace[x] = col_newname
colname_replace['Q14_OTHER_TEXT'] = 'Text'
notebook_qs = notebook_qs.rename(columns=colname_replace).drop(0).fillna(0).replace('[^\\d]',1, regex=True)

plot_df = notebook_qs.mean().sort_values().copy() * 100
plot_df = plot_df.round(2)
plot_df.plot.barh(title = 'Which of the following hosted notebooks have you used at work or school in the last 5 years?',
                                          figsize=(10, 8),
                 color=color_pal)

for i, (p, pr) in enumerate(zip(plot_df.index, plot_df.values)):
    if pr > 2:
        plt.text(s=str(pr)+"%", x=pr-0.3, y=i, color="w",
                 verticalalignment="center", horizontalalignment="right", size=12)
ax1.set_xlabel("% of Respondents")
plt.tight_layout()
plt.show()
plt.figure(figsize=(15, 8))
c = venn3(subsets=(len(notebook_qs.loc[(notebook_qs['Kaggle Kernels'] == 1) & (notebook_qs['JupyterHub/Binder'] == 0) & (notebook_qs['Google Colab'] == 0)]),
               len(notebook_qs.loc[(notebook_qs['Kaggle Kernels'] == 0) & (notebook_qs['JupyterHub/Binder'] == 1) & (notebook_qs['Google Colab'] == 0)]),
               len(notebook_qs.loc[(notebook_qs['Kaggle Kernels'] == 1) & (notebook_qs['JupyterHub/Binder'] == 1) & (notebook_qs['Google Colab'] == 0)]),
               len(notebook_qs.loc[(notebook_qs['Kaggle Kernels'] == 0) & (notebook_qs['JupyterHub/Binder'] == 0) & (notebook_qs['Google Colab'] == 1)]),
               len(notebook_qs.loc[(notebook_qs['Kaggle Kernels'] == 1) & (notebook_qs['JupyterHub/Binder'] == 0) & (notebook_qs['Google Colab'] == 1)]),
               len(notebook_qs.loc[(notebook_qs['Kaggle Kernels'] == 0) & (notebook_qs['JupyterHub/Binder'] == 1) & (notebook_qs['Google Colab'] == 1)]),
               len(notebook_qs.loc[(notebook_qs['Kaggle Kernels'] == 1) & (notebook_qs['JupyterHub/Binder'] == 1) & (notebook_qs['Google Colab'] == 1)])),
      set_labels=('Kaggle Kernels', 'JupyterHub/Binder', 'Google Colab'))
plt.title('Hosted Notebook Used in Past 5 Years (top 3 responses)')
c.get_patch_by_id('10').set_color('orange')
plt.show()
# Load freeform responses
ff = pd.read_csv('../input/freeFormResponses.csv')

# Format into lower strings
ff['count'] = 1
ff['IDE_lower'] = ff['Q13_OTHER_TEXT'].str.lower()
ff.drop(0)[['IDE_lower','count']].groupby('IDE_lower').sum()[['count']].sort_values('count', ascending=False)

# Create wordcloud
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
plt.figure(figsize=[15,8])

# Create and generate a word cloud image:
ide_words = ' '.join(ff['IDE_lower'].drop(0).dropna().values)
wordcloud = WordCloud(colormap="tab10",
                      width=1200,
                      height=480,
                      normalize_plurals=False,
                      background_color="white",
                      random_state=5).generate(ide_words)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
# Create IDE binary dataset, dropping no responses
ide_qs_binary = ide_qs.rename(columns=column_rename).fillna(0).replace('[^\\d]',1, regex=True)
ide_qs_binary['no reponse'] = ide_qs_binary.sum(axis=1).apply(lambda x: 1 if x == 0 else 0)
ide_qs_binary = ide_qs_binary.loc[ide_qs_binary['no reponse'] == 0].drop('no reponse', axis=1).copy()

# Make the clusters using sklean's KMeans
from sklearn.cluster import KMeans
y_pred = KMeans(n_clusters=4, random_state=1).fit_predict(ide_qs_binary)
ide_qs_binary['cluster'] = y_pred

# Name the clusters
y_pred_named = ['Cluster1' if x == 0 else \
                'Cluster2' if x == 1 else \
                'Cluster3' if x == 2 else \
                'Cluster4' for x in y_pred]

ide_qs_binary['cluster_name'] = y_pred_named

cluster1 = ide_qs_binary.loc[ide_qs_binary['cluster'] == 0]
cluster2 = ide_qs_binary.loc[ide_qs_binary['cluster'] == 1]
cluster3 = ide_qs_binary.loc[ide_qs_binary['cluster'] == 2]
cluster4 = ide_qs_binary.loc[ide_qs_binary['cluster'] == 3]

ide_qs_binary = ide_qs_binary.replace({ide_qs_binary.groupby('cluster_name').sum().sort_values('Jupyter/IPython', ascending=False).iloc[0].name: 'Jupyter Lovers',
                     ide_qs_binary.groupby('cluster_name').sum().sort_values('Jupyter/IPython', ascending=True).iloc[0].name: 'Anti-Jupyters',
                     ide_qs_binary.groupby('cluster_name').sum().sort_values('RStudio', ascending=False).iloc[0].name: 'RStudio and Jupyter',
                     ide_qs_binary.groupby('cluster_name').sum().sort_values('PyCharm', ascending=False).iloc[0].name: 'Jack of All IDEs'}).copy()

mc_and_ide['cluster_name'] = ide_qs_binary['cluster_name']
mc_and_ide['cluster_name'] = mc_and_ide['cluster_name'].fillna('No Response')
mc_and_ide['count'] = 1
def ven3_jrn(df):
    #df = df.rename({'Jupyter/IPython':'Jupyter'}, axis=1)
    top_3 = d.groupby('cluster_name').sum().T.sort_values(i, ascending=False).drop('cluster').index[:3].values
    return venn3(subsets=(len(df.loc[(df[top_3[0]] == 1) & (df[top_3[1]] == 0) & (df[top_3[2]] == 0)]),
               len(df.loc[(df[top_3[0]] == 0) & (df[top_3[1]] == 1) & (df[top_3[2]] == 0)]),
               len(df.loc[(df[top_3[0]] == 1) & (df[top_3[1]] == 1) & (df[top_3[2]] == 0)]),
               len(df.loc[(df[top_3[0]] == 0) & (df[top_3[1]] == 0) & (df[top_3[2]] == 1)]),
               len(df.loc[(df[top_3[0]] == 1) & (df[top_3[1]] == 0) & (df[top_3[2]] == 1)]),
               len(df.loc[(df[top_3[0]] == 0) & (df[top_3[1]] == 1) & (df[top_3[2]] == 1)]),
               len(df.loc[(df[top_3[0]] == 1) & (df[top_3[1]] == 1) & (df[top_3[2]] == 1)])),
      set_labels=(top_3[0], top_3[1], top_3[2]))

plt.figure(figsize=(15, 10))
n = 1

for i, d in ide_qs_binary.rename({'Jupyter/IPython':'Jupyter'}, axis=1).groupby('cluster_name'):
    plt.subplot(1, 4, n)
    ven3_jrn(d)
    plt.title(i)
    n += 1
plt.show()
ide_qs_binary.groupby('cluster_name') \
    .sum() \
    .T \
    .sort_values('Jupyter Lovers', ascending=False) \
    .T \
    .drop('cluster', axis=1) \
    .plot(kind='bar', figsize=(15, 5), title='IDE Use by Cluster', rot=0, colormap=plt.get_cmap('tab20'))
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.show()
# Credit for 
color_pal = sns.color_palette("Set2", 11)

anti_jupyters = mc_and_ide.loc[mc_and_ide['cluster_name'] == 'Anti-Jupyters']
everyone_else = mc_and_ide.loc[mc_and_ide['cluster_name'] != 'Anti-Jupyters']
df = pd.DataFrame([everyone_else.groupby('Q17').count()['count'] / len(everyone_else) * 100,
              anti_jupyters.groupby('Q17').count()['count'] / len(anti_jupyters) * 100]).T
df.columns = ['Everyone Else','Anti-Jupyters']
df = df[['Anti-Jupyters', 'Everyone Else']]
df = df.sort_values('Anti-Jupyters').drop('Julia')

# Plot 1
plt.subplot(1, 2, 1)
ax1 = df['Anti-Jupyters'].plot(kind='barh', figsize=(10, 8), color=color_pal[1], title='Anti-Jupyters')
ax1_y = ax1.get_yaxis()
ax1_y.set_visible(True)
ax1.set_xlim(0,40)
ax1.invert_xaxis()
ax1.grid(axis='y')

# Plot 2
plt.subplot(1, 2, 2)
ax2 = df['Everyone Else'].plot(kind='barh', figsize=(10, 8), color=color_pal[0], title='Everyone Else')
ax2_y = ax2.get_yaxis()
ax2.grid(axis='y')
ax2_y.set_visible(False)
ax2.set_xlim(0,40)

# Title and make tight
plt.suptitle('Programming laguage used most often', fontsize=16, y=1.03, x=0.555)
plt.tight_layout()
plt.subplots_adjust(wspace=0)
ax1.set_xlabel('% of Group')
ax2.set_xlabel('% of Group')
plt.show()
anti_jupyters = mc_and_ide.loc[mc_and_ide['cluster_name'] == 'Anti-Jupyters']
everyone_else = mc_and_ide.loc[mc_and_ide['cluster_name'] != 'Anti-Jupyters']
df = pd.DataFrame([everyone_else.groupby('Q23').count()['count'] / everyone_else.groupby('Q23').count().dropna()['count'].sum() * 100,
              anti_jupyters.groupby('Q23').count()['count'] / anti_jupyters.groupby('Q23').count().dropna()['count'].sum() * 100]).T
df.columns = ['Everyone Else','Anti-Jupyters']
df = df[['Anti-Jupyters', 'Everyone Else']]
# Rename and Order the columns
df = df.rename({'0% of my time': '0%',
                '1% to 25% of my time': '1% to 25%', 
                '25% to 49% of my time': '25% to 49%',
                '50% to 74% of my time' : '50% to 74%',
                '75% to 99% of my time' : '75% to 99%',
                '100% of my time': '100%'})

df.index = pd.Categorical(df.index, ['0%',
                                     '1% to 25%', 
                                     '25% to 49%',
                                     '50% to 74%',
                                     '75% to 99%',
                                     '100%'])

df = df.sort_index()
plt.subplot(1, 2, 1)
ax = df['Anti-Jupyters'].plot(kind='bar',
                         color=color_pal[0],
                         figsize=(15, 3),
                         title='Anti-Jupyters',
                         rot=0)
for p in ax.patches:
    ax.annotate(str(p.get_height().round(1)) + '%', (p.get_x() * 1.03, p.get_height() + 1.03))

plt.ylabel('% of Group')
plt.ylim(0,35)
plt.subplot(1, 2, 2)
ax2 = df['Everyone Else'].plot(kind='bar',
                         color=color_pal[1],
                         figsize=(15, 3),
                         title='Everyone Else',
                         rot=0)
for p in ax2.patches:
    ax2.annotate(str(p.get_height().round(1)) + '%', (p.get_x() * 1.03, p.get_height() + 1.03))
    
plt.ylabel('% of Group')
plt.ylim(0,35)
plt.suptitle('Percent of time spent coding', fontsize=15, y=1.05)

plt.show()
anti_jupyters = mc_and_ide.loc[mc_and_ide['cluster_name'] == 'Anti-Jupyters']
everyone_else = mc_and_ide.loc[mc_and_ide['cluster_name'] != 'Anti-Jupyters']
df = pd.DataFrame([everyone_else.groupby('Q25').count()['count'] / everyone_else.groupby('Q25').count().dropna()['count'].sum() * 100,
              anti_jupyters.groupby('Q25').count()['count'] / anti_jupyters.groupby('Q25').count().dropna()['count'].sum() * 100]).T
df.columns = ['Everyone Else','Anti-Jupyters']

# Order the columns
df.index = pd.Categorical(df.index, ['I have never studied machine learning but plan to learn in the future',
                                     '< 1 year',
                                     '1-2 years',
                                     '2-3 years',
                                     '3-4 years',
                                     '4-5 years',
                                     '5-10 years',
                                     '10-15 years',
                                     '20+ years',
                                     'I have never studied machine learning and I do not plan to'])
df = df.sort_index(ascending=False)
df = df.rename({'I have never studied machine learning but plan to learn in the future' : 'Never but plan to',
           'I have never studied machine learning and I do not plan to': 'Dont plan to'})
plt.subplot(1, 2, 1)
plt.ylim(0,40)
plt.ylabel('% of Group')
ax = df['Anti-Jupyters'].plot(kind='bar',
                         color=color_pal[0],
                         figsize=(15, 3),
                         title='Anti-Jupyters')
for p in ax.patches:
    ax.annotate(str(p.get_height().round(1)) + '%', (p.get_x() * 1, p.get_height() + 1.05))
plt.subplot(1, 2, 2)
plt.ylim(0,40)
ax2 = df['Everyone Else'].plot(kind='bar',
                         color=color_pal[1],
                         figsize=(15, 3),
                         title='Everyone Else')
for p in ax2.patches:
    ax2.annotate(str(p.get_height().round(1)) + '%', (p.get_x() * 1, p.get_height() + 1.05))
plt.ylabel('% of Group')
plt.suptitle('How many years have you used machine learning methods?', fontsize=15, y=1.05)
plt.show()
anti_jupyters = mc_and_ide.loc[mc_and_ide['cluster_name'] == 'Anti-Jupyters']
everyone_else = mc_and_ide.loc[mc_and_ide['cluster_name'] != 'Anti-Jupyters']
df = pd.DataFrame([everyone_else.groupby('Q26').count().dropna()['count'] / everyone_else.groupby('Q26').count().dropna()['count'].sum() * 100,
              anti_jupyters.groupby('Q26').count()['count'] / anti_jupyters.groupby('Q26').count().dropna()['count'].sum() * 100]).T
df.columns = ['Everyone Else','Anti-Jupyters']

# Order the columns
df.index = pd.Categorical(df.index, ['Definitely not', 'Probably not', 'Maybe', 'Probably yes', 'Definitely yes'])
df = df.sort_index(ascending=True)

plt.subplot(1, 2, 1)
plt.ylim(0,30)
plt.ylabel('% of Group')
ax = df['Anti-Jupyters'].plot(kind='bar',
                         color=color_pal[0],
                         figsize=(15, 3),
                         title='Anti-Jupyters')
for p in ax.patches:
    ax.annotate(str(p.get_height().round(1)) + '%', (p.get_x() + 0.1, p.get_height() + 1.05))
plt.subplot(1, 2, 2)
plt.ylim(0,30)
ax2 = df['Everyone Else'].plot(kind='bar',
                         color=color_pal[1],
                         figsize=(15, 3),
                         title='Everyone Else')
for p in ax2.patches:
    ax2.annotate(str(p.get_height().round(1)) + '%', (p.get_x() + 0.1, p.get_height() + 1.05))
plt.ylabel('% of Group')
plt.suptitle('Do you consider youself a data scientist?', fontsize=15, y=1.05)
plt.show()
anti_jupyters = mc_and_ide.loc[mc_and_ide['cluster_name'] == 'Anti-Jupyters']
everyone_else = mc_and_ide.loc[mc_and_ide['cluster_name'] != 'Anti-Jupyters']
df = pd.DataFrame([everyone_else.groupby('Q46').count()['count'] / len(everyone_else) * 100,
              anti_jupyters.groupby('Q46').count()['count'] / len(anti_jupyters) * 100]).T
df.columns = ['Everyone Else','Anti-Jupyters']

# Order the columns
df.index = pd.Categorical(df.index, ['0', '0-10','10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100'
       ])
plt.subplot(1, 2, 1)
plt.ylim(0,13)
plt.ylabel('% of Group')
ax = df['Anti-Jupyters'].plot(kind='bar',
                         color=color_pal[0],
                         figsize=(15, 3),
                         title='Anti-Jupyters')
for p in ax.patches:
    ax.annotate(str(p.get_height().round(1)) + '%', (p.get_x() + 0.05, p.get_height() + 0.5))
plt.subplot(1, 2, 2)
plt.ylim(0,13)
ax2 = df['Everyone Else'].plot(kind='bar',
                         color=color_pal[1],
                         figsize=(15, 3),
                         title='Everyone Else')
for p in ax2.patches:
    ax2.annotate(str(p.get_height().round(1)) + '%', (p.get_x() + 0.005, p.get_height() + 0.5))
plt.ylabel('% of Group')
plt.suptitle('Approximately what percent of your data projects involve exploring model insights?', fontsize=15, y=1.05)
plt.show()