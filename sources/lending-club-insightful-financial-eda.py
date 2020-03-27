
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from scipy import stats

#To plot figs on jupyter
#%matplotlib inline
# figure size in inches
rcParams['figure.figsize'] = 14,6
df_loan = pd.read_csv("../input/loan.csv",low_memory=False)
def resumetable(df):
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values    
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values

    for name in summary['Name'].value_counts().index:
        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 

    return summary
resumetable(df_loan[:100000])[:60]
resumetable(df_loan[:100000])[60:112]
resumetable(df_loan[:100000])[112:]
total = len(df_loan)

plt.figure(figsize = (14,6))

g = sns.countplot(x="loan_status", data=df_loan, 
                  color='blue')
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_xlabel("Loan Status Categories", fontsize=12)
g.set_ylabel("Count", fontsize=15)
g.set_title("Loan Status Types Distribution", fontsize=20)
sizes=[]
for p in g.patches:
    height = p.get_height()
    sizes.append(height)
    g.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total*100),
            ha="center", fontsize=12) 
g.set_ylim(0, max(sizes) * 1.10)

plt.show()
df_loan = df_loan.loc[df_loan['loan_status'].isin(['Fully Paid', 'Charged Off', 'Default'])]

plt.figure(figsize=(14,6))

g = sns.countplot(x='purpose', data=df_loan, 
                  color='blue')
g.set_title("Client Purposes for Loan Credit", fontsize=22)
g.set_xlabel("Purpose Titles", fontsize=18)
g.set_ylabel('Count', fontsize=18)

sizes=[]

for p in g.patches:
    height = p.get_height()
    sizes.append(height)
    g.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total*100),
            ha="center", fontsize=14) 
    
g.set_ylim(0, max(sizes) * 1.10)
g.set_xticklabels(g.get_xticklabels(),
                  rotation=45)

plt.show()
df_loan['int_round'] = df_loan['int_rate'].round(0).astype(int)

#I will start looking the loan_amnt column
plt.figure(figsize=(14,10))

# Loan Amt plot
plt.subplot(211)
g = sns.distplot(df_loan["loan_amnt"])
g.set_xlabel("Loan Amount Value", fontsize=16)
g.set_ylabel("Frequency", fontsize=16)
g.set_title("Loan Amount Distribuition", fontsize=20)

## Interest plot
plt.subplot(212)
g1 = sns.countplot(x="int_round", data=df_loan, 
                  color='blue')
g1.set_xlabel("Loan Interest Rate", fontsize=16)
g1.set_ylabel("Count", fontsize=16)
g1.set_title("Interest Rate Distribuition", fontsize=20)
sizes=[] # Get highest values in y
for p in g1.patches:
    height = p.get_height()
    sizes.append(height)
    g1.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total*100),
            ha="center", fontsize=12) 
g1.set_ylim(0, max(sizes) * 1.10) # set y limit based on highest heights

plt.subplots_adjust(hspace = 0.4,top = 0.9)

plt.show()
df_loan.loc[df_loan.loan_status == \
            'Does not meet the credit policy. Status:Fully Paid', 'loan_status'] = 'NMCP Fully Paid'
df_loan.loc[df_loan.loan_status == \
            'Does not meet the credit policy. Status:Charged Off', 'loan_status'] = 'NMCP Charged Off'
plt.figure(figsize = (12,16))

plt.subplot(311)
g = sns.countplot(x="loan_status", data=df_loan, 
                  color='blue')
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_xlabel("Loan Status Categories", fontsize=12)
g.set_ylabel("Count", fontsize=15)
g.set_title("Loan Status Types Distribution", fontsize=20)
sizes=[]
for p in g.patches:
    height = p.get_height()
    sizes.append(height)
    g.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total*100),
            ha="center", fontsize=12) 
g.set_ylim(0, max(sizes) * 1.10)

plt.subplot(312)
g1 = sns.boxplot(x="loan_status", y="int_round", data=df_loan, 
                 color='blue')
g1.set_xticklabels(g1.get_xticklabels(),rotation=45)
g1.set_xlabel("Loan Status Categories", fontsize=12)
g1.set_ylabel("Interest Rate Distribution", fontsize=15)
g1.set_title("Loan Status by Interest Rate", fontsize=20)

plt.subplot(313)
g2 = sns.boxplot(x="loan_status", y="loan_amnt", data=df_loan, 
                 color='blue')
g2.set_xticklabels(g2.get_xticklabels(),rotation=45)
g2.set_xlabel("Loan Status Categories", fontsize=15)
g2.set_ylabel("Loan Amount Distribution", fontsize=15)
g2.set_title("Loan Status by Loan Amount", fontsize=20)

plt.subplots_adjust(hspace = 0.7,top = 0.9)

plt.show()
df_loan['issue_month'], df_loan['issue_year'] = df_loan['issue_d'].str.split('-', 1).str
months_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
df_loan['issue_month'] = pd.Categorical(df_loan['issue_month'],
                                        categories=months_order, 
                                        ordered=True)
#Issue_d x loan_amount
plt.figure(figsize = (15,16))

plt.subplot(311)
g = sns.countplot(x='issue_month', hue='issue_year', 
                  data=df_loan[df_loan['issue_year'].astype(int) >= 2012])
#g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_xlabel("Months of Year", fontsize=15)
g.set_ylabel("Count Loans", fontsize=15)
g.legend(loc='best')
g.set_title("Loan Amount by Months", fontsize=20)

plt.subplot(312)
#Looking the count of defaults though the issue_d that is The month which the loan was funded
g1 = sns.countplot(x='issue_year', hue='term', 
                   data=df_loan)
g1.set_xticklabels(g1.get_xticklabels(),rotation=90)
g1.set_xlabel("Years", fontsize=15)
g1.set_ylabel("Count Loans", fontsize=15)
g1.set_title("Total Loans by Years - With Term requested", fontsize=20)
sizes=[]
for p in g1.patches:
    height = p.get_height()
    sizes.append(height)
    g1.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total*100),
            ha="center", fontsize=12) 
g1.set_ylim(0, max(sizes) * 1.10)

plt.subplot(313)
#Looking the count of defaults though the issue_d that is The month which the loan was funded
g2 = sns.countplot(x='issue_year', data=df_loan.loc[(df_loan['loan_status'] == 'Charged Off') | 
                                                   (df_loan['loan_status'] == 'NMCP Charged Off') |
                                                   (df_loan['loan_status'] == 'Default')],
                  hue='term',)
g2.set_xticklabels(g.get_xticklabels(),rotation=90)
g2.set_xlabel("Dates", fontsize=15)
g2.set_ylabel("Count", fontsize=15)
g2.set_title("Analysing Charge Off and Default by Years", fontsize=20)
sizes=[]
for p in g2.patches:
    height = p.get_height()
    sizes.append(height)
    g2.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total*100),
            ha="center", fontsize=12) 
g2.set_ylim(0, max(sizes) * 1.10)

plt.subplots_adjust(hspace = 0.3,top = 0.9)

plt.show()
#Exploring the loan_status x purpose
purp_loan= ['purpose', 'loan_status']
cm = sns.light_palette("green", as_cmap=True)
(round(pd.crosstab(df_loan[purp_loan[0]], df_loan[purp_loan[1]], 
                   normalize='columns') * 100,2)).style.background_gradient(cmap = cm)
loan_grade = ['loan_status', 'grade']
cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_loan[loan_grade[0]], df_loan[loan_grade[1]]).style.background_gradient(cmap = cm)

loan_grade = ['loan_status', 'grade']
cm = sns.light_palette("green", as_cmap=True)
round(pd.crosstab(df_loan[loan_grade[0]], df_loan[loan_grade[1]], 
            values=df_loan['int_rate'], aggfunc='mean'),2).fillna(0).style.background_gradient(cmap = cm)
plt.figure(figsize = (13,6))

g = sns.countplot(x="verification_status", data=df_loan, 
                  color='blue')
g.set_xlabel("Loan Status Categories", fontsize=15)
g.set_ylabel("Count", fontsize=15)
g.set_title("Loan Status Types Distribution", fontsize=20)
sizes=[]
for p in g.patches:
    height = p.get_height()
    sizes.append(height)
    g.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total*100),
            ha="center", fontsize=14) 
g.set_ylim(0, max(sizes) * 1.10)

plt.show()
#Looking the 'verification_status' column that is the Indicates 
#if the co-borrowers' joint income was verified by LC, not verified, or if the income source was verified
loan_verification = ['loan_status', 'verification_status']
cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_loan[loan_verification[0]], df_loan[loan_verification[1]], 
            normalize='index').style.background_gradient(cmap = cm)
plt.figure(figsize=(12,5))

sns.distplot(df_loan['installment'])
plt.title("Installment Distribution", fontsize=20)
plt.xlabel("Installment Range", fontsize=17)
plt.ylabel("Density", fontsize=17)

plt.show()
plt.figure(figsize = (14,12))

plt.subplot(211)
g = sns.violinplot(x='loan_status', y="installment",
                   data=df_loan, color='blue')
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_xlabel("Loan Status", fontsize=17)
g.set_ylabel("Installment", fontsize=17)
g.set_title("Installment Distribution by Loan Status", fontsize=20)

plt.subplot(212)
g1 = sns.violinplot(x='loan_status', y="total_acc",
                   data=df_loan, color='blue')
g1.set_xticklabels(g.get_xticklabels(),rotation=45)
g1.set_xlabel("Loan Status", fontsize=17)
g1.set_ylabel("Total Account lines", fontsize=17)
g1.set_title("Total Account Lines Distribution by Loan Status", fontsize=20)

plt.subplots_adjust(hspace = 0.5,top = 0.9)

plt.show()
#Exploring the loan_status x Application_type
loan_application = ['loan_status', 'application_type']
cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_loan[loan_application[0]], df_loan[loan_application[1]]).style.background_gradient(cmap = cm)
plt.figure(figsize = (12,14))
#The amount and int rate x application_type 
plt.subplot(211)
g = sns.violinplot(x="application_type", y="loan_amnt",data=df_loan, 
            palette="hls")
g.set_title("Application Type - Loan Amount", fontsize=20)
g.set_xlabel("", fontsize=15)
g.set_ylabel("Loan Amount", fontsize=15)

plt.subplot(212)
g1 = sns.violinplot(x="application_type", y="int_rate",data=df_loan,
               palette="hls")
g1.set_title("Application Type - Interest Rate", fontsize=20)
g1.set_xlabel("", fontsize=15)
g1.set_ylabel("Int Rate", fontsize=15)

plt.subplots_adjust(wspace = 0.4, hspace = 0.4,top = 0.9)

plt.show()
plt.figure(figsize = (14,12))

plt.subplot(211)
g = sns.violinplot(x="home_ownership",y="loan_amnt",data=df_loan,
               kind="violin",
               split=True,palette="hls",
               hue="application_type")
g.set_title("Homer Ownership - Loan Amount Distribuition", fontsize=20)
g.set_xlabel("", fontsize=15)
g.set_ylabel("Loan Amount", fontsize=15)

plt.subplot(212)
g1 = sns.violinplot(x="home_ownership",y="int_rate",data=df_loan,
               kind="violin",
               split=True,palette="hls",
               hue="application_type")
g1.set_title("Homer Ownership - Interest Rate Distribuition", fontsize=20)
g1.set_xlabel("", fontsize=15)
g1.set_ylabel("Interest Rate", fontsize=15)

plt.subplots_adjust(hspace = 0.3,top = 0.9)

plt.show()
loan_home = ['loan_status', 'home_ownership']
cm = sns.light_palette("green", as_cmap=True)
round(pd.crosstab(df_loan[loan_home[0]], df_loan[loan_home[1]], 
            normalize='index'),2).fillna(0).style.background_gradient(cmap = cm)
# Now will start exploring the Purpose variable
plt.figure(figsize = (14,13))

plt.subplot(211)
g = sns.violinplot(x="purpose",y="int_rate",data=df_loan,
                    hue="application_type", split=True)
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_title("Purposes - Interest Rate Distribution by Application Type", fontsize=20)
g.set_xlabel("Purpose Category's", fontsize=17)
g.set_ylabel("Interest Rate Distribution", fontsize=17)

plt.subplot(212)
g1 = sns.violinplot(x="purpose",y="loan_amnt",data=df_loan,
                    hue="application_type", split=True)
g1.set_xticklabels(g1.get_xticklabels(),rotation=45)
g1.set_title("Purposes - Loan Amount Distribution by Application Type", fontsize=20)
g1.set_xlabel("Purpose Category's", fontsize=17)
g1.set_ylabel("Loan Amount Distribution", fontsize=17)

plt.subplots_adjust(hspace = 0.5,top = 0.9)

plt.show()
order_sub = df_loan.groupby("sub_grade")['int_rate'].count().index

plt.figure(figsize=(14,16))

plt.suptitle('Grade and Sub-Grade Distributions \n# Interest Rate and Loan Amount #', fontsize=22)

plt.subplot(311)
g = sns.boxplot(x="grade", y="loan_amnt", data=df_loan,
                palette="hls", hue="application_type", 
                order=["A",'B','C','D','E','F', 'G'])
g.set_xlabel("Grade Values", fontsize=17)
g.set_ylabel("Loan Amount", fontsize=17)
g.set_title("Lending Club Loan - Loan Amount Distribution by Grade", fontsize=20)
g.legend(loc='upper right')

plt.subplot(312)
g1 = sns.boxplot(x='grade', y="int_rate",data=df_loan, 
               hue="application_type", palette = "hls",  
               order=["A",'B','C','D','E','F', 'G'])
g1.set_xlabel("Grade Values", fontsize=17)
g1.set_ylabel("Interest Rate", fontsize=17)
g1.set_title("Lending Club Loan - Interest Rate Distribution by Grade", fontsize=20)

plt.subplot(313)
g2 = sns.boxenplot(x="sub_grade", y="int_rate", data=df_loan, 
                   palette="hls", order=order_sub)
g2.set_xlabel("Sub Grade Values", fontsize=15)
g2.set_ylabel("Interest Rate", fontsize=15)
g2.set_title("Lending Club Loan - Interest Rate Distribution by Sub-Grade", fontsize=20)

plt.subplots_adjust(hspace = 0.4,top = 0.9)

plt.show()
loan_grade = ['sub_grade', 'grade']
cm = sns.light_palette("green", as_cmap=True)
round(pd.crosstab(df_loan[loan_grade[0]], df_loan[loan_grade[1]], 
            values=df_loan['int_rate'], aggfunc='mean'),2).fillna(0).style.background_gradient(cmap = cm)
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import warnings
from collections import Counter
#First plot
trace0 = go.Bar(
    x = df_loan.emp_title.value_counts()[:40].index.values,
    y = df_loan.emp_title.value_counts()[:40].values,
    marker=dict(
        color=df_loan.emp_title.value_counts()[:40].values
    ),
)

data = [trace0]

layout = go.Layout(
    yaxis=dict(
        title='Count'
    ),
    xaxis=dict(
        title='Employment name'
    ),
    title='TOP 40 Employment Title'
)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='emp-title-bar')
title_mask = df_loan.emp_title.value_counts()[:20].index.values 
cm = sns.light_palette("green", as_cmap=True)

round(pd.crosstab(df_loan[df_loan['emp_title'].isin(title_mask)]['emp_title'], 
                  df_loan[df_loan['emp_title'].isin(title_mask)]['sub_grade'], 
                  normalize='index') * 100,2).style.background_gradient(cmap = cm)
#First plot
trace0 = go.Bar(
    x = df_loan.title.value_counts()[:40].index.values,
    y = df_loan.title.value_counts()[:40].values,
    marker=dict(
        color=df_loan.title.value_counts()[:40].values
    ),
)

data = [trace0]

layout = go.Layout(
    yaxis=dict(
        title='Count'
    ),
    xaxis=dict(
        title='Employment name'
    ),
    title='TOP 40 Employment Title'
)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='emp-title-bar')
# emp_lenght description: 
# Employment length in years. Possible values are between 0 and 10 where 0 means 
# less than one year and 10 means ten or more years. 

emp_ord = ['< 1 year', '1 year', '2 years', '3 years',
           '4 years', '5 years', '6 years', '7 years',
           '8 years', '9 years', '10+ years']

fig, ax = plt.subplots(2,1, figsize=(14,11))
g = sns.boxplot(x="emp_length", y="int_rate", data=df_loan,
                ax=ax[0], color='blue',
                order=emp_ord)

z = sns.violinplot(x="emp_length", y="loan_amnt",data=df_loan, 
                   ax=ax[1], color='blue',
                   order=emp_ord)
               
plt.legend(loc='upper left')
plt.show()
# Emp Length interest rate mean
df_loan.groupby(["emp_length"])['int_rate'].mean().reset_index().T
order_sub = df_loan.groupby("sub_grade")['int_rate'].count().index

plt.figure(figsize=(14,18))

plt.suptitle('Term Distributions \n# Count, Interest Rate and Loan Amount #', fontsize=22)

plt.subplot(311)
g = sns.countplot(x="term", data=df_loan,color='blue')
g.set_xlabel("Term Values", fontsize=17)
g.set_ylabel("Count", fontsize=17)
g.set_title("Lending Club Loan \nTerm Count Distribution", fontsize=20)
sizes=[]
for p in g.patches:
    height = p.get_height()
    sizes.append(height)
    g.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total*100),
            ha="center", fontsize=14) 
g.set_ylim(0, max(sizes) * 1.10)

plt.subplot(312)
g1 = sns.violinplot(x='term', y="int_rate",data=df_loan )
g1.set_xlabel("Term Values", fontsize=17)
g1.set_ylabel("Interest Rate", fontsize=17)
g1.set_title("Lending Club Loan \nInterest Rate Distribution by Term Values", fontsize=20)

plt.subplot(313)
g2 = sns.violinplot(x="term", y="loan_amnt", data=df_loan)
g2.set_xlabel("Term Values", fontsize=17)
g2.set_ylabel("Loan Amount", fontsize=17)
g2.set_title("Lending Club Loan \nLoan Amount Distribution by Term Values", fontsize=20)

plt.subplots_adjust(hspace = 0.4, top = 0.9)

plt.show()
#Exploring the State Adress x Loan Status
adress_loan = ['addr_state', 'loan_status']
cm = sns.light_palette("green", as_cmap=True)
round(pd.crosstab(df_loan[adress_loan[0]], 
                  df_loan[adress_loan[1]], 
                  normalize='all')*100,2).style.background_gradient(cmap = cm)
cols = ['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term', 'int_rate',
        'installment', 'grade', 'sub_grade', 'emp_title', 'emp_length',
        'home_ownership', 'desc', 'purpose', 'total_pymnt', 'total_pymnt_inv',]
df_loan['settlement_status'].fillna("None", inplace=True)
plt.figure(figsize=(16,18))

plt.subplot(311)
g = sns.countplot(x='settlement_status', data=df_loan, color='blue')
g.set_title('Settlement Status Distribution', fontsize=21)
g.set_ylabel("Count", fontsize=17)
g.set_xlabel("Settlement Status", fontsize=17)

sizes=[]
for p in g.patches:
    height = p.get_height()
    sizes.append(height)
    g.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total*100),
            ha="center", fontsize=14) 
g.set_ylim(0, max(sizes) * 1.10)

plt.subplot(312)
g1 = sns.violinplot(x='settlement_status', y="int_rate",
                    data=df_loan, color='blue') 
g1.set_xlabel("Settlement Status", fontsize=17)
g1.set_ylabel("Interest Rate", fontsize=17)
g1.set_title("Interest Rate Distribution by Settlement Status", 
             fontsize=20)

plt.subplot(313)
g2 = sns.violinplot(x="settlement_status", y="loan_amnt", 
                    data=df_loan, color='blue')
g2.set_xlabel("Settlement Status", fontsize=17)
g2.set_ylabel("Loan Amount", fontsize=17)
g2.set_title("Loan Amount Distribution by Settlement Status", 
             fontsize=20)

plt.subplots_adjust(hspace = 0.4, top = 0.9)

plt.show()
df_loan['loan_value'] = (100 * df_loan.revol_bal) / df_loan['revol_util']
round(pd.crosstab(df_loan['delinq_2yrs'],df_loan['loan_status'],   normalize='columns') * 100,2)[:15]
cm = sns.light_palette("green", as_cmap=True)

round((pd.crosstab(df_loan['loan_status'], df_loan['purpose'],
            values=df_loan['dti'], aggfunc='mean')).fillna(0),
      2).style.background_gradient(cmap = cm)
cm = sns.light_palette("green", as_cmap=True)

round((pd.crosstab(df_loan['loan_status'], df_loan['purpose'],
            values=df_loan['revol_util'], aggfunc='mean')).fillna(0),
      2).style.background_gradient(cmap = cm)
resumetable(df_loan.reset_index())[:45]
df_loan.select_dtypes(include=[int, float]).head()
numericals_clusters = ['loan_amnt', 'annual_inc', 'dti', 'total_acc', 'loan_status',
                       'revol_bal', 'revol_util', 'installment', 'int_rate']
resumetable(df_loan[numericals_clusters].reset_index())
df_loan[df_loan['dti'].isna()][numericals_clusters].nunique()
df_loan[df_loan['dti'] == -1][numericals_clusters]
df_loan['dti'].value_counts()[df_loan['dti'].value_counts() == 312]
