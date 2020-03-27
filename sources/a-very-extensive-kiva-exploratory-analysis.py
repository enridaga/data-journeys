
import pandas as pd # package for high-performance, easy-to-use data structures and data analysis
import numpy as np # fundamental package for scientific computing with Python
import matplotlib
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
import plotly.tools as tls
import squarify
# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings("ignore")

# Print all rows and columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
kiva_loans_data = pd.read_csv("../input/kiva_loans.csv")
kiva_mpi_locations_data = pd.read_csv("../input/kiva_mpi_region_locations.csv")
loan_theme_ids_data = pd.read_csv("../input/loan_theme_ids.csv")
loan_themes_by_region_data = pd.read_csv("../input/loan_themes_by_region.csv")
print("Size of kiva_loans_data",kiva_loans_data.shape)
print("Size of kiva_mpi_locations_data",kiva_mpi_locations_data.shape)
print("Size of loan_theme_ids_data",loan_theme_ids_data.shape)
print("Size of loan_themes_by_region_data",loan_themes_by_region_data.shape)
kiva_loans_data.head()
kiva_mpi_locations_data.head()
loan_theme_ids_data.head()
loan_themes_by_region_data.head()
kiva_loans_data.info()
kiva_loans_data.describe()
kiva_loans_data.describe(include=["O"])
# checking missing data in kiva_loans data 
total = kiva_loans_data.isnull().sum().sort_values(ascending = False)
percent = (kiva_loans_data.isnull().sum()/kiva_loans_data.isnull().count()).sort_values(ascending = False)
missing_kiva_loans_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_kiva_loans_data
# missing data in kiva_mpi_locations data 
total = kiva_mpi_locations_data.isnull().sum().sort_values(ascending = False)
percent = (kiva_mpi_locations_data.isnull().sum()/kiva_mpi_locations_data.isnull().count()).sort_values(ascending = False)
missing_kiva_mpi_locations_data= pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_kiva_mpi_locations_data
# missing data in loan_theme_ids data 
total = loan_theme_ids_data.isnull().sum().sort_values(ascending = False)
percent = (loan_theme_ids_data.isnull().sum()/loan_theme_ids_data.isnull().count()).sort_values(ascending = False)
missing_loan_theme_ids_data= pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_loan_theme_ids_data
# missing data in loan_themes_by_region data 
total = loan_themes_by_region_data.isnull().sum().sort_values(ascending = False)
percent = (loan_themes_by_region_data.isnull().sum()/loan_themes_by_region_data.isnull().count()).sort_values(ascending = False)
missing_loan_themes_by_region_data= pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_loan_themes_by_region_data
print("Top sectors in which more loans were given : ", len(kiva_loans_data["sector"].unique()))
print(kiva_loans_data["sector"].value_counts().head(10))
sector_name = kiva_loans_data['sector'].value_counts().head(20)
plt.figure(figsize=(15,8))
sns.barplot(sector_name.index, sector_name.values, alpha=0.9, color=color[0])
plt.xticks(rotation='vertical')
plt.xlabel('Sector Name', fontsize=12)
plt.ylabel('Number of loans were given', fontsize=12)
plt.title("Top sectors in which more loans were given", fontsize=16)
plt.show()
print("Types of repayment intervals with their count : ", len(kiva_loans_data["repayment_interval"].unique()))
print(kiva_loans_data["repayment_interval"].value_counts().head(10))
sector_name = kiva_loans_data['repayment_interval'].value_counts().head(20)
plt.figure(figsize=(15,8))
sns.barplot(sector_name.index, sector_name.values, alpha=0.9, color=color[0])
plt.xticks(rotation='vertical')
plt.xlabel('Types of repayment interval', fontsize=12)
plt.ylabel('count', fontsize=12)
plt.title("Types of repayment intervals with their count", fontsize=16)
plt.show()
kiva_loans_data['repayment_interval'].value_counts().plot(kind="pie",figsize=(12,12))
# Occurance of landmark_id in decreasing order(Top categories)
temp = pd.DataFrame(kiva_loans_data.country.value_counts().head(13))
temp.reset_index(inplace=True)
temp.columns = ['country','count']
temp
# Plot the most frequent countries
plt.figure(figsize = (9, 8))
plt.title('Most frequent countries')
sns.set_color_codes("pastel")
sns.barplot(x="country", y="count", data=temp,
            label="Count")
plt.show()
kiva_loans_data.columns
# Distribution of funded anount
sns.distplot(kiva_loans_data['funded_amount'])
plt.show() 
plt.scatter(range(kiva_loans_data.shape[0]), np.sort(kiva_loans_data.funded_amount.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('loan_amount', fontsize=12)
plt.title("Loan Amount Distribution")
plt.show()
# Distribution of loan amount
sns.distplot(kiva_loans_data['loan_amount'])
plt.show()
plt.scatter(range(kiva_loans_data.shape[0]), np.sort(kiva_loans_data.loan_amount.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('loan_amount', fontsize=12)
plt.title("Loan Amount Distribution")
plt.show()
kiva_mpi_locations_data.columns
# Distribution of world regions
fig, ax2 = plt.subplots(figsize=(10,10))
plt.xticks(rotation='vertical')
sns.countplot(x='world_region', data=kiva_mpi_locations_data)
plt.show()
#Distribution of lender count(Number of lenders contributing to loan)
print("Number of lenders contributing to loan : ", len(kiva_loans_data["lender_count"].unique()))
print(kiva_loans_data["lender_count"].value_counts().head(10))
lender = kiva_loans_data['lender_count'].value_counts().head(40)
plt.figure(figsize=(15,8))
sns.barplot(lender.index, lender.values, alpha=0.9, color=color[0])
plt.xticks(rotation='vertical')
plt.xlabel('lender count(Number of lenders contributing to loan)', fontsize=12)
plt.ylabel('count', fontsize=12)
plt.title("Distribution of lender count", fontsize=16)
plt.show()
#Distribution of Loan Activity type
lender = kiva_loans_data['activity'].value_counts().head(40)
plt.figure(figsize=(15,8))
sns.barplot(lender.index, lender.values, alpha=0.9, color=color[0])
plt.xticks(rotation='vertical', fontsize=20)
plt.xlabel('Activity', fontsize=12)
plt.ylabel('count', fontsize=12)
plt.title("Top Loan Activity type", fontsize=16)
plt.show()
#Distribution of Number of months over which loan was scheduled to be paid back
print("Number of months over which loan was scheduled to be paid back : ", len(kiva_loans_data["term_in_months"].unique()))
print(kiva_loans_data["term_in_months"].value_counts().head(10))
lender = kiva_loans_data['term_in_months'].value_counts().head(70)
plt.figure(figsize=(15,8))
sns.barplot(lender.index, lender.values, alpha=0.9, color=color[0])
plt.xticks(rotation='vertical')
plt.xlabel('Number of months over which loan was scheduled to be paid back', fontsize=12)
plt.ylabel('count', fontsize=12)
plt.title("Distribution of Number of months over which loan was scheduled to be paid back", fontsize=16)
plt.show()
plt.figure(figsize=(15,8))
count = kiva_loans_data['sector'].value_counts()
squarify.plot(sizes=count.values,label=count.index, value=count.values)
plt.title('Distribution of sectors')
plt.figure(figsize=(15,8))
count = kiva_loans_data['activity'].value_counts()
squarify.plot(sizes=count.values,label=count.index, value=count.values)
plt.title('Distribution of Activities')
plt.figure(figsize=(15,8))
count = kiva_loans_data['repayment_interval'].value_counts()
squarify.plot(sizes=count.values,label=count.index, value=count.values)
plt.title('Distribution of repayment_interval')
gender_list = []
for gender in kiva_loans_data["borrower_genders"].values:
    if str(gender) != "nan":
        gender_list.extend( [lst.strip() for lst in gender.split(",")] )
temp_data = pd.Series(gender_list).value_counts()

labels = (np.array(temp_data.index))
sizes = (np.array((temp_data / temp_data.sum())*100))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(title='Borrower Gender')
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="BorrowerGender")
kiva_loans_data.borrower_genders = kiva_loans_data.borrower_genders.astype(str)
gender_data = pd.DataFrame(kiva_loans_data.borrower_genders.str.split(',').tolist())
kiva_loans_data['sex_borrowers'] = gender_data[0]
kiva_loans_data.loc[kiva_loans_data.sex_borrowers == 'nan', 'sex_borrowers'] = np.nan
sex_mean = pd.DataFrame(kiva_loans_data.groupby(['sex_borrowers'])['funded_amount'].mean().sort_values(ascending=False)).reset_index()
print(sex_mean)
g1 = sns.barplot(x='sex_borrowers', y='funded_amount', data=sex_mean)
g1.set_title("Mean funded Amount by Gender ", fontsize=15)
g1.set_xlabel("Gender")
g1.set_ylabel("Average funded Amount(US)", fontsize=12)
f, ax = plt.subplots(figsize=(15, 5))
print("Genders count with repayment interval monthly\n",kiva_loans_data['sex_borrowers'][kiva_loans_data['repayment_interval'] == 'monthly'].value_counts())
print("Genders count with repayment interval weekly\n",kiva_loans_data['sex_borrowers'][kiva_loans_data['repayment_interval'] == 'weekly'].value_counts())
print("Genders count with repayment interval bullet\n",kiva_loans_data['sex_borrowers'][kiva_loans_data['repayment_interval'] == 'bullet'].value_counts())
print("Genders count with repayment interval irregular\n",kiva_loans_data['sex_borrowers'][kiva_loans_data['repayment_interval'] == 'irregular'].value_counts())

sns.countplot(x="sex_borrowers", hue='repayment_interval', data=kiva_loans_data).set_title('sex borrowers with repayment_intervals');
#Distribution of Kiva Field Partner Names with funding count
print("Top Kiva Field Partner Names with funding count : ", len(loan_themes_by_region_data["Field Partner Name"].unique()))
print(loan_themes_by_region_data["Field Partner Name"].value_counts().head(10))
lender = loan_themes_by_region_data['Field Partner Name'].value_counts().head(40)
plt.figure(figsize=(15,8))
sns.barplot(lender.index, lender.values, alpha=0.9, color=color[0])
plt.xticks(rotation='vertical', fontsize=14)
plt.xlabel('Field Partner Name', fontsize=18)
plt.ylabel('Funding count', fontsize=18)
plt.title("Top Kiva Field Partner Names with funding count", fontsize=25)
plt.show()
countries_funded_amount = kiva_loans_data.groupby('country').mean()['funded_amount'].sort_values(ascending = False)
print("Top Countries with funded_amount(Dollar value of loan funded on Kiva.org)(Mean values)\n",countries_funded_amount.head(10))
data = [dict(
        type='choropleth',
        locations= countries_funded_amount.index,
        locationmode='country names',
        z=countries_funded_amount.values,
        text=countries_funded_amount.index,
        colorscale='Red',
        marker=dict(line=dict(width=0.7)),
        colorbar=dict(autotick=False, tickprefix='', title='Top Countries with funded_amount(Mean value)'),
)]
layout = dict(title = 'Top Countries with funded_amount(Dollar value of loan funded on Kiva.org)',)
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False)
mpi_region_amount = round(loan_themes_by_region_data.groupby('mpi_region').mean()['amount'].sort_values(ascending = False))
print("Top mpi_region with amount(Dollar value of loans funded in particular LocationName)(Mean values)\n",mpi_region_amount.head(10))
data = [dict(
        type='choropleth',
        locations= mpi_region_amount.index,
        locationmode='country names',
        z=mpi_region_amount.values,
        text=mpi_region_amount.index,
        colorscale='Red',
        marker=dict(line=dict(width=0.7)),
        colorbar=dict(autotick=False, tickprefix='', title='Top mpi_regions with amount(Mean value)'),
)]
layout = dict(title = 'Top mpi_regions with amount(Dollar value of loans funded in particular LocationName)',)
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False)
plt.subplots(figsize=(15,7))
sector_popular_loan = pd.DataFrame(kiva_loans_data.groupby(['sector'])['loan_amount'].mean()).reset_index()
sns.barplot(x='sector',y='loan_amount',data=sector_popular_loan)
plt.xticks(rotation=90,fontsize=20)
plt.xlabel('Loan sector', fontsize=20)
plt.ylabel('Average loan amount in Dollar', fontsize=20)
plt.title('Popular loan sector in terms of loan amount', fontsize=24)
plt.show()
plt.subplots(figsize=(15,7))
sector_popular_loan = pd.DataFrame(kiva_loans_data.groupby(['activity'])['loan_amount'].mean().sort_values(ascending=False)[:15]).reset_index()
sns.barplot(x='activity',y='loan_amount',data=sector_popular_loan)
plt.xticks(rotation=90,fontsize=20)
plt.xlabel('Loan activity', fontsize=20)
plt.ylabel('Average loan amount in Dollar', fontsize=20)
plt.title('Popular loan activity in terms of loan amount', fontsize=24)
plt.show()
plt.subplots(figsize=(15,7))
sector_popular_loan = pd.DataFrame(kiva_loans_data.groupby(['country'])['loan_amount'].mean().sort_values(ascending=False)[:20]).reset_index()
sns.barplot(x='country',y='loan_amount',data=sector_popular_loan)
plt.xticks(rotation=90,fontsize=20)
plt.xlabel('Countries', fontsize=20)
plt.ylabel('Average loan amount in Dollar', fontsize=20)
plt.title('Popular countries in terms of loan amount', fontsize=24)
plt.show()
plt.subplots(figsize=(15,7))
sector_popular_loan = pd.DataFrame(kiva_loans_data.groupby(['region'])['loan_amount'].mean().sort_values(ascending=False)[:25]).reset_index()
sns.barplot(x='region',y='loan_amount',data=sector_popular_loan)
plt.xticks(rotation=90,fontsize=20)
plt.xlabel('regions(locations within countries)', fontsize=20)
plt.ylabel('Average loan amount in Dollar', fontsize=20)
plt.title('Popular regions(locations within countries) in terms of loan amount', fontsize=24)
plt.show()
from wordcloud import WordCloud

names = kiva_loans_data["country"][~pd.isnull(kiva_loans_data["country"])]
#print(names)
wordcloud = WordCloud(max_font_size=50, width=600, height=300).generate(' '.join(names))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.title("Wordcloud for country Names", fontsize=35)
plt.axis("off")
plt.show() 
kiva_loans_data['date'] = pd.to_datetime(kiva_loans_data['date'])
kiva_loans_data['date_month_year'] = kiva_loans_data['date'].dt.to_period("M")
plt.figure(figsize=(8,10))
g1 = sns.pointplot(x='date_month_year', y='loan_amount', 
                   data=kiva_loans_data, hue='repayment_interval')
g1.set_xticklabels(g1.get_xticklabels(),rotation=90)
g1.set_title("Mean Loan by Month Year", fontsize=15)
g1.set_xlabel("")
g1.set_ylabel("Loan Amount", fontsize=12)
plt.show()
kiva_loans_data['Century'] = kiva_loans_data.date.dt.year
loan = kiva_loans_data.groupby(['country', 'Century'])['loan_amount'].mean().unstack()
loan = loan.sort_values([2017], ascending=False)
f, ax = plt.subplots(figsize=(15, 20)) 
loan = loan.fillna(0)
temp = sns.heatmap(loan, cmap='Reds')
plt.show()
sector_repayment = ['sector', 'repayment_interval']
cm = sns.light_palette("red", as_cmap=True)
pd.crosstab(kiva_loans_data[sector_repayment[0]], kiva_loans_data[sector_repayment[1]]).style.background_gradient(cmap = cm)
sector_repayment = ['country', 'repayment_interval']
cm = sns.light_palette("red", as_cmap=True)
pd.crosstab(kiva_loans_data[sector_repayment[0]], kiva_loans_data[sector_repayment[1]]).style.background_gradient(cmap = cm)
#Correlation Matrix
corr = kiva_loans_data.corr()
plt.figure(figsize=(12,12))
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, annot=True, cmap='cubehelix', square=True)
plt.title('Correlation between different features')
corr
fig = plt.figure(figsize=(15,8))
ax=sns.kdeplot(kiva_loans_data['term_in_months'][kiva_loans_data['repayment_interval'] == 'monthly'] , color='b',shade=True, label='monthly')
ax=sns.kdeplot(kiva_loans_data['term_in_months'][kiva_loans_data['repayment_interval'] == 'weekly'] , color='r',shade=True, label='weekly')
ax=sns.kdeplot(kiva_loans_data['term_in_months'][kiva_loans_data['repayment_interval'] == 'irregular'] , color='g',shade=True, label='irregular')
ax=sns.kdeplot(kiva_loans_data['term_in_months'][kiva_loans_data['repayment_interval'] == 'bullet'] , color='y',shade=True, label='bullet')
plt.title('Term in months(Number of months over which loan was scheduled to be paid back) vs Repayment intervals')
ax.set(xlabel='Terms in months', ylabel='Frequency')
loan_use_in_india = kiva_loans_data['use'][kiva_loans_data['country'] == 'India']
percentages = round(loan_use_in_india.value_counts() / len(loan_use_in_india) * 100, 2)[:13]
trace = go.Pie(labels=percentages.keys(), values=percentages.values, hoverinfo='label+percent', 
                textfont=dict(size=18, color='#000000'))
data = [trace]
layout = go.Layout(width=800, height=800, title='Top 13 loan uses in India',titlefont= dict(size=20), 
                   legend=dict(x=0.1,y=-5))

fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, show_link=False)