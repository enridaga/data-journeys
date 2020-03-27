
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pylab
import seaborn as sns
plt.style.use('fivethirtyeight')
#%matplotlib inline
pylab.rcParams['figure.figsize'] = (10.0, 8.0)
loans_df = pd.read_csv("../input/kiva_loans.csv", parse_dates=['disbursed_time', 'funded_time', 'posted_time'])
loans_df.shape
loans_df.head()
# From: https://deparkes.co.uk/2016/11/04/sort-pandas-boxplot/
def boxplot_sorted(df, by, column):
    # use dict comprehension to create new dataframe from the iterable groupby object
    # each group name becomes a column in the new dataframe
    df2 = pd.DataFrame({col:vals[column] for col, vals in df.groupby(by)})
    # find and sort the median values in this new dataframe
    meds = df2.median().sort_values()
    # use the columns in the dataframe, ordered sorted by median value
    # return axes so changes can be made outside the function
    return df2[meds.index].boxplot(return_type="axes")
pylab.rcParams['figure.figsize'] = (24.0, 8.0)
plt.style.use('fivethirtyeight')
loans_df.groupby(loans_df.country).id.count().sort_values().plot.bar(color='cornflowerblue');
plt.title("Loan Count by Country");
pylab.rcParams['figure.figsize'] = (8.0, 8.0)
loans_df.groupby(loans_df.sector).id.count().sort_values().plot.bar(color='cornflowerblue');
plt.title("Loan Count by Sector");
loans_df.term_in_months.plot.hist(bins=100);
plt.title("Loan Count by Loan Duration");
loans_df.lender_count.plot.box();
plt.title("Distribution of Number of Lenders per loan");
axes = plt.gca()
axes.set_xlim([0,500])
loans_df.lender_count.plot.hist(bins=1000);
plt.title("Distribution of Number of Lenders where number < 500");
max(loans_df.lender_count)
def process_gender(x):
    
    if type(x) is float and np.isnan(x):
        return "nan"
    genders = x.split(",")
    male_count = sum(g.strip() == 'male' for g in genders)
    female_count = sum(g.strip() == 'female' for g in genders)
    
    if(male_count > 0 and female_count > 0):
        return "MF"
    elif(female_count > 0):
        return "F"
    elif (male_count > 0):
        return "M"
loans_df.borrower_genders = loans_df.borrower_genders.apply(process_gender)
loans_df.borrower_genders.value_counts().plot.bar(color='cornflowerblue');
plt.title("Loan Count by Gender of Borrower");
loans_df.funded_amount.plot.box();
plt.title("Distribution of Loan Funded Amount");
# Q3 + 1.5 * IQR
IQR = loans_df.funded_amount.quantile(0.75) - loans_df.funded_amount.quantile(0.25)
upper_whisker = loans_df.funded_amount.quantile(0.75) + 1.5 * IQR
loans_above_upper_whisker = loans_df[loans_df.funded_amount > upper_whisker]
loans_above_upper_whisker.shape
# percentage of loans above upper whisker
loans_above_upper_whisker.shape[0]/loans_df.shape[0]
loans_below_upper_whisker = loans_df[loans_df.funded_amount < upper_whisker]
loans_below_upper_whisker.funded_amount.plot.hist();
plt.title("Distribution of Loan Funded amount < $2000");
df = loans_above_upper_whisker[loans_above_upper_whisker.funded_amount < 20000]
df.funded_amount.plot.hist();
plt.title("Distribution of Loan Funded Amount between \$2,000 and \$20,000");
df.shape
df = loans_above_upper_whisker[(loans_above_upper_whisker.funded_amount > 20000) & (loans_above_upper_whisker.funded_amount < 60000)]
df.funded_amount.plot.hist()
plt.title("Distribution of Loan Funded Amount between \$20,000 and \$60,000");
df.shape
df.sector.value_counts().sort_values().plot.bar(color='cornflowerblue');
plt.title("Loan Count by Sector for Loan Amount between \$20,000 and \$60,000");
loans_df[loans_df.funded_amount > 60000]
pylab.rcParams['figure.figsize'] = (16.0, 8.0)
boxplot_sorted(loans_df[loans_df.funded_amount < 10000], by=["sector"], column="funded_amount");
plt.xticks(rotation=90);
pylab.rcParams['figure.figsize'] = (8.0, 8.0)
boxplot_sorted(loans_df[(loans_df.funded_amount < 10000) & (loans_df.borrower_genders != "nan")], by=["borrower_genders"], column="funded_amount");
loan_amount_values = loans_df[(loans_df.funded_amount < 10000) & (loans_df.borrower_genders != "nan")].groupby("borrower_genders").loan_amount
loan_amount_values.median()
loan_amount_values.quantile(0.75) - loan_amount_values.quantile(0.25)
pylab.rcParams['figure.figsize'] = (24.0, 8.0)
boxplot_sorted(loans_df[(loans_df.funded_amount < 10000) & (loans_df.borrower_genders != "nan")], by=["country"], column="funded_amount");
plt.xticks(rotation=90);
loans_df[loans_df.country == 'Afghanistan']
pylab.rcParams['figure.figsize'] = (8.0, 8.0)
loans_df[loans_df.country == 'Chile'].sector.value_counts().plot.bar(color='cornflowerblue');
time_to_fund = (loans_df.funded_time - loans_df.posted_time)
time_to_fund_in_days = (time_to_fund.astype('timedelta64[s]')/(3600 * 24))
loans_df = loans_df.assign(time_to_fund=time_to_fund)
loans_df = loans_df.assign(time_to_fund_in_days=time_to_fund_in_days)


max(time_to_fund_in_days)
lower = loans_df.time_to_fund_in_days.quantile(0.01)
upper = loans_df.time_to_fund_in_days.quantile(0.99)
loans_df[(loans_df.time_to_fund_in_days > lower)].time_to_fund_in_days.plot.hist();
loans_df[(loans_df.time_to_fund_in_days > 100)].shape
loans_df[(loans_df.time_to_fund_in_days > 100)].shape[0]/loans_df.shape[0]
loans_df[(loans_df.time_to_fund_in_days > 100)].time_to_fund_in_days.plot.hist();
pylab.rcParams['figure.figsize'] = (8.0, 8.0)
boxplot_sorted(loans_df[loans_df.borrower_genders != 'nan'], by=["borrower_genders"], column="time_to_fund_in_days");
pylab.rcParams['figure.figsize'] = (24.0, 8.0)
#loans_df[["time_to_fund_in_days", "country"]].boxplot(by="country");
axes = boxplot_sorted(loans_df, by=["country"], column="time_to_fund_in_days")
axes.set_title("Time to Fund by country in days")
plt.xticks(rotation=90);
