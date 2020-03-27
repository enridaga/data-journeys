
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# reading the data into a dataframe
accident_df = pd.read_csv('../input/us-accidents/US_Accidents_May19.csv')
# finding out number of rows and columns
shape = accident_df.shape
print( "Number of rows" , "{:,}".format(shape[0]) )
print( "Number of variables" , shape[1] )
# print a sample to get sense of data
accident_df.head(5)
cols = accident_df.columns
num_cols = accident_df._get_numeric_data().columns
', '.join( str(x) for x in list(num_cols) )
str_col = list( set(cols) - set(num_cols) )
', '.join( str(x) for x in str_col ) 
# Street Number is float: can't find different way to drop decimal points and then change Number column to string other than this way
accident_df['Number'] = accident_df['Number'].replace(np.nan, 0)
accident_df['Number'] = accident_df['Number'].astype(int)
accident_df['Number'] = accident_df['Number'].astype(str)
accident_df['Number'] = accident_df['Number'].replace('0', '')
accident_df['street_name_num'] = accident_df['Number'] + ' ' + accident_df['Street']
accident_df.sample(5)[ ['Number' , 'Street' , 'street_name_num'] ] 
percent_missing = accident_df.isnull().sum() * 100 / len(accident_df)
missing_value_df = pd.DataFrame({'column_name': accident_df.columns,
                                 'percent_missing': percent_missing})
missing_value_df.sort_values('percent_missing', inplace=False, ascending = False)[missing_value_df[ 'percent_missing'] != 0]

def create_plots(columns_list, ncols=3):
    nrows= int( (len(columns_list) -1) / ncols)  + 1
    fig, axs = plt.subplots(nrows=nrows , ncols=ncols, figsize= (20,nrows*5))
    plt.subplots_adjust(hspace=0.7)
    plt.subplots_adjust(wspace=0.5)

    sns.set(style="darkgrid")

    for index ,column in enumerate( columns_list ):
        order = accident_df[column].value_counts().iloc[:10].index

        if nrows == 1:
            g = sns.countplot(accident_df[column], alpha=0.9 ,  
                         order= order,
                         ax=axs[ index ])
        else:
            g = sns.countplot(accident_df[column], alpha=0.9 ,  
                         order= order,
                         ax=axs[ int(index / ncols) ][ int(index % ncols) ])
            
        g.set_xticklabels(rotation=60, labels = order )
        g.set_title(column)
columns_list = ['City', 'County', 'State' , 'Zipcode' , 'Country' , 'street_name_num' ]
create_plots(columns_list, ncols=3)
columns_list = ['Timezone' , 'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight']
create_plots(columns_list,ncols=3)
columns_list = ['Airport_Code', 'Wind_Direction', 'Weather_Condition', 'Amenity', 'Bump', 'Crossing', 'Side', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop', 
                'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop'] 
create_plots(columns_list,ncols=3)
columns_list = ['Severity' , 'Source']
create_plots(columns_list, ncols=2)
from scipy.stats import describe

sns.set(color_codes=True)

for column_name in ['Distance(mi)', 'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']:

    mean = accident_df[column_name].mean()
    std = accident_df[column_name].std()
    min_ = accident_df[column_name].min()
    max_ = accident_df[column_name].max()
    kurt = accident_df[column_name].kurt()
    skew = accident_df[column_name].skew()

    print( column_name, ',min =' , min_ , ',max =' , max_ , ',avg =' , mean , ',std =' , std, ',skewness =' , skew, ',kurtosis =' , kurt , end='\n')
    print()
fig, axs = plt.subplots(nrows=1 , ncols=2, figsize= (10,5))

sns.distplot(  accident_df[  accident_df['Precipitation(in)'].isnull() == False ]['Precipitation(in)'] , ax=axs[ 0 ])
sns.distplot(  accident_df[  accident_df['Temperature(F)'].isnull() == False ]['Temperature(F)'] , ax=axs[ 1 ])