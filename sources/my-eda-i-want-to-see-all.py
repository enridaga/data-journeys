
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import KFold
import warnings
import gc
import time
import sys
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
from sklearn import metrics

plt.style.use('seaborn')
sns.set(font_scale=2)
pd.set_option('display.max_columns', 500)
dtypes = {
        'MachineIdentifier':                                    'category',
        'ProductName':                                          'category',
        'EngineVersion':                                        'category',
        'AppVersion':                                           'category',
        'AvSigVersion':                                         'category',
        'IsBeta':                                               'int8',
        'RtpStateBitfield':                                     'float16',
        'IsSxsPassiveMode':                                     'int8',
        'DefaultBrowsersIdentifier':                            'float16',
        'AVProductStatesIdentifier':                            'float32',
        'AVProductsInstalled':                                  'float16',
        'AVProductsEnabled':                                    'float16',
        'HasTpm':                                               'int8',
        'CountryIdentifier':                                    'int16',
        'CityIdentifier':                                       'float32',
        'OrganizationIdentifier':                               'float16',
        'GeoNameIdentifier':                                    'float16',
        'LocaleEnglishNameIdentifier':                          'int8',
        'Platform':                                             'category',
        'Processor':                                            'category',
        'OsVer':                                                'category',
        'OsBuild':                                              'int16',
        'OsSuite':                                              'int16',
        'OsPlatformSubRelease':                                 'category',
        'OsBuildLab':                                           'category',
        'SkuEdition':                                           'category',
        'IsProtected':                                          'float16',
        'AutoSampleOptIn':                                      'int8',
        'PuaMode':                                              'category',
        'SMode':                                                'float16',
        'IeVerIdentifier':                                      'float16',
        'SmartScreen':                                          'category',
        'Firewall':                                             'float16',
        'UacLuaenable':                                         'float32',
        'Census_MDC2FormFactor':                                'category',
        'Census_DeviceFamily':                                  'category',
        'Census_OEMNameIdentifier':                             'float16',
        'Census_OEMModelIdentifier':                            'float32',
        'Census_ProcessorCoreCount':                            'float16',
        'Census_ProcessorManufacturerIdentifier':               'float16',
        'Census_ProcessorModelIdentifier':                      'float16',
        'Census_ProcessorClass':                                'category',
        'Census_PrimaryDiskTotalCapacity':                      'float32',
        'Census_PrimaryDiskTypeName':                           'category',
        'Census_SystemVolumeTotalCapacity':                     'float32',
        'Census_HasOpticalDiskDrive':                           'int8',
        'Census_TotalPhysicalRAM':                              'float32',
        'Census_ChassisTypeName':                               'category',
        'Census_InternalPrimaryDiagonalDisplaySizeInInches':    'float16',
        'Census_InternalPrimaryDisplayResolutionHorizontal':    'float16',
        'Census_InternalPrimaryDisplayResolutionVertical':      'float16',
        'Census_PowerPlatformRoleName':                         'category',
        'Census_InternalBatteryType':                           'category',
        'Census_InternalBatteryNumberOfCharges':                'float32',
        'Census_OSVersion':                                     'category',
        'Census_OSArchitecture':                                'category',
        'Census_OSBranch':                                      'category',
        'Census_OSBuildNumber':                                 'int16',
        'Census_OSBuildRevision':                               'int32',
        'Census_OSEdition':                                     'category',
        'Census_OSSkuName':                                     'category',
        'Census_OSInstallTypeName':                             'category',
        'Census_OSInstallLanguageIdentifier':                   'float16',
        'Census_OSUILocaleIdentifier':                          'int16',
        'Census_OSWUAutoUpdateOptionsName':                     'category',
        'Census_IsPortableOperatingSystem':                     'int8',
        'Census_GenuineStateName':                              'category',
        'Census_ActivationChannel':                             'category',
        'Census_IsFlightingInternal':                           'float16',
        'Census_IsFlightsDisabled':                             'float16',
        'Census_FlightRing':                                    'category',
        'Census_ThresholdOptIn':                                'float16',
        'Census_FirmwareManufacturerIdentifier':                'float16',
        'Census_FirmwareVersionIdentifier':                     'float32',
        'Census_IsSecureBootEnabled':                           'int8',
        'Census_IsWIMBootEnabled':                              'float16',
        'Census_IsVirtualDevice':                               'float16',
        'Census_IsTouchEnabled':                                'int8',
        'Census_IsPenCapable':                                  'int8',
        'Census_IsAlwaysOnAlwaysConnectedCapable':              'float16',
        'Wdft_IsGamer':                                         'float16',
        'Wdft_RegionIdentifier':                                'float16',
        'HasDetections':                                        'int8'
        }
%time train = pd.read_csv("../input/train.csv", dtype=dtypes)
%time test = pd.read_csv("../input/test.csv", dtype=dtypes)
print(train.shape, test.shape)
train['HasDetections'].value_counts().plot.bar()
plt.title('HasDetections(target)')
%%time
# checking missing data
total = train.isnull().sum().sort_values(ascending = False)
percent = (train.isnull().sum()/train.isnull().count()*100).sort_values(ascending = False)
missing_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_train_data.head(50)
train_small = train # train.sample(frac=0.2).copy() # not small for now
print(train_small['PuaMode'].dtypes)
sns.countplot(x='PuaMode', hue='HasDetections',data=train_small)
plt.show()
print(train_small['Census_ProcessorClass'].dtypes)
sns.countplot(x='Census_ProcessorClass', hue='HasDetections',data=train_small)
plt.show()
print(train_small['DefaultBrowsersIdentifier'].dtypes)
fig, ax = plt.subplots(1, 2, figsize=(16, 8))
sns.kdeplot(train_small.loc[train['HasDetections'] == 0, 'DefaultBrowsersIdentifier'], ax=ax[0], label='NoDetection(0)')
sns.kdeplot(train_small.loc[train['HasDetections'] == 1, 'DefaultBrowsersIdentifier'], ax=ax[0], label='HasDetection(1)')

train_small.loc[train['HasDetections'] == 0, 'DefaultBrowsersIdentifier'].hist(ax=ax[1])
train_small.loc[train['HasDetections'] == 1, 'DefaultBrowsersIdentifier'].hist(ax=ax[1])
ax[1].legend(['NoDetection(0)', 'HasDetection(1)'])

plt.show()
print(train_small['Census_IsFlightingInternal'].dtypes)
fig, ax = plt.subplots(1, 2, figsize=(16, 8))
sns.kdeplot(train_small.loc[train['HasDetections'] == 0, 'Census_IsFlightingInternal'], ax=ax[0], label='NoDetection(0)')
sns.kdeplot(train_small.loc[train['HasDetections'] == 1, 'Census_IsFlightingInternal'], ax=ax[0], label='HasDetection(1)')

train_small.loc[train['HasDetections'] == 0, 'Census_IsFlightingInternal'].hist(ax=ax[1])
train_small.loc[train['HasDetections'] == 1, 'Census_IsFlightingInternal'].hist(ax=ax[1])
ax[1].legend(['NoDetection(0)', 'HasDetection(1)'])

plt.show()
train_small.loc[train['HasDetections'] == 1, 'Census_IsFlightingInternal'].value_counts()
train_small.loc[train['HasDetections'] == 0, 'Census_IsFlightingInternal'].value_counts()
print(train_small['Census_InternalBatteryType'].dtypes)
train_small['Census_InternalBatteryType'].value_counts()
def group_battery(x):
    x = x.lower()
    if 'li' in x:
        return 1
    else:
        return 0
    
train_small['Census_InternalBatteryType'] = train_small['Census_InternalBatteryType'].apply(group_battery)
sns.countplot(x='Census_InternalBatteryType', hue='HasDetections',data=train_small)
plt.show()
null_cols_to_remove = ['DefaultBrowsersIdentifier', 'PuaMode',
                       'Census_IsFlightingInternal', 'Census_InternalBatteryType']

train.drop(null_cols_to_remove, axis=1, inplace=True)
test.drop(null_cols_to_remove, axis=1, inplace=True)
categorical_features = [
        'ProductName',                                          
        'EngineVersion',                                        
        'AppVersion',                                           
        'AvSigVersion',                                         
        'Platform',                                             
        'Processor',                                            
        'OsVer',                                                
        'OsPlatformSubRelease',                                 
        'OsBuildLab',                                           
        'SkuEdition',                                           
        'SmartScreen',                                          
        'Census_MDC2FormFactor',                                
        'Census_DeviceFamily',                                  
        'Census_PrimaryDiskTypeName',                           
        'Census_ChassisTypeName',                               
        'Census_PowerPlatformRoleName',                         
        'Census_OSVersion',                                     
        'Census_OSArchitecture',                                
        'Census_OSBranch',                                      
        'Census_OSEdition',                                     
        'Census_OSSkuName',                                     
        'Census_OSInstallTypeName',                             
        'Census_OSWUAutoUpdateOptionsName',                     
        'Census_GenuineStateName',                              
        'Census_ActivationChannel',                             
        'Census_FlightRing',                                    
]
def plot_category_percent_of_target(col):
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    cat_percent = train_small[[col, 'HasDetections']].groupby(col, as_index=False).mean()
    cat_size = train_small[col].value_counts().reset_index(drop=False)
    cat_size.columns = [col, 'count']
    cat_percent = cat_percent.merge(cat_size, on=col, how='left')
    cat_percent['HasDetections'] = cat_percent['HasDetections'].fillna(0)
    cat_percent = cat_percent.sort_values(by='count', ascending=False)[:20]
    sns.barplot(ax=ax, x='HasDetections', y=col, data=cat_percent, order=cat_percent[col])

    for i, p in enumerate(ax.patches):
        ax.annotate('{}'.format(cat_percent['count'].values[i]), (p.get_width(), p.get_y()+0.5), fontsize=20)

    plt.xlabel('% of HasDetections(target)')
    plt.ylabel(col)
    plt.show()
col = categorical_features[0]
plot_category_percent_of_target(col)
col = categorical_features[1]
plot_category_percent_of_target(col)
col = categorical_features[2]
plot_category_percent_of_target(col)
col = categorical_features[3]
plot_category_percent_of_target(col)
col = categorical_features[4]
plot_category_percent_of_target(col)
col = categorical_features[5]
plot_category_percent_of_target(col)
col = categorical_features[6]
plot_category_percent_of_target(col)
col = categorical_features[7]
plot_category_percent_of_target(col)
col = categorical_features[8]
plot_category_percent_of_target(col)
col = categorical_features[9]
plot_category_percent_of_target(col)
col = categorical_features[10]
plot_category_percent_of_target(col)
col = categorical_features[11]
plot_category_percent_of_target(col)
col = categorical_features[12]
plot_category_percent_of_target(col)
col = categorical_features[13]
plot_category_percent_of_target(col)
col = categorical_features[14]
plot_category_percent_of_target(col)
col = categorical_features[15]
plot_category_percent_of_target(col)
col = categorical_features[16]
plot_category_percent_of_target(col)
col = categorical_features[17]
plot_category_percent_of_target(col)
col = categorical_features[18]
plot_category_percent_of_target(col)
col = categorical_features[19]
plot_category_percent_of_target(col)
col = categorical_features[20]
plot_category_percent_of_target(col)
col = categorical_features[21]
plot_category_percent_of_target(col)
col = categorical_features[22]
plot_category_percent_of_target(col)
col = categorical_features[23]
plot_category_percent_of_target(col)
col = categorical_features[24]
plot_category_percent_of_target(col)
col = categorical_features[25]
plot_category_percent_of_target(col)
numeric_features = [
        'IsBeta',                                               
        'RtpStateBitfield',                                     
        'IsSxsPassiveMode',                                     
        'DefaultBrowsersIdentifier',                            
        'AVProductStatesIdentifier',                            
        'AVProductsInstalled',                                  
        'AVProductsEnabled',                                    
        'HasTpm',                                               
        'CountryIdentifier',                                    
        'CityIdentifier',                                       
        'OrganizationIdentifier',                               
        'GeoNameIdentifier',                                    
        'LocaleEnglishNameIdentifier',                          
        'OsBuild',                                              
        'OsSuite',                                              
        'IsProtected',                                          
        'AutoSampleOptIn',                                      
        'SMode',                                                
        'IeVerIdentifier',                                      
        'Firewall',                                             
        'UacLuaenable',                                         
        'Census_OEMNameIdentifier',                             
        'Census_OEMModelIdentifier',                            
        'Census_ProcessorCoreCount',                            
        'Census_ProcessorManufacturerIdentifier',               
        'Census_ProcessorModelIdentifier',                      
        'Census_PrimaryDiskTotalCapacity',                      
        'Census_SystemVolumeTotalCapacity',                     
        'Census_HasOpticalDiskDrive',                           
        'Census_TotalPhysicalRAM',                              
        'Census_InternalPrimaryDiagonalDisplaySizeInInches',    
        'Census_InternalPrimaryDisplayResolutionHorizontal',    
        'Census_InternalPrimaryDisplayResolutionVertical',      
        'Census_InternalBatteryNumberOfCharges',                
        'Census_OSBuildNumber',                                 
        'Census_OSBuildRevision',                               
        'Census_OSInstallLanguageIdentifier',                   
        'Census_OSUILocaleIdentifier',                          
        'Census_IsPortableOperatingSystem',                     
        'Census_IsFlightsDisabled',                             
        'Census_ThresholdOptIn',                                
        'Census_FirmwareManufacturerIdentifier',                
        'Census_FirmwareVersionIdentifier',                     
        'Census_IsSecureBootEnabled',                           
        'Census_IsWIMBootEnabled',                              
        'Census_IsVirtualDevice',                               
        'Census_IsTouchEnabled',                                
        'Census_IsPenCapable',                                  
        'Census_IsAlwaysOnAlwaysConnectedCapable',              
        'Wdft_IsGamer',                                         
        'Wdft_RegionIdentifier',                                
]
def plot_category_percent_of_target_for_numeric(col):
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    cat_percent = train_small[[col, 'HasDetections']].groupby(col, as_index=False).mean()
    cat_size = train_small[col].value_counts().reset_index(drop=False)
    cat_size.columns = [col, 'count']
    cat_percent = cat_percent.merge(cat_size, on=col, how='left')
    cat_percent['HasDetections'] = cat_percent['HasDetections'].fillna(0)
    cat_percent = cat_percent.sort_values(by='count', ascending=False)[:20]
    cat_percent[col] = cat_percent[col].astype('category')
    sns.barplot(ax=ax[0], x='HasDetections', y=col, data=cat_percent,  order=cat_percent[col])

    for i, p in enumerate(ax[0].patches):
        ax[0].annotate('{}'.format(cat_percent['count'].values[i]), (p.get_width(), p.get_y()+0.5), fontsize=20)

    ax[0].set_title('Barplot sorted by count', fontsize=20)

    sns.barplot(ax=ax[1], x='HasDetections', y=col, data=cat_percent)
    for i, p in enumerate(ax[0].patches):
        ax[1].annotate('{}'.format(cat_percent['count'].sort_index().values[i]), (0, p.get_y()+0.6), fontsize=20)
    ax[1].set_title('Barplot sorted by index', fontsize=20)

    plt.xlabel('% of HasDetections(target)')
    plt.ylabel(col)
    plt.subplots_adjust(wspace=0.5, hspace=0)
    plt.show()

def plot_kde_hist_for_numeric(col):
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    sns.kdeplot(train_small.loc[train['HasDetections'] == 0, col], ax=ax[0], label='NoDetection(0)')
    sns.kdeplot(train_small.loc[train['HasDetections'] == 1, col], ax=ax[0], label='HasDetection(1)')

    train_small.loc[train['HasDetections'] == 0, col].hist(ax=ax[1], bins=100)
    train_small.loc[train['HasDetections'] == 1, col].hist(ax=ax[1], bins=100)

    plt.suptitle(col, fontsize=30)
    ax[0].set_yscale('log')
    ax[0].set_title('KDE plot')
    
    ax[1].set_title('Histogram')
    ax[1].legend(['NoDetection(0)', 'HasDetection(1)'])
    ax[1].set_yscale('log')
    plt.show()
col = numeric_features[0]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)
col = numeric_features[1]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)
col = numeric_features[2]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)
col = numeric_features[3]

plot_kde_hist_for_numeric(col)
# plot_category_percent_of_target_for_numeric(col)
col = numeric_features[4]

plot_kde_hist_for_numeric(col)
# plot_category_percent_of_target_for_numeric(col)
col = numeric_features[5]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)
col = numeric_features[6]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)
col = numeric_features[7]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)
col = numeric_features[8]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)
col = numeric_features[9]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)
col = numeric_features[10]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)
col = numeric_features[11]

plot_kde_hist_for_numeric(col)
# plot_category_percent_of_target_for_numeric(col)
col = numeric_features[12]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)
col = numeric_features[13]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)
col = numeric_features[14]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)
col = numeric_features[15]

# plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)
col = numeric_features[16]

# plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)
col = numeric_features[17]

# plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)
col = numeric_features[18]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)
col = numeric_features[19]

# plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)
col = numeric_features[20]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)
col = numeric_features[21]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)
col = numeric_features[22]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)
col = numeric_features[23]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)
col = numeric_features[24]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)
col = numeric_features[25]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)
col = numeric_features[26]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)
col = numeric_features[27]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)
col = numeric_features[28]

# plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)
col = numeric_features[29]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)
col = numeric_features[30]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)
col = numeric_features[31]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)
col = numeric_features[32]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)
col = numeric_features[34]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)
col = numeric_features[36]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)
col = numeric_features[37]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)
col = numeric_features[38]

# plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)
col = numeric_features[39]

# plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)
col = numeric_features[40]

# plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)
col = numeric_features[41]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)
col = numeric_features[42]

plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)
col = numeric_features[43]

# plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)
col = numeric_features[44]

# plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)
col = numeric_features[45]

# plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)
col = numeric_features[46]

# plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)
col = numeric_features[47]

# plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)
col = numeric_features[48]

# plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)
col = numeric_features[49]

# plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)
col = numeric_features[50]

# plot_kde_hist_for_numeric(col)
plot_category_percent_of_target_for_numeric(col)
corr = train_small.corr()['HasDetections']
abs(corr).sort_values(ascending=False)
def corr_heatmap(cols):
    correlations = train_small[cols+['HasDetections']].corr()
    # Create color map ranging between two colors
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(correlations, cmap=cmap, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .75})
    plt.show()
corr_heatmap(numeric_features[:10])
corr_heatmap(numeric_features[10:20])
corr_heatmap(numeric_features[20:30])
corr_heatmap(numeric_features[30:40])
corr_heatmap(numeric_features[40:])
corr = train_small.corr()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
upper.head()
threshold = 0.3
for i, df in (upper.iterrows()):
    for ele in df[df.abs() > threshold].items():
        if ele[0] == i:
            break
        else:
            plt.figure(figsize=(7, 7))
            sns.lmplot(x=i, y=ele[0], data=train_small[:100000], hue='HasDetections', palette='Set1', scatter_kws={'alpha':0.3})
            plt.show()
            print('{:50}, {:50} : {}'.format(i, ele[0], ele[1]))


