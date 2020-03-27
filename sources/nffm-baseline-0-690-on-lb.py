
#Download ctrNet-tool 
#You can find the code in https://github.com/guoday/ctrNet-tool
!git clone https://github.com/guoday/ctrNet-tool.git
!cp -r ctrNet-tool/* ./
!rm -r ctrNet-tool data .git
!ls -all
import ctrNet
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from src import misc_utils as utils
import os
import gc
random.seed(2019)
np.random.seed(2019)
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
print('Loading Train and Test Data.\n')
train = pd.read_csv('../input/train.csv', dtype=dtypes, low_memory=True)
train['MachineIdentifier'] = train.index.astype('uint32')
test  = pd.read_csv('../input/test.csv',  dtype=dtypes, low_memory=True)
test['MachineIdentifier']  = test.index.astype('uint32')
test['HasDetections']=[0]*len(test)
def make_bucket(data,num=10):
    data.sort()
    bins=[]
    for i in range(num):
        bins.append(data[int(len(data)*(i+1)//num)-1])
    return bins
float_features=['Census_SystemVolumeTotalCapacity','Census_PrimaryDiskTotalCapacity']
for f in float_features:
    train[f]=train[f].fillna(1e10)
    test[f]=test[f].fillna(1e10)
    data=list(train[f])+list(test[f])
    bins=make_bucket(data,num=50)
    train[f]=np.digitize(train[f],bins=bins)
    test[f]=np.digitize(test[f],bins=bins)
    
features=train.columns.tolist()[1:-1]
hparam=tf.contrib.training.HParams(
            model='nffm',
            norm=True,
            batch_norm_decay=0.9,
            hidden_size=[128,128],
            k=8,
            hash_ids=int(2e5),
            batch_size=1024,
            optimizer="adam",
            learning_rate=0.001,
            num_display_steps=1000,
            num_eval_steps=1000,
            epoch=1,
            metric='auc',
            init_method='uniform',
            init_value=0.1,
            feature_nums=len(features),
            kfold=5)
utils.print_hparams(hparam)
index=set(range(train.shape[0]))
K_fold=[]
for i in range(hparam.kfold):
    if i == hparam.kfold-1:
        tmp=index
    else:
        tmp=random.sample(index,int(1.0/hparam.kfold*train.shape[0]))
    index=index-set(tmp)
    print("Number:",len(tmp))
    K_fold.append(tmp)
    

for i in range(hparam.kfold):
    print("Fold",i)
    dev_index=K_fold[i]
    dev_index=random.sample(dev_index,int(0.1*len(dev_index)))
    train_index=[]
    for j in range(hparam.kfold):
        if j!=i:
            train_index+=K_fold[j]
    model=ctrNet.build_model(hparam)
    model.train(train_data=(train.iloc[train_index][features],train.iloc[train_index]['HasDetections']),\
                dev_data=(train.iloc[dev_index][features],train.iloc[dev_index]['HasDetections']))
    print("Training Done! Inference...")
    if i==0:
        preds=model.infer(dev_data=(test[features],test['HasDetections']))/hparam.kfold
    else:
        preds+=model.infer(dev_data=(test[features],test['HasDetections']))/hparam.kfold
submission = pd.read_csv('../input/sample_submission.csv')
submission['HasDetections'] = preds
print(submission['HasDetections'].head())
submission.to_csv('nffm_submission.csv', index=False)