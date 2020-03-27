
# General imports
import pandas as pd
import os, sys, gc, warnings

warnings.filterwarnings('ignore')
########################### DATA LOAD/MIX/EXPORT
#################################################################################
# Simple lgbm (0.0948)
sub_1 = pd.read_csv('../input/ieee-simple-lgbm/submission.csv')

# Blend of two kernels with old features (0.9468)
sub_2 = pd.read_csv('../input/ieee-cv-options/submission.csv')

# Add new features lgbm with CV (0.09485)
sub_3 = pd.read_csv('../input/ieee-lgbm-with-groupkfold-cv/submission.csv')

# Add catboost (0.09407)
sub_4 = pd.read_csv('../input/ieee-catboost-baseline-with-groupkfold-cv/submission.csv')

sub_1['isFraud'] += sub_2['isFraud']
sub_1['isFraud'] += sub_3['isFraud']
sub_1['isFraud'] += sub_4['isFraud']

sub_1.to_csv('submission.csv', index=False)