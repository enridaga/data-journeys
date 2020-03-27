
import os

import numpy as np 

import pandas as pd 

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
sub_path = "../input/statoil-iceberg-submissions"

all_files = os.listdir(sub_path)



# Read and concatenate submissions

outs = [pd.read_csv(os.path.join(sub_path, f), index_col=0) for f in all_files]

concat_sub = pd.concat(outs, axis=1)

cols = list(map(lambda x: "is_iceberg_" + str(x), range(len(concat_sub.columns))))

concat_sub.columns = cols

concat_sub.reset_index(inplace=True)

concat_sub.head()

# check correlation

concat_sub.corr()
# get the data fields ready for stacking

concat_sub['is_iceberg_max'] = concat_sub.iloc[:, 1:6].max(axis=1)

concat_sub['is_iceberg_min'] = concat_sub.iloc[:, 1:6].min(axis=1)

concat_sub['is_iceberg_mean'] = concat_sub.iloc[:, 1:6].mean(axis=1)

concat_sub['is_iceberg_median'] = concat_sub.iloc[:, 1:6].median(axis=1)
# set up cutoff threshold for lower and upper bounds, easy to twist 

cutoff_lo = 0.8

cutoff_hi = 0.2
concat_sub['is_iceberg'] = concat_sub['is_iceberg_mean']

concat_sub[['id', 'is_iceberg']].to_csv('stack_mean.csv', 

                                        index=False, float_format='%.6f')
concat_sub['is_iceberg'] = concat_sub['is_iceberg_median']

concat_sub[['id', 'is_iceberg']].to_csv('stack_median.csv', 

                                        index=False, float_format='%.6f')
concat_sub['is_iceberg'] = np.where(np.all(concat_sub.iloc[:,1:6] > cutoff_lo, axis=1), 1, 

                                    np.where(np.all(concat_sub.iloc[:,1:6] < cutoff_hi, axis=1),

                                             0, concat_sub['is_iceberg_median']))

concat_sub[['id', 'is_iceberg']].to_csv('stack_pushout_median.csv', 

                                        index=False, float_format='%.6f')
concat_sub['is_iceberg'] = np.where(np.all(concat_sub.iloc[:,1:6] > cutoff_lo, axis=1), 

                                    concat_sub['is_iceberg_max'], 

                                    np.where(np.all(concat_sub.iloc[:,1:6] < cutoff_hi, axis=1),

                                             concat_sub['is_iceberg_min'], 

                                             concat_sub['is_iceberg_mean']))

concat_sub[['id', 'is_iceberg']].to_csv('stack_minmax_mean.csv', 

                                        index=False, float_format='%.6f')
concat_sub['is_iceberg'] = np.where(np.all(concat_sub.iloc[:,1:6] > cutoff_lo, axis=1), 

                                    concat_sub['is_iceberg_max'], 

                                    np.where(np.all(concat_sub.iloc[:,1:6] < cutoff_hi, axis=1),

                                             concat_sub['is_iceberg_min'], 

                                             concat_sub['is_iceberg_median']))

concat_sub[['id', 'is_iceberg']].to_csv('stack_minmax_median.csv', 

                                        index=False, float_format='%.6f')
# load the model with best base performance

sub_base = pd.read_csv('../input/statoil-iceberg-submissions/sub_200_ens_densenet.csv')
concat_sub['is_iceberg_base'] = sub_base['is_iceberg']

concat_sub['is_iceberg'] = np.where(np.all(concat_sub.iloc[:,1:6] > cutoff_lo, axis=1), 

                                    concat_sub['is_iceberg_max'], 

                                    np.where(np.all(concat_sub.iloc[:,1:6] < cutoff_hi, axis=1),

                                             concat_sub['is_iceberg_min'], 

                                             concat_sub['is_iceberg_base']))

concat_sub[['id', 'is_iceberg']].to_csv('stack_minmax_bestbase.csv', 

                                        index=False, float_format='%.6f')