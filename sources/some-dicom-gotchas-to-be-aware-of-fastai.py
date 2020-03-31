
### pip install torch torchvision feather-format kornia pyarrow --upgrade   > /dev/null
### pip install git+https://github.com/fastai/fastai_dev             > /dev/null

from fastai2.basics           import *
from fastai2.medical.imaging  import *

np.set_printoptions(linewidth=120)
path_inp = Path('../input')
path = path_inp/'rsna-intracranial-hemorrhage-detection'
path_trn = path/'stage_1_train_images'
path_tst = path/'stage_1_test_images'
path_df = path_inp/'creating-a-metadata-dataframe'

df_lbls = pd.read_feather(path_df/'labels.fth')
df_tst = pd.read_feather(path_df/'df_tst.fth')
df_trn = pd.read_feather(path_df/'df_trn.fth')
comb = df_trn.join(df_lbls.set_index('ID'), 'SOPInstanceUID')
assert not len(comb[comb['any'].isna()])
comb.head().T
repr_flds = ['BitsStored','PixelRepresentation']
comb.pivot_table(values=['img_mean','img_max','img_min','PatientID','any'], index=repr_flds,
                   aggfunc={'img_mean':'mean','img_max':'max','img_min':'min','PatientID':'count','any':'mean'})
comb.pivot_table(values=['WindowCenter','WindowWidth', 'RescaleIntercept', 'RescaleSlope'], index=repr_flds,
                   aggfunc={'mean','max','min','std','median'})
df1 = comb.query('(BitsStored==12) & (PixelRepresentation==0)')
df2 = comb.query('(BitsStored==12) & (PixelRepresentation==1)')
df3 = comb.query('BitsStored==16')
dfs = [df1,df2,df3]
def distrib_summ(t):
    plt.hist(t,40)
    return array([t.min(),*np.percentile(t,[0.1,1,5,50,95,99,99.9]),t.max()], dtype=np.int)
distrib_summ(df3.img_max.values)
distrib_summ(df3.img_min.values)
distrib_summ(df_tst.img_max.values)
dcms = path_trn.ls(10).map(dcmread)
dcm = dcms[0]
dcm
dcms_px = dcms.attrgot('pixel_array')
list(zip(dcms_px.attrgot('dtype'),
         dcms.attrgot('PixelRepresentation'),
         dcms.attrgot('BitsStored')))
dcm.scaled_px.type()
dcm.show(figsize=(6,6))
scales = False, True, dicom_windows.brain, dicom_windows.subdural
titles = 'raw','normalized','brain windowed','subdural windowed'
for s,a,t in zip(scales, subplots(2,2,imsize=5)[1].flat, titles):
    dcm.show(scale=s, ax=a, title=t)