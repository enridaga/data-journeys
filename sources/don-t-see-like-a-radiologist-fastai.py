
!pip install torch torchvision feather-format kornia pyarrow --upgrade   > /dev/null
!pip install git+https://github.com/fastai/fastai_dev                    > /dev/null

from fastai2.basics           import *
from fastai2.medical.imaging  import *

np.set_printoptions(linewidth=120)
path = Path('../input/rsna-intracranial-hemorrhage-detection/')
path_trn = path/'stage_1_train_images'
path_tst = path/'stage_1_test_images'
fname = path_trn/'ID_9d9cc6b01.dcm'
dcm = fname.dcmread()
dcm.show(scale=False)
dcm.show(scale=dicom_windows.brain)
px = dcm.scaled_px.flatten()
plt.hist(px, bins=40);
bins = px.freqhist_bins(20)
print(bins)
plt.hist(px, bins=bins);
plt.plot(bins, torch.linspace(0,1,len(bins)));
plt.imshow(dcm.hist_scaled(), cmap=plt.cm.bone);
dcm.show()
path_inp = Path('../input')
path_df = path_inp/'creating-a-metadata-dataframe-fastai'
df_lbls = pd.read_feather(path_df/'labels.fth')
df_tst = pd.read_feather(path_df/'df_tst.fth')
df_trn = pd.read_feather(path_df/'df_trn.fth')

comb = df_trn.join(df_lbls.set_index('ID'), 'SOPInstanceUID')
repr_flds = ['BitsStored','PixelRepresentation']
df1 = comb.query('(BitsStored==12) & (PixelRepresentation==0)')
df2 = comb.query('(BitsStored==12) & (PixelRepresentation==1)')
df3 = comb.query('BitsStored==16')
dfs = L(df1,df2,df3)
htypes = 'any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural'

def get_samples(df):
    recs = [df.query(f'{c}==1').sample() for c in htypes]
    recs.append(df.query('any==0').sample())
    return pd.concat(recs).fname.values

sample_fns = concat(*dfs.map(get_samples))
sample_dcms = L(Path(o).dcmread() for o in sample_fns)
samples = torch.stack(tuple(sample_dcms.attrgot('scaled_px')))
samples.shape
bins = samples.freqhist_bins()
plt.plot(bins, torch.linspace(0,1,len(bins)));
dcm.show(bins)
dcm.hist_scaled(bins)
scaled_samples = torch.stack(tuple(o.hist_scaled(bins) for o in sample_dcms))
scaled_samples.mean(),scaled_samples.std()
dcm.show(cmap=plt.cm.gist_ncar, figsize=(6,6))