
!pip install torch torchvision feather-format kornia pyarrow --upgrade   > /dev/null
!pip install git+https://github.com/fastai/fastai_dev                    > /dev/null
from fastai2.basics           import *
from fastai2.vision.all       import *
from fastai2.medical.imaging  import *

np.set_printoptions(linewidth=120)
matplotlib.rcParams['image.cmap'] = 'bone'
set_seed(42)
set_num_threads(1)

path = Path('../input/rsna-intracranial-hemorrhage-detection/')
path_trn = path/'stage_1_train_images'
path_tst = path/'stage_1_test_images'
path_dest = Path()
path_dest.mkdir(exist_ok=True)

path_inp = Path('../input')
path_df = path_inp/'creating-a-metadata-dataframe'
df_lbls = pd.read_feather(path_df/'labels.fth')
df_tst = pd.read_feather(path_df/'df_tst.fth')
df_trn = pd.read_feather(path_df/'df_trn.fth').dropna(subset=['img_pct_window'])
comb = df_trn.join(df_lbls.set_index('ID'), 'SOPInstanceUID')
repr_flds = ['BitsStored','PixelRepresentation']
df1 = comb.query('(BitsStored==12) & (PixelRepresentation==0)')
df2 = comb.query('(BitsStored==12) & (PixelRepresentation==1)')
df3 = comb.query('BitsStored==16')
dfs = L(df1,df2,df3)
def df2dcm(df): return L(Path(o).dcmread() for o in df.fname.values)
df_iffy = df1[df1.RescaleIntercept>-100]
dcms = df2dcm(df_iffy)

_,axs = subplots(2,4, imsize=3)
for i,ax in enumerate(axs.flat): dcms[i].show(ax=ax)
dcm = dcms[2]
d = dcm.pixel_array
plt.hist(d.flatten());
d1 = df2dcm(df1.iloc[[0]])[0].pixel_array
plt.hist(d1.flatten());
scipy.stats.mode(d.flatten()).mode[0]
d += 1000

px_mode = scipy.stats.mode(d.flatten()).mode[0]
d[d>=px_mode] = d[d>=px_mode] - px_mode
dcm.PixelData = d.tobytes()
dcm.RescaleIntercept = -1000
plt.hist(dcm.pixel_array.flatten());
_,axs = subplots(1,2)
dcm.show(ax=axs[0]);   dcm.show(dicom_windows.brain, ax=axs[1])
def fix_pxrepr(dcm):
    if dcm.PixelRepresentation != 0 or dcm.RescaleIntercept<-100: return
    x = dcm.pixel_array + 1000
    px_mode = 4096
    x[x>=px_mode] = x[x>=px_mode] - px_mode
    dcm.PixelData = x.tobytes()
    dcm.RescaleIntercept = -1000
dcms = df2dcm(df_iffy)
dcms.map(fix_pxrepr)

_,axs = subplots(2,4, imsize=3)
for i,ax in enumerate(axs.flat): dcms[i].show(ax=ax)
df_iffy.img_pct_window[:10].values
plt.hist(comb.img_pct_window,40);
comb = comb.assign(pct_cut = pd.cut(comb.img_pct_window, [0,0.02,0.05,0.1,0.2,0.3,1]))
comb.pivot_table(values='any', index='pct_cut', aggfunc=['sum','count']).T
comb.drop(comb.query('img_pct_window<0.02').index, inplace=True)
df_lbl = comb.query('any==True')
n_lbl = len(df_lbl)
n_lbl
df_nonlbl = comb.query('any==False').sample(n_lbl//2)
len(df_nonlbl)
comb = pd.concat([df_lbl,df_nonlbl])
len(comb)
dcm = Path(dcms[3].filename).dcmread()
fix_pxrepr(dcm)
px = dcm.windowed(*dicom_windows.brain)
show_image(px);
blurred = gauss_blur2d(px, 100)
show_image(blurred);
show_image(blurred>0.3);
_,axs = subplots(1,4, imsize=3)
for i,ax in enumerate(axs.flat):
    dcms[i].show(dicom_windows.brain, ax=ax)
    show_image(dcms[i].mask_from_blur(dicom_windows.brain), cmap=plt.cm.Reds, alpha=0.6, ax=ax)
def pad_square(x):
    r,c = x.shape
    d = (c-r)/2
    pl,pr,pt,pb = 0,0,0,0
    if d>0: pt,pd = int(math.floor( d)),int(math.ceil( d))        
    else:   pl,pr = int(math.floor(-d)),int(math.ceil(-d))
    return np.pad(x, ((pt,pb),(pl,pr)), 'minimum')

def crop_mask(x):
    mask = x.mask_from_blur(dicom_windows.brain)
    bb = mask2bbox(mask)
    if bb is None: return
    lo,hi = bb
    cropped = x.pixel_array[lo[0]:hi[0],lo[1]:hi[1]]
    x.pixel_array = pad_square(cropped)
_,axs = subplots(1,2)
dcm.show(ax=axs[0])
crop_mask(dcm)
dcm.show(ax=axs[1]);
htypes = 'any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural'

def get_samples(df):
    recs = [df.query(f'{c}==1').sample() for c in htypes]
    recs.append(df.query('any==0').sample())
    return pd.concat(recs).fname.values

sample_fns = concat(*dfs.map(get_samples))
sample_dcms = tuple(Path(o).dcmread().scaled_px for o in sample_fns)
samples = torch.stack(sample_dcms)
bins = samples.freqhist_bins()
(path_dest/'bins.pkl').save(bins)
def dcm_tfm(fn): 
    fn = Path(fn)
    try:
        x = fn.dcmread()
        fix_pxrepr(x)
    except Exception as e:
        print(fn,e)
        raise SkipItemException
    if x.Rows != 512 or x.Columns != 512: x.zoom_to((512,512))
    return x.scaled_px
fns = list(comb.fname.values)
dest = path_dest/'train_jpg'
dest.mkdir(exist_ok=True)
# NB: Use bs=512 or 1024 when running on GPU
bs=4

dsrc = DataSource(fns, [[dcm_tfm],[os.path.basename]])
dl = TfmdDL(dsrc, bs=bs, num_workers=2)
def dest_fname(fname): return dest/Path(fname).with_suffix('.jpg')

def save_cropped_jpg(o, dest):
    fname,px = o
    px.save_jpg(dest_fname(fname), dicom_windows.brain, dicom_windows.subdural, bins=bins)
def process_batch(pxs, fnames, n_workers=4):
    pxs = to_device(pxs)
    masks = pxs.mask_from_blur(dicom_windows.brain)
    bbs = mask2bbox(masks)
    gs = crop_resize(pxs, bbs, 256).cpu().squeeze()
    parallel(save_cropped_jpg, zip(fnames, gs), n_workers=n_workers, progress=False, dest=dest)
# test and time a single batch. It's ~100x faster on a GPU!
%time process_batch(*dl.one_batch(), n_workers=3)
fn = dest.ls()[0]
im = Image.open(fn)
fn
show_images(tensor(im).permute(2,0,1), titles=['brain','subdural','normalized'])
# dest.mkdir(exist_ok=True)
# for b in progress_bar(dl): process_batch(*b, n_workers=8)
# Uncomment this to view some processed images

# for i,(ax,fn) in enumerate(zip(subplots(2,4)[1].flat,fns)):
#     jpgfn = dest/Path(fn).with_suffix('.jpg').name
#     a = jpgfn.jpg16read()
#     show_image(a,ax=ax)