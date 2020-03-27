
# You can edit the font size here to make rendered text more comfortable to read
# It was built on a 13" retina screen with 18px
from IPython.core.display import display, HTML
display(HTML("<style>.rendered_html { font-size: 18px; }</style>"))

# we'll also use this package to read tables
# it's generally useful for astrophysics work, including this challenge
# so we'd suggest installing it, even if you elect to work with pandas
from astropy.table import Table

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
#%matplotlib inline
%config InlineBackend.figure_format = 'retina'

def displayimage(image):
    img = np.array(Image.open(image))
    plt.figure(figsize=(10,10))
    plt.axis('off')
    _ = plt.imshow(img)
displayimage('../input/plasticc-astronomy-starter-kit-media/LSST_night.jpg')

displayimage('../input/plasticc-astronomy-starter-kit-media/LSST_FoV.jpg')
displayimage('../input/plasticc-astronomy-starter-kit-media/HST_field_of_view.jpg')
%%HTML
<div align="middle">
<video width="60%" controls>
      <source src="https://storage.googleapis.com/kaggle-datasets/52432/98042/SN98bu_LC.mp4?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1538266058&Signature=SgHGTPeH4qfe%2BvxF08OY0mBaQX5GxbSOUe2X2ibg%2FxWhGnJNaQ7JT57lFaAOpjN9HAEC%2Fy1o4OCzD3AdR%2F%2FLRXeSfY6BMZdsXclppAzS2Uf%2BXyhIYFXUhGe%2F1gc5Gch5myz%2FOre1h%2F5Fgh7AhK6dKAVMHhLYREWTWEyV8MJkhXtsy3a%2B%2BjwerdcdOiGtuQe3JzjHudBsw%2FjOCtLxYsA4SMNtI6F6750cgqAvC7OgNX6vt4tCpCQW6LxyZSn64sFD0ulZJF9IWWX9tqyMdVgcdb2CEYRke91kpXivdGAvjQrZDr1OkJXSYq6QcXY3JMGrx9egN0iMe3pIJ35Z8DPEsg%3D%3D" type="video/mp4">
</video>
</div>
displayimage('../input/plasticc-astronomy-starter-kit-media/variability_tree.jpg')
displayimage('../input/plasticc-astronomy-starter-kit-media/prism.jpg')
displayimage('../input/plasticc-astronomy-starter-kit-media/SN_Spectra.jpg')
displayimage('../input/plasticc-astronomy-starter-kit-media/smartt_supernovae_diversity.jpg')
def displayimageconverted(image):
    img = np.array(Image.open(image).convert("RGBA"))
    plt.figure(figsize=(10,10))
    plt.axis('off')
    _ = plt.imshow(img)
displayimageconverted('../input/plasticc-astronomy-starter-kit-media/atlas_of_variable_stars.png')
import io
import base64

video = io.open('../input/plasticc-astronomy-starter-kit-media/LSST_filter_change.mp4', 'rb').read()
encoded = base64.b64encode(video)
HTML(data='''
<video width = 90% controls>
   <source src="data:video/mp4;base64,{0}" type="video/mp4" />
</video>'''.format(encoded.decode('ascii')))


displayimage("../input/plasticc-astronomy-starter-kit-media/LSST_passbands.jpg")
displayimage("../input/plasticc-astronomy-starter-kit-media/LSST_construction.jpg")
obj1 = Table.read('../input/plasticc-astronomy-starter-kit-media/fake010.csv', format='csv')
obj1
displayimageconverted("../input/plasticc-astronomy-starter-kit-media/2007-X-025_R.PS.png")
displayimageconverted("../input/plasticc-astronomy-starter-kit-media/2007-X-025_I.PS.png")
import os
import numpy as np
import scipy.stats as spstat
import matplotlib.pyplot as plt
from collections import OrderedDict

#%matplotlib inline
class LightCurve(object):
    '''Light curve object for PLAsTiCC formatted data'''
    
    _passbands = OrderedDict([(0,'C4'),\
                              (1,'C2'),\
                              (2,'C3'),\
                              (3,'C1'),\
                              (4,'k'),\
                              (5,'C5')])
    
    _pbnames = ['u','g','r','i','z','y']
    
    def __init__(self, filename):
        '''Read in light curve data'''

        self.DFlc     = Table.read(filename, format='ascii.csv')
        self.filename = filename.replace('.csv','')
        self._finalize()
     
    # this is some simple code to demonstrate how to calculate features on these multiband light curves
    # we're not suggesting using these features specifically
    # there also might be additional pre-processing you do before computing anything
    # it's purely for illustration
    def _finalize(self):
        '''Store individual passband fluxes as object attributes'''
        # in this example, we'll use the weighted mean to normalize the features
        weighted_mean = lambda flux, dflux: np.sum(flux*(flux/dflux)**2)/np.sum((flux/dflux)**2)
        
        # define some functions to compute simple descriptive statistics
        normalized_flux_std = lambda flux, wMeanFlux: np.std(flux/wMeanFlux, ddof = 1)
        normalized_amplitude = lambda flux, wMeanFlux: (np.max(flux) - np.min(flux))/wMeanFlux
        normalized_MAD = lambda flux, wMeanFlux: np.median(np.abs((flux - np.median(flux))/wMeanFlux))
        beyond_1std = lambda flux, wMeanFlux: sum(np.abs(flux - wMeanFlux) > np.std(flux, ddof = 1))/len(flux)
        
        for pb in self._passbands:
            ind = self.DFlc['passband'] == pb
            pbname = self._pbnames[pb]
            
            if len(self.DFlc[ind]) == 0:
                setattr(self, f'{pbname}Std', np.nan)
                setattr(self, f'{pbname}Amp', np.nan)
                setattr(self, f'{pbname}MAD', np.nan)
                setattr(self, f'{pbname}Beyond', np.nan)
                setattr(self, f'{pbname}Skew', np.nan)
                continue
            
            f  = self.DFlc['flux'][ind]
            df = self.DFlc['flux_err'][ind]
            m  = weighted_mean(f, df)
            
            # we'll save the measurements in each passband to simplify access.
            setattr(self, f'{pbname}Flux', f)
            setattr(self, f'{pbname}FluxUnc', df)
            setattr(self, f'{pbname}Mean', m)
            
            # compute the features
            std = normalized_flux_std(f, df)
            amp = normalized_amplitude(f, m)
            mad = normalized_MAD(f, m)
            beyond = beyond_1std(f, m)
            skew = spstat.skew(f) 
            
            # and save the features
            setattr(self, f'{pbname}Std', std)
            setattr(self, f'{pbname}Amp', amp)
            setattr(self, f'{pbname}MAD', mad)
            setattr(self, f'{pbname}Beyond', beyond)
            setattr(self, f'{pbname}Skew', skew)
        
        # we can also construct features between passbands
        pbs = list(self._passbands.keys())
        for i, lpb in enumerate(pbs[0:-1]):
            rpb = pbs[i+1]
            
            lpbname = self._pbnames[lpb]
            rpbname = self._pbnames[rpb]
            
            colname = '{}Minus{}'.format(lpbname, rpbname.upper())
            lMean = getattr(self, f'{lpbname}Mean', np.nan)
            rMean = getattr(self, f'{rpbname}Mean', np.nan)
            col = -2.5*np.log10(lMean/rMean) if lMean> 0 and rMean > 0 else -999
            setattr(self, colname, col)
    
    # this is a simple routine to visualize a light curve
    # it can plot vs the MJD array of the light curve
    # or vs an optional `phase` array that you pass 
    def plot_multicolor_lc(self, phase=None):
        '''Plot the multiband light curve'''

        fig, ax = plt.subplots(figsize=(8,6))

        
        if phase is None:
            phase = []
        if len(phase) != len(self.DFlc):
            phase = self.DFlc['mjd']
            xlabel = 'MJD'
        else:
            xlabel = 'Phase'
            
        for i, pb in enumerate(self._passbands):
            pbname = self._pbnames[pb]
            ind = self.DFlc['passband'] == pb
            if len(self.DFlc[ind]) == 0:
                continue
            ax.errorbar(phase[ind], 
                     self.DFlc['flux'][ind],
                     self.DFlc['flux_err'][ind],
                     fmt = 'o', color = self._passbands[pb], label = f'{pbname}')
        ax.legend(ncol = 4, frameon = True)
        ax.set_xlabel(f'{xlabel}', fontsize='large')
        ax.set_ylabel('Flux', fontsize='large')
        fig.suptitle(self.filename, fontsize='x-large')
        fig.tight_layout(rect=[0, 0, 1, 0.97])
    
    def get_features(self):
        '''Return all the features for this object'''
        variables = ['Std', 'Amp', 'MAD', 'Beyond', 'Skew']
        feats = []
        for i, pb in enumerate(self._passbands):
            pbname = self._pbnames[pb]
            feats += [getattr(self, f'{pbname}{x}', np.nan) for x in variables]
        return feats
lc = LightCurve('../input/plasticc-astronomy-starter-kit-media/fake010.csv')
lc.plot_multicolor_lc()
lc = LightCurve('../input/plasticc-astronomy-starter-kit-media/fake030.csv')
lc.plot_multicolor_lc()
from gatspy.periodic import LombScargleMultiband
model = LombScargleMultiband(fit_period=True)

# we'll window the search range by setting minimums and maximums here
# but in general, the search range you want to evaluate will depend on the data
# and you will not be able to window like this unless you know something about
# the class of the object a priori
t_min = max(np.median(np.diff(sorted(lc.DFlc['mjd']))), 0.1)
t_max = min(10., (lc.DFlc['mjd'].max() - lc.DFlc['mjd'].min())/2.)

model.optimizer.set(period_range=(t_min, t_max), first_pass_coverage=5)
model.fit(lc.DFlc['mjd'], lc.DFlc['flux'], dy=lc.DFlc['flux_err'], filts=lc.DFlc['passband'])
period = model.best_period
print(f'{lc.filename} has a period of {period} days')
phase = (lc.DFlc['mjd'] /period) % 1
lc.plot_multicolor_lc(phase=phase)
header = Table.read('../input/plasticc-astronomy-starter-kit-media/plasticc_training_set_metadata_stub.csv', format='csv')
header
displayimage("../input/plasticc-astronomy-starter-kit-media/allsky_equatorial.jpg")
displayimage("../input/plasticc-astronomy-starter-kit-media/skymap_minion1016.jpg")
displayimage("../input/plasticc-astronomy-starter-kit-media/allsky_galactic.jpg")
displayimage("../input/plasticc-astronomy-starter-kit-media/redshift.jpg")
displayimage("../input/plasticc-astronomy-starter-kit-media/Gaia_milky_way.jpg")
displayimage("../input/plasticc-astronomy-starter-kit-media/large_scale_structure.jpg")
displayimage("../input/plasticc-astronomy-starter-kit-media/hubble_cepheid.jpeg")
displayimage("../input/plasticc-astronomy-starter-kit-media/hubble_law.jpeg")
displayimage("../input/plasticc-astronomy-starter-kit-media/JLA_HubbleDiagram.png")
displayimage("../input/plasticc-astronomy-starter-kit-media/UniverseTimeline.jpg")
displayimage("../input/plasticc-astronomy-starter-kit-media/dust_map.jpeg")
video = io.open('../input/plasticc-astronomy-starter-kit-media/MilkyWayDust.mp4', 'rb').read()
encoded = base64.b64encode(video)
HTML(data='''
<video width = 60% controls>
   <source src="data:video/mp4;base64,{0}" type="video/mp4" />
</video>'''.format(encoded.decode('ascii')))