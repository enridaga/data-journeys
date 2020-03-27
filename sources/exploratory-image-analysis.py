
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import glob, os
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
smjpegs = [f for f in glob.glob("../input/train_sm/*.jpeg")]
print(smjpegs[:9])
set175 = [smj for smj in smjpegs if "set175" in smj]
print(set175)
first = plt.imread('../input/train_sm/set175_1.jpeg')
dims = np.shape(first)
print(dims)
np.min(first), np.max(first)
pixel_matrix = np.reshape(first, (dims[0] * dims[1], dims[2]))
print(np.shape(pixel_matrix))
#plt.scatter(pixel_matrix[:,0], pixel_matrix[:,1])
_ = plt.hist2d(pixel_matrix[:,1], pixel_matrix[:,2], bins=(50,50))
fifth = plt.imread('../input/train_sm/set175_5.jpeg')
dims = np.shape(fifth)
pixel_matrix5 = np.reshape(fifth, (dims[0] * dims[1], dims[2]))
_ = plt.hist2d(pixel_matrix5[:,1], pixel_matrix5[:,2], bins=(50,50))
_ = plt.hist2d(pixel_matrix[:,2], pixel_matrix5[:,2], bins=(50,50))
plt.imshow(first)
plt.imshow(fifth)
plt.imshow(first[:,:,2] - fifth[:,:,1])
second = plt.imread('../input/train_sm/set175_2.jpeg')
plt.imshow(first[:,:,2] - second[:,:,2])
plt.imshow(second)
# simple k means clustering
from sklearn import cluster

kmeans = cluster.KMeans(5)
clustered = kmeans.fit_predict(pixel_matrix)

dims = np.shape(first)
clustered_img = np.reshape(clustered, (dims[0], dims[1]))
plt.imshow(clustered_img)
plt.imshow(first)
ind0, ind1, ind2, ind3 = [np.where(clustered == x)[0] for x in [0, 1, 2, 3]]
# quick look at color value histograms for pixel matrix from first image
import seaborn as sns
sns.distplot(pixel_matrix[:,0], bins=12)
sns.distplot(pixel_matrix[:,1], bins=12)
sns.distplot(pixel_matrix[:,2], bins=12)
# even subsampling is throwing memory error for me, :p
#length = np.shape(pixel_matrix)[0]
#rand_ind = np.random.choice(length, size=50000)
#sns.pairplot(pixel_matrix[rand_ind,:])
set79 = [smj for smj in smjpegs if "set79" in smj]
print(set79)
img79_1, img79_2, img79_3, img79_4, img79_5 = \
  [plt.imread("../input/train_sm/set79_" + str(n) + ".jpeg") for n in range(1, 6)]
img_list = (img79_1, img79_2, img79_3, img79_4, img79_5)

plt.figure(figsize=(8,10))
plt.imshow(img_list[0])
plt.show()
class MSImage():
    """Lightweight wrapper for handling image to matrix transforms. No setters,
    main point of class is to remember image dimensions despite transforms."""
    
    def __init__(self, img):
        """Assume color channel interleave that holds true for this set."""
        self.img = img
        self.dims = np.shape(img)
        self.mat = np.reshape(img, (self.dims[0] * self.dims[1], self.dims[2]))

    @property
    def matrix(self):
        return self.mat
        
    @property
    def image(self):
        return self.img
    
    def to_flat_img(self, derived):
        """"Use dims property to reshape a derived matrix back into image form when
        derived image would only have one band."""
        return np.reshape(derived, (self.dims[0], self.dims[1]))
    
    def to_matched_img(self, derived):
        """"Use dims property to reshape a derived matrix back into image form."""
        return np.reshape(derived, (self.dims[0], self.dims[1], self.dims[2]))
msi79_1 = MSImage(img79_1)
print(np.shape(msi79_1.matrix))
print(np.shape(msi79_1.img))
def bnormalize(mat):
    """much faster brightness normalization, since it's all vectorized"""
    bnorm = np.zeros_like(mat, dtype=np.float32)
    maxes = np.max(mat, axis=1)
    bnorm = mat / np.vstack((maxes, maxes, maxes)).T
    return bnorm
bnorm = bnormalize(msi79_1.matrix)
bnorm_img = msi79_1.to_matched_img(bnorm)
plt.figure(figsize=(8,10))
plt.imshow(bnorm_img)
plt.show()
msi79_2 = MSImage(img79_2)
bnorm79_2 = bnormalize(msi79_2.matrix)
bnorm79_2_img = msi79_2.to_matched_img(bnorm79_2)
plt.figure(figsize=(8,10))
plt.imshow(bnorm79_2_img)
plt.show()
msinorm79_1 = MSImage(bnorm_img)
msinorm79_2 = MSImage(bnorm79_2_img)

_ = plt.hist2d(msinorm79_1.matrix[:,2], msinorm79_2.matrix[:,2], bins=(50,50))
_ = plt.hist2d(msinorm79_1.matrix[:,1], msinorm79_2.matrix[:,1], bins=(50,50))
_ = plt.hist2d(msinorm79_1.matrix[:,0], msinorm79_2.matrix[:,0], bins=(50,50))
import seaborn as sns
sns.distplot(msinorm79_1.matrix[:,0], bins=12)
sns.distplot(msinorm79_1.matrix[:,1], bins=12)
sns.distplot(msinorm79_1.matrix[:,2], bins=12)
plt.figure(figsize=(8,10))
plt.imshow(img79_1)
plt.show()
np.max(img79_1[:,:,0])
plt.figure(figsize=(10,15))
plt.subplot(121)
plt.imshow(img79_1[:,:,0] > 230)
plt.subplot(122)
plt.imshow(img79_1)
plt.show()
plt.figure(figsize=(10,15))
plt.subplot(121)
plt.imshow(img79_2[:,:,0] > 230)
plt.subplot(122)
plt.imshow(img79_2)
plt.show()
print(np.min(bnorm79_2_img[:,:,0]))
print(np.max(bnorm79_2_img[:,:,0]))
print(np.mean(bnorm79_2_img[:,:,0]))
print(np.std(bnorm79_2_img[:,:,0]))
plt.figure(figsize=(10,15))
plt.subplot(121)
plt.imshow(bnorm79_2_img[:,:,0] > 0.98)
plt.subplot(122)
plt.imshow(img79_2)
plt.show()
plt.figure(figsize=(10,15))
plt.subplot(121)
plt.imshow(bnorm_img[:,:,0] > 0.98)
plt.subplot(122)
plt.imshow(img79_1)
plt.show()
plt.figure(figsize=(10,15))
plt.subplot(121)
plt.imshow((bnorm79_2_img[:,:,0] > 0.9999) & \
           (bnorm79_2_img[:,:,1] < 0.9999) & \
           (bnorm79_2_img[:,:,2] < 0.9999))
plt.subplot(122)
plt.imshow(img79_2)
plt.show()
plt.figure(figsize=(10,15))
plt.subplot(121)
plt.imshow(bnorm_img[:,:,0] > 0.995)
plt.subplot(122)
plt.imshow(img79_1)
plt.show()
plt.figure(figsize=(10,6))
plt.subplot(121)
plt.plot(bnorm_img[2000, 1000, :])
plt.subplot(122)
plt.plot(img79_1[2000, 1000, :])
from scipy import spatial

pixel = msi79_1.matrix[2000 * 1000, :]
np.shape(pixel)
set144 = [MSImage(plt.imread(smj)) for smj in smjpegs if "set144" in smj]
plt.imshow(set144[0].image)
import skimage
from skimage.feature import greycomatrix, greycoprops
from skimage.filters import sobel
# a sobel filter is a basic way to get an edge magnitude/gradient image
fig = plt.figure(figsize=(8, 8))
plt.imshow(sobel(set144[0].image[:750,:750,2]))
from skimage.filters import sobel_h

# can also apply sobel only across one direction.
fig = plt.figure(figsize=(8, 8))
plt.imshow(sobel_h(set144[0].image[:750,:750,2]), cmap='BuGn')
from sklearn.decomposition import PCA

pca = PCA(3)
pca.fit(set144[0].matrix)
set144_0_pca = pca.transform(set144[0].matrix)
set144_0_pca_img = set144[0].to_matched_img(set144_0_pca)
fig = plt.figure(figsize=(8, 8))
plt.imshow(set144_0_pca_img[:,:,0], cmap='BuGn')
fig = plt.figure(figsize=(8, 8))
plt.imshow(set144_0_pca_img[:,:,1], cmap='BuGn')
fig = plt.figure(figsize=(8, 8))
plt.imshow(set144_0_pca_img[:,:,2], cmap='BuGn')
sub = set144[0].image[:150,:150,2]
def glcm_image(img, measure="dissimilarity"):
    """TODO: allow different window sizes by parameterizing 3, 4. Also should
    parameterize direction vector [1] [0]"""
    texture = np.zeros_like(sub)

    # quadratic looping in python w/o vectorized routine, yuck!
    for i in range(img.shape[0] ):  
        for j in range(sub.shape[1] ):  
          
            # don't calculate at edges
            if (i < 3) or \
               (i > (img.shape[0])) or \
               (j < 3) or \
               (j > (img.shape[0] - 4)):          
                continue  
        
            # calculate glcm matrix for 7 x 7 window, use dissimilarity (can swap in
            # contrast, etc.)
            glcm_window = img[i-3: i+4, j-3 : j+4]  
            glcm = greycomatrix(glcm_window, [1], [0],  symmetric = True, normed = True )   
            texture[i,j] = greycoprops(glcm, measure)  
    return texture
dissimilarity = glcm_image(sub, "dissimilarity")
fig = plt.figure(figsize=(8, 8))
plt.subplot(1,2,1)
plt.imshow(dissimilarity, cmap="bone")
plt.subplot(1,2,2)
plt.imshow(sub, cmap="bone")
from skimage import color

hsv = color.rgb2hsv(set144[0].image)
fig = plt.figure(figsize=(8, 8))
plt.subplot(2,2,1)
plt.imshow(set144[0].image, cmap="bone")
plt.subplot(2,2,2)
plt.imshow(hsv[:,:,0], cmap="bone")
plt.subplot(2,2,3)
plt.imshow(hsv[:,:,1], cmap='bone')
plt.subplot(2,2,4)
plt.imshow(hsv[:,:,2], cmap='bone')
fig = plt.figure(figsize=(8, 8))
plt.subplot(2,2,1)
plt.imshow(set144[0].image[:200,:200,:])
plt.subplot(2,2,2)
plt.imshow(hsv[:200,:200,0], cmap="PuBuGn")
plt.subplot(2,2,3)
plt.imshow(hsv[:200,:200,1], cmap='bone')
plt.subplot(2,2,4)
plt.imshow(hsv[:200,:200,2], cmap='bone')
fig = plt.figure(figsize=(8, 6))
plt.imshow(hsv[200:500,200:500,0], cmap='bone')
hsvmsi = MSImage(hsv)
import seaborn as sns
sns.distplot(hsvmsi.matrix[:,0], bins=12)
sns.distplot(hsvmsi.matrix[:,1], bins=12)
sns.distplot(hsvmsi.matrix[:,2], bins=12)
plt.imshow(hsvmsi.image[:,:,2] < 0.4, cmap="plasma")
fig = plt.figure(figsize=(8, 8))
plt.subplot(1,2,1)
plt.imshow(set144[0].image[:250,:250,:])
plt.subplot(1,2,2)
plt.imshow(hsvmsi.image[:250,:250,2] < 0.4, cmap="plasma")
fig = plt.figure(figsize=(8, 8))
img2 = plt.imshow(set144[0].image[:250,:250,:], interpolation='nearest')
img3 = plt.imshow(hsvmsi.image[:250,:250,2] < 0.4, cmap='binary_r', alpha=0.4)
plt.show()
# -*- coding: utf-8 -*-
# imreg.py

# Copyright (c) 2011-2014, Christoph Gohlke
# Copyright (c) 2011-2014, The Regents of the University of California
# Produced at the Laboratory for Fluorescence Dynamics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""FFT based image registration.

Implements an FFT-based technique for translation, rotation and scale-invariant
image registration [1].

:Author:
  `Christoph Gohlke <http://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics, University of California, Irvine

:Version: 2013.01.18

Requirements
------------
* `CPython 2.7 or 3.3 <http://www.python.org>`_
* `Numpy 1.7 <http://www.numpy.org>`_
* `Scipy 0.12 <http://www.scipy.org>`_
* `Matplotlib 1.2 <http://www.matplotlib.org>`_  (optional for plotting)

Notes
-----
The API and algorithms are not stable yet and are expected to change between
revisions.

References
----------
(1) An FFT-based technique for translation, rotation and scale-invariant
    image registration. BS Reddy, BN Chatterji.
    IEEE Transactions on Image Processing, 5, 1266-1271, 1996
(2) An IDL/ENVI implementation of the FFT-based algorithm for automatic
    image registration. H Xiea, N Hicksa, GR Kellera, H Huangb, V Kreinovich.
    Computers & Geosciences, 29, 1045-1055, 2003.
(3) Image Registration Using Adaptive Polar Transform. R Matungka, YF Zheng,
    RL Ewing. IEEE Transactions on Image Processing, 18(10), 2009.

Examples
--------
>>> im0 = imread('t400')
>>> im1 = imread('Tr19s1.3')
>>> im2, scale, angle, (t0, t1) = similarity(im0, im1)
>>> imshow(im0, im1, im2)

>>> im0 = imread('t350380ori')
>>> im1 = imread('t350380shf')
>>> t0, t1 = translation(im0, im1)

"""

from __future__ import division, print_function

import math

import numpy
from numpy.fft import fft2, ifft2, fftshift

try:
    import scipy.ndimage.interpolation as ndii
except ImportError:
    import ndimage.interpolation as ndii

__version__ = '2013.01.18'
__docformat__ = 'restructuredtext en'
__all__ = ['translation', 'similarity']


def translation(im0, im1):
    """Return translation vector to register images."""
    shape = im0.shape
    f0 = fft2(im0)
    f1 = fft2(im1)
    ir = abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))
    t0, t1 = numpy.unravel_index(numpy.argmax(ir), shape)
    if t0 > shape[0] // 2:
        t0 -= shape[0]
    if t1 > shape[1] // 2:
        t1 -= shape[1]
    return [t0, t1]


def similarity(im0, im1):
    """Return similarity transformed image im1 and transformation parameters.

    Transformation parameters are: isotropic scale factor, rotation angle (in
    degrees), and translation vector.

    A similarity transformation is an affine transformation with isotropic
    scale and without shear.

    Limitations:
    Image shapes must be equal and square.
    All image areas must have same scale, rotation, and shift.
    Scale change must be less than 1.8.
    No subpixel precision.

    """
    if im0.shape != im1.shape:
        raise ValueError("Images must have same shapes.")
    elif len(im0.shape) != 2:
        raise ValueError("Images must be 2 dimensional.")

    f0 = fftshift(abs(fft2(im0)))
    f1 = fftshift(abs(fft2(im1)))

    h = highpass(f0.shape)
    f0 *= h
    f1 *= h
    del h

    f0, log_base = logpolar(f0)
    f1, log_base = logpolar(f1)

    f0 = fft2(f0)
    f1 = fft2(f1)
    r0 = abs(f0) * abs(f1)
    ir = abs(ifft2((f0 * f1.conjugate()) / r0))
    i0, i1 = numpy.unravel_index(numpy.argmax(ir), ir.shape)
    angle = 180.0 * i0 / ir.shape[0]
    scale = log_base ** i1

    if scale > 1.8:
        ir = abs(ifft2((f1 * f0.conjugate()) / r0))
        i0, i1 = numpy.unravel_index(numpy.argmax(ir), ir.shape)
        angle = -180.0 * i0 / ir.shape[0]
        scale = 1.0 / (log_base ** i1)
        if scale > 1.8:
            raise ValueError("Images are not compatible. Scale change > 1.8")

    if angle < -90.0:
        angle += 180.0
    elif angle > 90.0:
        angle -= 180.0

    im2 = ndii.zoom(im1, 1.0/scale)
    im2 = ndii.rotate(im2, angle)

    if im2.shape < im0.shape:
        t = numpy.zeros_like(im0)
        t[:im2.shape[0], :im2.shape[1]] = im2
        im2 = t
    elif im2.shape > im0.shape:
        im2 = im2[:im0.shape[0], :im0.shape[1]]

    f0 = fft2(im0)
    f1 = fft2(im2)
    ir = abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))
    t0, t1 = numpy.unravel_index(numpy.argmax(ir), ir.shape)

    if t0 > f0.shape[0] // 2:
        t0 -= f0.shape[0]
    if t1 > f0.shape[1] // 2:
        t1 -= f0.shape[1]

    im2 = ndii.shift(im2, [t0, t1])

    # correct parameters for ndimage's internal processing
    if angle > 0.0:
        d = int((int(im1.shape[1] / scale) * math.sin(math.radians(angle))))
        t0, t1 = t1, d+t0
    elif angle < 0.0:
        d = int((int(im1.shape[0] / scale) * math.sin(math.radians(angle))))
        t0, t1 = d+t1, d+t0
    scale = (im1.shape[1] - 1) / (int(im1.shape[1] / scale) - 1)

    return im2, scale, angle, [-t0, -t1]


def similarity_matrix(scale, angle, vector):
    """Return homogeneous transformation matrix from similarity parameters.

    Transformation parameters are: isotropic scale factor, rotation angle (in
    degrees), and translation vector (of size 2).

    The order of transformations is: scale, rotate, translate.

    """
    S = numpy.diag([scale, scale, 1.0])
    R = numpy.identity(3)
    angle = math.radians(angle)
    R[0, 0] = math.cos(angle)
    R[1, 1] = math.cos(angle)
    R[0, 1] = -math.sin(angle)
    R[1, 0] = math.sin(angle)
    T = numpy.identity(3)
    T[:2, 2] = vector
    return numpy.dot(T, numpy.dot(R, S))


def logpolar(image, angles=None, radii=None):
    """Return log-polar transformed image and log base."""
    shape = image.shape
    center = shape[0] / 2, shape[1] / 2
    if angles is None:
        angles = shape[0]
    if radii is None:
        radii = shape[1]
    theta = numpy.empty((angles, radii), dtype=numpy.float64)
    theta.T[:] = -numpy.linspace(0, numpy.pi, angles, endpoint=False)
    #d = radii
    d = numpy.hypot(shape[0]-center[0], shape[1]-center[1])
    log_base = 10.0 ** (math.log10(d) / (radii))
    radius = numpy.empty_like(theta)
    radius[:] = numpy.power(log_base, numpy.arange(radii,
                                                   dtype=numpy.float64)) - 1.0
    x = radius * numpy.sin(theta) + center[0]
    y = radius * numpy.cos(theta) + center[1]
    output = numpy.empty_like(x)
    ndii.map_coordinates(image, [x, y], output=output)
    return output, log_base


def highpass(shape):
    """Return highpass filter to be multiplied with fourier transform."""
    x = numpy.outer(
        numpy.cos(numpy.linspace(-math.pi/2., math.pi/2., shape[0])),
        numpy.cos(numpy.linspace(-math.pi/2., math.pi/2., shape[1])))
    return (1.0 - x) * (2.0 - x)


def imread(fname, norm=True):
    """Return image data from img&hdr uint8 files."""
    with open(fname+'.hdr', 'r') as fh:
        hdr = fh.readlines()
    img = numpy.fromfile(fname+'.img', numpy.uint8, -1)
    img.shape = int(hdr[4].split()[-1]), int(hdr[3].split()[-1])
    if norm:
        img = img.astype(numpy.float64)
        img /= 255.0
    return img


def imshow(im0, im1, im2, im3=None, cmap=None, **kwargs):
    """Plot images using matplotlib."""
    from matplotlib import pyplot
    if cmap is None:
        cmap = 'coolwarm'
    if im3 is None:
        im3 = abs(im2 - im0)
    pyplot.subplot(221)
    pyplot.imshow(im0, cmap, **kwargs)
    pyplot.subplot(222)
    pyplot.imshow(im1, cmap, **kwargs)
    pyplot.subplot(223)
    pyplot.imshow(im3, cmap, **kwargs)
    pyplot.subplot(224)
    pyplot.imshow(im2, cmap, **kwargs)
    pyplot.show()
img1 = plt.imread(set175[1])
img2 = plt.imread(set175[3])
plt.figure(figsize=(8,10))
plt.subplot(121)
plt.imshow(img1)
plt.subplot(122)
plt.imshow(img2)
img3, scale, angle, (t0, t1) = similarity(img1[:,:,2], img2[:,:,2])
imshow(img1[:,:,2], img2[:,:,2], img3)
# just diff
plt.imshow(img3 - img1[:,:,2])
# well, was working in a standalone private notebook, need to figure out what's off.
# it's fairly obviously offset/translation/shift is wrong somehow.
plt.imshow(img3)
from skimage.transform import downscale_local_mean

img1 = plt.imread(set175[1])[:,:,2]
img2 = plt.imread(set175[3])[:,:,2]
np.shape(img1), np.shape(img2)
img1ds = downscale_local_mean(img1, (10, 10))
img2ds = downscale_local_mean(img2, (10, 10))
plt.figure(figsize=(8,10))
plt.subplot(121)
plt.imshow(img1ds[:225,:300])
plt.subplot(122)
plt.imshow(img2ds[:225,:300])
def read_and_downscale(f):
    img = plt.imread(f)
    ds = downscale_local_mean(img[:,:,2], (10, 10))
    return ds[:225, :300]
set79 = [read_and_downscale("../input/train_sm/set79_" + str(n) + ".jpeg") for n in range(1, 6)]
def read_ds_set(setn):
    match_pre = "../input/train_sm/set" + str(setn) + "_"
    return [read_and_downscale(match_pre + str(n) + ".jpeg") for n in range(1, 6)]
set79 = read_ds_set(79)
set285 = read_ds_set(285)
set35 = read_ds_set(35)
set175 = read_ds_set(175)
plt.imshow(set79[0])
plt.imshow(set285[2])
plt.imshow(set35[3])
def get_pairs(imgls):
    pairl = []
    for imgl in imgls:
        pairl += [(a,b) for a,b in zip(imgl[:-1], imgl[1:])]
    return pairl
            
paired = get_pairs([set35, set79, set285, set175])
# Let's just start with a dorky example.
rev_pairs = [(imgb, imga) for imga, imgb in paired]
img_a, img_b = rev_pairs[0]
concat_img = np.vstack((img_a, img_b))
print(np.shape(concat_img))
plt.imshow(concat_img)
import random

def concatter(imgpairs):
    for a, b in imgpairs:
        yield np.vstack((a, b))

concats = [cimg for cimg in concatter(paired + rev_pairs)]
random.shuffle(concats)
plt.figure(figsize=(10,15))
plt.subplot(321)
plt.imshow(concats[0])
plt.subplot(322)
plt.imshow(concats[1])
plt.subplot(323)
plt.imshow(concats[2])
plt.subplot(324)
plt.imshow(concats[3])
plt.subplot(325)
plt.imshow(concats[4])
plt.subplot(326)
plt.imshow(concats[5])
plt.show()
# Need to simplify or build up next block from convnet template.