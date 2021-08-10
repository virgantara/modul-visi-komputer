import skimage.io as skm
import skimage.feature as ft
import numpy as np

img = skm.imread('dataset/cameraman.png',as_gray=True)
img = np.uint8(img * 255)

glcm = ft.greycomatrix(img, distances=[6], angles=[0],levels=256, normed=True)
dissimilarity = ft.greycoprops(P=glcm, prop='dissimilarity')
correlation = ft.greycoprops(P=glcm, prop='correlation')
homogeneity = ft.greycoprops(P=glcm, prop='homogeneity')
energy = ft.greycoprops(P=glcm, prop='energy')
contrast = ft.greycoprops(P=glcm, prop='contrast')
ASM = ft.greycoprops(P=glcm, prop='ASM')
glcm_props = [dissimilarity, correlation, homogeneity, energy, contrast, ASM]
print('Dissimilarity',dissimilarity,'\nCorrelation',correlation,
'\nHomogeneity',homogeneity,'\nEnergy',energy,'\nContrast',contrast,
'\nASM',ASM)