from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import numpy as np

def pca_redutor(feature_array, numFinalDimenison=2):
  redutor = PCA(n_components=numFinalDimenison)
  return redutor.fit_transform(feature_array)

def svd_redutor(feature_array, numFinalDimenison=2):
  redutor = TruncatedSVD(n_components=numFinalDimenison)
  return redutor.fit_transform(feature_array)

def tsne_redutor(feature_array, numFinalDimenison=2):
  array = np.asarray(feature_array)
  redutor = TSNE(n_components=numFinalDimenison)
  return redutor.fit_transform(array)