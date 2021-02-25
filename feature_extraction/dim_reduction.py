
from sklearn.decomposition import PCA

def principal_component_analysis(x, pca=None):
    ''' Apply PCA for dimensionality reduction. Select number of components 
    such that cumulative explained variance > 95%. '''
    if pca:
        x = pca.transform(x)
        return x
    else:
        pca = PCA(n_components=0.95).fit(x)
        x = pca.transform(x)
        return x, pca

