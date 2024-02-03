from sklearn import decomposition
def reduce_dim_fit(x):
    pca = decomposition.PCA(n_components=10)
    pca.fit(x)  # use a set of vectors to learn the PCA transformation
    return pca
def reduce_dim_infer(pca, x):
    z = pca.transform(x)  # transform a set of vectors to reduce their dim
    return z  # (it is possible that Z=X)