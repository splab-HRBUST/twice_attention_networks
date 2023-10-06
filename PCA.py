import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
def pca(inputs,n_component):
    print("inputs.shape = ",inputs.shape)
    sc = StandardScaler()
    inputs_fit = sc.fit_transform(inputs)
    box_pca = PCA(n_components=n_component)
    print(box_pca.fit(inputs))
    output_box = box_pca.fit_transform(inputs_fit)
    print("output_box.shape = ",output_box.shape)
    return output_box
if __name__ == "__main__":
    n_samples = 251
    dim = 192
    noise = np.random.randn(n_samples, dim)
    pca(inputs=noise, n_component=40)
