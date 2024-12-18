import umap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def umap_embed(image_features, labels, title='Latent Space'):
    features = image_features
    labels = labels
    umap_obj = umap.UMAP(n_neighbors=5, min_dist=0.3, random_state=42)
    embedding = umap_obj.fit_transform(features)
    df = pd.DataFrame(embedding[:,0:2],columns=['UMAP Dimension 1', 'UMAP Dimension 2'])
    fig, ax = plt.subplots(figsize=(10,8))
    sns.scatterplot(data = df, x = 'UMAP Dimension 1', y = 'UMAP Dimension 2', hue = labels, legend = False, ax=ax).set_title(title)
    #sns.move_legend(ax, "best")
    plt.show()

