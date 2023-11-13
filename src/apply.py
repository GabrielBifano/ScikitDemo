import random
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class Apply:
    
    def __init__(self):
        self.rand = lambda : random.randint(0, 0xFFFFFFFF - 1)

    def get_specs(model: KMeans):
        """"Expects K-Means model from SK-Learn and returns it's attributes"""
        return {
            'inertia': model.inertia_,
            'clusters': model.cluster_centers_,
            'labels': model.labels_,
            'iterations': model.n_iter_,
            'dimensions': model.n_features_in_,
            'features' : model.features_names_in_,
        }

    def optimal_silhouette(self, data:pd.DataFrame, plot:bool, from_:int=2, to_:int=15):
        
        if from_ < 2: raise ValueError("from_ must be greater than 1")
        if from_ >= to_: raise ValueError("from_ must be less than to_")
        
        scores = []
        range_cl = [x for x in range(from_, to_)]

        for n in range_cl:
            model = KMeans(
                n_clusters = n,
                init = 'k-means++',
                max_iter = 300,
                n_init = 10,
                random_state = self.rand()
            )
            labels = model.fit_predict(data)

            avg = silhouette_score(data, labels)
            scores.append(avg)
        
        if plot:
            # TODO implement graphic repr for silhoeutte score
            True

        opt = range_cl[np.argmax(scores)]
        return opt

    def kmeans(self, data: pd.DataFrame, k: int):
        
        model = KMeans(
            n_clusters = k,
            init = 'k-means++',
            max_iter = 300,
            n_init = 10,
            random_state = self.rand()
        )

        model.fit(data)
        processed = data.copy()
        processed['labels'] = model.labels_
        specs = self.get_specs()

        return (processed, specs)