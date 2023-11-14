import random
import numpy as np
import pandas as pd

from timer import timer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class Modeller:
    
    def __init__(self):
        self.cache = {}
        self.rand = lambda : random.randint(0, 0xFFFFFFFF - 1)

    #TODO def show_cache() must print all specs of a given model

    def dump_cache(self):
        self.cache.clear()

    @timer
    def optimal_silhouette(self, data:pd.DataFrame, plot:bool=False, from_:int=2, to_:int=15):
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

    @timer
    def kmeans(self, data: pd.DataFrame, k: int, key: str='kmeans'):
        model = KMeans(
            n_clusters = k,
            init = 'k-means++',
            max_iter = 300,
            n_init = 10,
            random_state = self.rand()
        )
        model.fit(data)
        self.cache[key] = model
        processed = data.copy()
        processed['labels'] = model.labels_
        return (processed, model.labels_, key)