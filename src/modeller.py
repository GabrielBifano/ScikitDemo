import random
import numpy as np
import pandas as pd
from timer import timer
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from ansi import ANSI

class Modeller:
    
    def __init__(self):
        self.cache = {}
        self.rand = lambda : random.randint(0, 0xFFFFFFFF - 1)
        
        '''K-Means configs'''
        self.max_iter = 300
        self.init = 'k-means++'
        self.n_init = 10
        self.tol = 1e-4
        self.verbose = 0
        self.copy_x = True
        self.km_algorithm = 'lloyd'
        
        '''DBSCAN configs'''
        self.min_samples = 10
        self.dbs_metric = 'euclidean'
        self.metric_params = None
        self.dbs_algorithm = 'auto'
        self.leaf_size = 30
        self.p = None
        self.n_jobs = None

        '''Agglomerative Clustering configs'''
        self.n_clusters = 2
        self.ac_metric = None
        self.memory = None
        self.connectivity = None
        self.compute_full_tree = 'auto'
        self.linkage = 'ward'
        self.distance_threshold = None
        self.compute_distances = False


    def dump_cache(self) -> None:
        self.cache.clear()

    def keys(self, plot: bool=True):
        if plot == False:
            return self.cache.keys()

        a = ANSI()
        if len(self.cache) == 0:
            print(f'{a.b}{a.green}Cache{a.res} is {a.b}{a.red}empty{a.res}')
            return self.cache.keys()
        
        print(f'Models inside {a.b}{a.green}cache:{a.res}')
        for key in self.cache.keys():
            print(f'  {a.cyan}* {key}{a.res}')
        return self.cache.keys()


    def config(self,
            max_iter=       None,
            init=           None,
            n_init=         None,
            tol=            None,
            verbose=        None,
            copy_x=         None,
            km_algorithm=   None,

            min_samples=    None,
            dbs_metric=     None,
            metric_params=  None,
            dbs_algorithm=  None,
            leaf_size=      None,
            p=              None,
            n_jobs=         None,

            n_clusters=     None,
            ac_metric=      None,
            memory=         None,
            connectivity=   None,
            compute_full_tree=None,
            linkage=        None,
            distance_threshold=None,
            compute_distances=None,
        ):
        
        self.max_iter = max_iter if max_iter is not None else self.max_iter
        self.n_init = n_init if n_init is not None else self.n_init
        self.init = init if init is not None else self.init
        self.tol = tol if tol is not None else self.tol
        self.verbose = verbose if verbose is not None else self.verbose
        self.copy_x = copy_x if copy_x is not None else self.copy_x
        self.km_algorithm = km_algorithm if km_algorithm is not None else self.km_algorithm
        
        self.min_samples = min_samples if min_samples is not None else self.min_samples
        self.dbs_metric = dbs_metric if dbs_metric is not None else self.dbs_metric
        self.metric_params = metric_params if metric_params is not None else self.metric_params
        self.dbs_algorithm = dbs_algorithm if dbs_algorithm is not None else self.dbs_algorithm
        self.leaf_size = leaf_size if leaf_size is not None else self.leaf_size
        self.p = p if p is not None else self.p
        self.n_jobs = n_jobs if n_jobs is not None else self.n_jobs

        self.n_clusters = n_clusters if n_clusters is not None else self.n_clusters
        self.ac_metric = ac_metric if ac_metric is not None else self.ac_metric
        self.memory = memory if memory is not None else self.memory
        self.connectivity = connectivity if connectivity is not None else self.connectivity
        self.compute_full_tree = compute_full_tree if compute_full_tree is not None else self.compute_full_tree
        self.linkage = linkage if linkage is not None else self.linkage
        self.distance_threshold = distance_threshold if distance_threshold is not None else self.distance_threshold
        self.compute_distances = compute_distances if compute_distances is not None else self.compute_distances

        return self


    def save_model(self, model, key: str) -> None:
        self.cache[key] = model

    def get_model(self, key: str):
        if key not in self.cache.keys(): return None
        return self.cache[key]

    @timer
    def optimal_silhouette(self, data:pd.DataFrame, plot:bool=False, from_:int=2, to_:int=15) -> int:
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
    def kmeans(self, data: pd.DataFrame, k: int, key: str='kmeans') -> tuple:
        model = KMeans(
            n_clusters = k,
            max_iter = self.max_iter,
            init = self.init,
            n_init = self.n_init,
            tol = self.tol,
            verbose = self.verbose,
            copy_x = self.copy_x,
            algorithm = self.km_algorithm,
            random_state = self.rand()
        )
        model.fit(data)
        self.save_model(model, key)
        data['labels'] = model.labels_
        return (data, model.labels_, key)

    @timer
    def dbscan(self, data: pd.DataFrame, eps: float, key: str='dbscan') -> tuple:
        model = DBSCAN(
            eps=eps,
            min_samples = self.min_samples,
            metric = self.dbs_metric,
            metric_params = self.metric_params,
            algorithm = self.dbs_algorithm,
            leaf_size = self.leaf_size,
            p = self.p,
            n_jobs = self.n_jobs,
        )
        model.fit(data)
        self.save_model(model, key)
        data['labels'] = model.labels_
        return (data, model.labels_, key)

    @timer
    def agglomerative_clustering(self, data: pd.DataFrame, k: int, key: str='ag-clustering') -> tuple:
        model = AgglomerativeClustering(
            n_clusters = k,
            metric = self.ac_metric,
            memory = self.memory,
            connectivity = self.connectivity,
            compute_full_tree = self.compute_full_tree,
            linkage = self.linkage,
            distance_threshold = self.distance_threshold,
            compute_distances = self.compute_distances,
        )
        model.fit(data)
        self.save_model(model, key)
        data['labels'] = model.labels_
        return (data, model.labels_, key)

    @timer
    def hdbscan():
        pass