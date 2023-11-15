import random
import numpy as np
import pandas as pd

from timer import timer
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

from ansi import BOLD, RESET, CYAN, GREEN

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
        self.metric = 'euclidean'
        self.metric_params = None
        self.dbs_algorithm = 'auto'
        self.leaf_size = 30
        self.p = None
        self.n_jobs = None


    def config(self,
            max_iter=       None,
            init=           None,
            n_init=         None,
            tol=            None,
            verbose=        None,
            copy_x=         None,
            km_algorithm=   None,

            min_samples=    None,
            metric=         None,
            metric_params=  None,
            dbs_algorithm=  None,
            leaf_size=      None,
            p=              None,
            n_jobs=         None,
        ):
        
        self.max_iter = max_iter if max_iter != None else self.max_iter
        self.n_init = n_init if n_init != None else self.n_init
        self.init = init if init != None else self.init
        self.tol = tol if tol != None else self.tol
        self.verbose = verbose if verbose != None else self.verbose
        self.copy_x = copy_x if copy_x != None else self.copy_x
        self.km_algorithm = km_algorithm if km_algorithm != None else self.km_algorithm
        
        self.min_samples = min_samples if min_samples != None else self.min_samples
        self.metric = metric if metric != None else self.metric
        self.metric_params = metric_params if metric_params != None else self.metric_params
        self.dbs_algorithm = dbs_algorithm if dbs_algorithm != None else self.dbs_algorithm
        self.leaf_size = leaf_size if leaf_size != None else self.leaf_size
        self.p = p if p != None else self.p
        self.n_jobs = n_jobs if n_jobs != None else self.n_jobs

        return self


    #TODO def metrics() must print all specs of a given model
    def metrics(self, key):
        pass

    def keys(self, p: bool=True):
        if p == False: return self.cache.keys()
        print(f'models inside {BOLD}{GREEN}cache:{RESET}')
        for key in self.cache.keys():
            print(f'  {CYAN}* {key}{RESET}')

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
            max_iter = self.max_iter,
            n_init = self.n_init,
            random_state = self.rand()
        )
        model.fit(data)
        self.cache[key] = model
        processed = data.copy()
        processed['labels'] = model.labels_
        return (processed, model.labels_, key)

    @timer
    def dbscan(self, data: pd.DataFrame, eps: float, key: str='dbscan'):
        model = DBSCAN(
            eps=eps,
            min_samples=self.min_samples,
        )
        model.fit(data)
        self.cache[key] = model
        processed = data.copy()
        processed['labels'] = model.labels_
        return (processed, model.labels_, key)

    @timer
    def agglomerative_clustering():
        pass

    @timer
    def hdbscan():
        pass