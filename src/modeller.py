import random
import numpy as np
import pandas as pd
from timer import timer
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import BisectingKMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import HDBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import FeatureAgglomeration
from sklearn.cluster import Birch
from sklearn.cluster import MeanShift
from sklearn.metrics import silhouette_score

from ansi import ANSI

class Modeller:
    
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

    def save_model(self, model, key: str) -> None:
        self.cache[key] = model

    def get_model(self, key: str):
        if key not in self.cache.keys(): return None
        return self.cache[key]

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
                random_state = self.rand(),
            )
            labels = model.fit_predict(data)
            avg = silhouette_score(data, labels)
            scores.append(avg)
        if plot:
            # TODO implement graphic repr for silhoeutte score
            True
        opt = range_cl[np.argmax(scores)]
        return opt

    def kmeans(self, data: pd.DataFrame, k: int, key: str='kmeans') -> tuple:
        model = KMeans(
            n_clusters = k,
            max_iter = self.kmeans_max_iter,
            init = self.kmeans_init,
            n_init = self.kmeans_n_init,
            tol = self.kmeans_tol,
            verbose = self.kmeans_verbose,
            copy_x = self.kmeans_copy_x,
            algorithm = self.kmeans_algorithm,
            random_state = self.rand(),
        )
        model.fit(data)
        self.save_model(model, key)
        data['labels'] = model.labels_
        return (data, model.labels_, key)

    def gmixture(self, data: pd.DataFrame, n_components: int, key: str='gmixture') -> tuple:
        model = GaussianMixture(
            n_components = n_components,
            covariance_type = self.gmixture_covariance_type,
            tol = self.gmixture_tol,
            reg_covar = self.gmixture_reg_covar,
            max_iter = self.gmixture_max_iter,
            n_init = self.gmixture_n_init,
            init_params = self.gmixture_init_params,
            weights_init = self.gmixture_weights_init,
            means_init = self.gmixture_means_init,
            precisions_init = self.gmixture_precisions_init,
            random_state = self.rand(),
            warm_start = self.gmixture_warm_start,
            verbose = self.gmixture_verbose,
            verbose_interval = self.gmixture_verbose_interval,
        )
        model.fit(data)
        self.save_model(model, key)
        data['labels'] = model.predict(data)
        return (data, data['labels'], key)

    def bskmeans(self, data: pd.DataFrame, k: int, key: str='kmeans') -> tuple:
        model = BisectingKMeans(
            n_clusters = k,
            init = self.bskmeans_init,
            n_init = self.bskmeans_n_init,
            random_state = self.rand(),
            max_iter = self.bskmeans_max_iter,
            verbose = self.bskmeans_verbose,
            tol = self.bskmeans_tol,
            copy_x = self.bskmeans_copy_x,
            algorithm = self.bskmeans_algorithm,
            bisecting_strategy = self.bskmeans_bisecting_strategy,
        )
        model.fit(data)
        self.save_model(model, key)
        data['labels'] = model.labels_
        return (data, model.labels_, key)


    def dbscan(self, data: pd.DataFrame, eps: float, key: str='dbscan') -> tuple:
        model = DBSCAN(
            eps=eps,
            min_samples = self.dbscan_min_samples,
            metric = self.dbscan_metric,
            metric_params = self.dbscan_metric_params,
            algorithm = self.dbscan_algorithm,
            leaf_size = self.dbscan_leaf_size,
            p = self.dbscan_p,
            n_jobs = self.dbscan_n_jobs,
        )
        model.fit(data)
        self.save_model(model, key)
        data['labels'] = model.labels_
        return (data, model.labels_, key)

    def hdbscan(self, data: pd.DataFrame, eps, key: str='hdbscan'):
        model = HDBSCAN(
            min_cluster_size=self.hdbscan_min_cluster_size,
            min_samples=self.hdbscan_min_samples,
            # cluster_selection_epsilon=self.hdbscan_cluster_selection_epsilon,
            cluster_selection_epsilon=eps,
            max_cluster_size=self.hdbscan_max_cluster_size,
            metric=self.hdbscan_metric,
            metric_params=self.hdbscan_metric_params,
            alpha=self.hdbscan_alpha,
            algorithm=self.hdbscan_algorithm,
            leaf_size=self.hdbscan_leaf_size,
            n_jobs=self.hdbscan_n_jobs,
            cluster_selection_method=self.hdbscan_cluster_selection_method,
            allow_single_cluster=self.hdbscan_allow_single_cluster,
            store_centers=self.hdbscan_store_centers,
            copy=self.hdbscan_copy
        )
        model.fit(data)
        self.save_model(model, key)
        data['labels'] = model.labels_
        return (data, model.labels_, key)

    def affprop(self, key: str='affprop'):
        pass

    def agcluster(self, data: pd.DataFrame, k: int, key: str='agcluster'):
        model = AgglomerativeClustering(
            n_clusters = k,
            metric = self.agcluster_ac_metric,
            memory = self.agcluster_memory,
            connectivity = self.agcluster_connectivity,
            compute_full_tree = self.agcluster_compute_full_tree,
            linkage = self.agcluster_linkage,
            distance_threshold = self.agcluster_distance_threshold,
            compute_distances = self.agcluster_compute_distances,
        )
        model.fit(data)
        self.save_model(model, key)
        data['labels'] = model.labels_
        return (data, model.labels_, key)

    def featagg(self, data: pd.DataFrame, k: int, key: str='featagg'):
        model = FeatureAgglomeration(
            n_clusters = k,
            metric = self.featagg_metric,
            memory = self.featagg_memory,
            connectivity = self.featagg_connectivity,
            compute_full_tree = self.featagg_compute_full_tree,
            linkage = self.featagg_linkage,
            pooling_func =  self.featagg_pooling_func,
            distance_threshold = self.featagg_distance_threshold,
            compute_distances = self.featagg_compute_distances,
        )
        model.fit(data)
        self.save_model(model, key)
        data['labels'] = model.labels_
        return (data, model.labels_, key)

    def birch(self, data: pd.DataFrame, k: int, key: str='birch'):
        model = Birch(
            n_clusters=k,
            threshold=self.birch_threshold,
            branching_factor=self.birch_branching_factor,
            compute_labels=self.birch_compute_labels,
            copy=self.birch_copy,
        )
        model.fit(data)
        self.save_model(model, key)
        data['labels'] = model.labels_
        return (data, model.labels_, key)

    def meanshift(self, key: str='meanshift'):
        model = MeanShift(
            n_jobs = self.meanshift_n_jobs,
            bandwidth = self.meanshift_bandwidth,
            seeds = self.meanshift_seeds,
            bin_seeding = self.meanshift_bin_seeding,
            min_bin_freq = self.meanshift_min_bin_freq,
            cluster_all = self.meanshift_cluster_all,
            max_iter = self.meanshift_max_iter,
        )
        pass

    def optics(self, key: str='optics'):
        pass

    def spcluster(self, key: str='spcluster'):
        pass





    def __init__(self):
        
        self.cache = {}
        self.rand = lambda : random.randint(0, 0xFFFFFFFF - 1)
        
        '''K-Means configs'''
        self.kmeans_max_iter = 300
        self.kmeans_init = 'k-means++'
        self.kmeans_n_init = 10
        self.kmeans_tol = 1e-4
        self.kmeans_verbose = 0
        self.kmeans_copy_x = True
        self.kmeans_algorithm = 'lloyd'

        '''Gaussian Mixture configs'''
        self.gmixture_covariance_type = 'full'
        self.gmixture_tol = 0.001
        self.gmixture_reg_covar = 1e-06
        self.gmixture_max_iter = 100
        self.gmixture_n_init = 1
        self.gmixture_init_params = 'kmeans'
        self.gmixture_weights_init = None
        self.gmixture_means_init = None
        self.gmixture_precisions_init = None
        self.gmixture_warm_start = False
        self.gmixture_verbose = 0
        self.gmixture_verbose_interval = 10

        '''Bisecting K-Means'''
        self.bskmeans_init = 'random'
        self.bskmeans_n_init = 1
        self.bskmeans_max_iter = 300
        self.bskmeans_verbose = 0
        self.bskmeans_tol = 1e-4
        self.bskmeans_copy_x = True
        self.bskmeans_algorithm = 'lloyd'
        self.bskmeans_bisecting_strategy = 'biggest_inertia'

        '''DBSCAN configs'''
        self.dbscan_min_samples = 10
        self.dbscan_metric = 'euclidean'
        self.dbscan_metric_params = None
        self.dbscan_algorithm = 'auto'
        self.dbscan_leaf_size = 30
        self.dbscan_p = None
        self.dbscan_n_jobs = None

        '''HDBSCAN configs'''
        self.hdbscan_min_cluster_size = 5
        self.hdbscan_min_samples = None
        self.hdbscan_cluster_selection_epsilon = 0.0
        self.hdbscan_max_cluster_size = None
        self.hdbscan_metric = 'euclidean'
        self.hdbscan_metric_params = None
        self.hdbscan_alpha = 1.0
        self.hdbscan_algorithm = 'auto'
        self.hdbscan_leaf_size = 40
        self.hdbscan_n_jobs = None
        self.hdbscan_cluster_selection_method = 'eom'
        self.hdbscan_allow_single_cluster = False
        self.hdbscan_store_centers = None
        self.hdbscan_copy = False

        '''Affinity Propagation configs'''
        self.affprop_damping = 0.5
        self.affprop_max_iter = 200
        self.affprop_convergence_iter = 15
        self.affprop_copy = True
        self.affprop_preference = None
        self.affprop_affinity = 'euclidean'
        self.affprop_verbose = False

        '''Agglomerative Clustering configs'''
        self.agcluster_n_clusters = 2
        self.agcluster_metric = None
        self.agcluster_memory = None
        self.agcluster_connectivity = None
        self.agcluster_compute_full_tree = 'auto'
        self.agcluster_linkage = 'ward'
        self.agcluster_distance_threshold = None
        self.agcluster_compute_distances = False

        '''Feature Agglomeration configs'''
        self.featagg_n_clusters = 2
        self.featagg_metric = None
        self.featagg_memory = None
        self.featagg_connectivity = None
        self.featagg_compute_full_tree = 'auto'
        self.featagg_linkage = 'ward'
        self.featagg_pooling_func = np.mean
        self.featagg_distance_threshold = None
        self.featagg_compute_distances = False

        '''Birch configs'''
        self.birch_threshold = 0.5
        self.birch_branching_factor = 50
        self.birch_n_clusters = 3
        self.birch_compute_labels = True
        self.birch_copy = True

        '''Mean Shift configs'''
        self.meanshift_bandwidth = None
        self.meanshift_seeds = None
        self.meanshift_bin_seeding = False
        self.meanshift_min_bin_freq = 1
        self.meanshift_cluster_all = True
        self.meanshift_n_jobs = None
        self.meanshift_max_iter = 300

        '''OPTICS configs'''
        self.optics_min_samples = 5
        self.optics_max_eps = float('inf')
        self.optics_metric = 'minkowski'
        self.optics_p = 2
        self.optics_metric_params = None
        self.optics_cluster_method = 'xi'
        self.optics_eps = None
        self.optics_xi = 0.05
        self.optics_predecessor_correction = True
        self.optics_min_cluster_size = None
        self.optics_algorithm = 'auto'
        self.optics_leaf_size = 30
        self.optics_memory = None
        self.optics_n_jobs = None

        '''Spectral Clustering configs'''
        self.spcluster_n_clusters = 8
        self.spcluster_eigen_solver = None
        self.spcluster_n_components = None
        self.spcluster_n_init = 10
        self.spcluster_gamma = 1.0
        self.spcluster_affinity = 'rbf'
        self.spcluster_n_neighbors = 10
        self.spcluster_eigen_tol = 'auto'
        self.spcluster_assign_labels = 'kmeans'
        self.spcluster_degree = 3
        self.spcluster_coef0 = 1
        self.spcluster_kernel_params = None
        self.spcluster_n_jobs = None
        self.spcluster_verbose = False



    def config(self,

            # K-Means
            kmeans_max_iter = None,
            kmeans_init = None,
            kmeans_n_init = None,
            kmeans_tol = None,
            kmeans_verbose = None,
            kmeans_copy_x = None,
            kmeans_algorithm = None,

            # Gaussian Mixture
            gmixture_covariance_type = None,
            gmixture_tol = None,
            gmixture_reg_covar = None,
            gmixture_max_iter = None,
            gmixture_n_init = None,
            gmixture_init_params = None,
            gmixture_weights_init = None,
            gmixture_means_init = None,
            gmixture_precisions_init = None,
            gmixture_warm_start = None,
            gmixture_verbose = None,
            gmixture_verbose_interval = None,

            # Bisecting K-Means
            bskmeans_init = None,
            bskmeans_n_init = None,
            bskmeans_max_iter = None,
            bskmeans_verbose = None,
            bskmeans_tol = None,
            bskmeans_copy_x = None,
            bskmeans_algorithm = None,
            bskmeans_bisecting_strategy = None,

            # DBSCAN
            dbscan_min_samples = None,
            dbscan_metric = None,
            dbscan_metric_params = None,
            dbscan_algorithm = None,
            dbscan_leaf_size = None,
            dbscan_p = None,
            dbscan_n_jobs = None,

            # HDBSCAN
            hdbscan_min_cluster_size = None,
            hdbscan_min_samples = None,
            hdbscan_cluster_selection_epsilon = None,
            hdbscan_max_cluster_size = None,
            hdbscan_metric = None,
            hdbscan_metric_params = None,
            hdbscan_alpha = None,
            hdbscan_algorithm = None,
            hdbscan_leaf_size = None,
            hdbscan_n_jobs = None,
            hdbscan_cluster_selection_method = None,
            hdbscan_allow_single_cluster = None,
            hdbscan_store_centers = None,
            hdbscan_copy = None,

            # Affinity Propagation
            affprop_damping = None,
            affprop_max_iter = None,
            affprop_convergence_iter = None,
            affprop_copy = None,
            affprop_preference = None,
            affprop_affinity = None,
            affprop_verbose = None,

            # Agglomerative Clustering
            agcluster_n_clusters = None,
            agcluster_metric = None,
            agcluster_memory = None,
            agcluster_connectivity = None,
            agcluster_compute_full_tree = None,
            agcluster_linkage = None,
            agcluster_distance_threshold = None,
            agcluster_compute_distances = None,

            # Feature Agglomeration
            featagg_n_clusters = None,
            featagg_metric = None,
            featagg_memory = None,
            featagg_connectivity = None,
            featagg_compute_full_tree = None,
            featagg_linkage = None,
            featagg_pooling_func = None,
            featagg_distance_threshold = None,
            featagg_compute_distances = None,

            # Birch
            birch_threshold = None,
            birch_branching_factor = None,
            birch_n_clusters = None,
            birch_compute_labels = None,
            birch_copy = None,

            # Mean Shift
            meanshift_bandwidth = None,
            meanshift_seeds = None,
            meanshift_bin_seeding = None,
            meanshift_min_bin_freq = None,
            meanshift_cluster_all = None,
            meanshift_n_jobs = None,
            meanshift_max_iter = None,

            # OPTICS
            optics_min_samples = None,
            optics_max_eps = None,
            optics_metric = None,
            optics_p = None,
            optics_metric_params = None,
            optics_cluster_method = None,
            optics_eps = None,
            optics_xi = None,
            optics_predecessor_correction = None,
            optics_min_cluster_size = None,
            optics_algorithm = None,
            optics_leaf_size = None,
            optics_memory = None,
            optics_n_jobs = None,

            # Spectral Clustering
            spcluster_n_clusters = None,
            spcluster_eigen_solver = None,
            spcluster_n_components = None,
            spcluster_n_init = None,
            spcluster_gamma = None,
            spcluster_affinity = None,
            spcluster_n_neighbors = None,
            spcluster_eigen_tol = None,
            spcluster_assign_labels = None,
            spcluster_degree = None,
            spcluster_coef0 = None,
            spcluster_kernel_params = None,
            spcluster_n_jobs = None,
            spcluster_verbose = None,
        ):
        
        # K-Means
        self.kmeans_max_iter = kmeans_max_iter if kmeans_max_iter is not None else self.kmeans_max_iter
        self.kmeans_init = kmeans_init if kmeans_init is not None else self.kmeans_init
        self.kmeans_n_init = kmeans_n_init if kmeans_n_init is not None else self.kmeans_n_init
        self.kmeans_tol = kmeans_tol if kmeans_tol is not None else self.kmeans_tol
        self.kmeans_verbose = kmeans_verbose if kmeans_verbose is not None else self.kmeans_verbose
        self.kmeans_copy_x = kmeans_copy_x if kmeans_copy_x is not None else self.kmeans_copy_x
        self.kmeans_algorithm = kmeans_algorithm if kmeans_algorithm is not None else self.kmeans_algorithm

        # Gaussian Mixture
        self.gmixture_covariance_type = gmixture_covariance_type if gmixture_covariance_type is not None else self.gmixture_covariance_type
        self.gmixture_tol = gmixture_tol if gmixture_tol is not None else self.gmixture_tol
        self.gmixture_reg_covar = gmixture_reg_covar if gmixture_reg_covar is not None else self.gmixture_reg_covar
        self.gmixture_max_iter = gmixture_max_iter if gmixture_max_iter is not None else self.gmixture_max_iter
        self.gmixture_n_init = gmixture_n_init if gmixture_n_init is not None else self.gmixture_n_init
        self.gmixture_init_params = gmixture_init_params if gmixture_init_params is not None else self.gmixture_init_params
        self.gmixture_weights_init = gmixture_weights_init if gmixture_weights_init is not None else self.gmixture_weights_init
        self.gmixture_means_init = gmixture_means_init if gmixture_means_init is not None else self.gmixture_means_init
        self.gmixture_precisions_init = gmixture_precisions_init if gmixture_precisions_init is not None else self.gmixture_precisions_init
        self.gmixture_warm_start = gmixture_warm_start if gmixture_warm_start is not None else self.gmixture_warm_start
        self.gmixture_verbose = gmixture_verbose if gmixture_verbose is not None else self.gmixture_verbose
        self.gmixture_verbose_interval = gmixture_verbose_interval if gmixture_verbose_interval is not None else self.gmixture_verbose_interval

        # Bisecting K-Means
        self.bskmeans_init = bskmeans_init if bskmeans_init is not None else self.bskmeans_init
        self.bskmeans_n_init = bskmeans_n_init if bskmeans_n_init is not None else self.bskmeans_n_init
        self.bskmeans_max_iter = bskmeans_max_iter if bskmeans_max_iter is not None else self.bskmeans_max_iter
        self.bskmeans_verbose = bskmeans_verbose if bskmeans_verbose is not None else self.bskmeans_verbose
        self.bskmeans_tol = bskmeans_tol if bskmeans_tol is not None else self.bskmeans_tol
        self.bskmeans_copy_x = bskmeans_copy_x if bskmeans_copy_x is not None else self.bskmeans_copy_x
        self.bskmeans_algorithm = bskmeans_algorithm if bskmeans_algorithm is not None else self.bskmeans_algorithm
        self.bskmeans_bisecting_strategy = bskmeans_bisecting_strategy if bskmeans_bisecting_strategy is not None else self.bskmeans_bisecting_strategy

        # DBSCAN
        self.dbscan_min_samples = dbscan_min_samples if dbscan_min_samples is not None else self.dbscan_min_samples
        self.dbscan_metric = dbscan_metric if dbscan_metric is not None else self.dbscan_metric
        self.dbscan_metric_params = dbscan_metric_params if dbscan_metric_params is not None else self.dbscan_metric_params
        self.dbscan_algorithm = dbscan_algorithm if dbscan_algorithm is not None else self.dbscan_algorithm
        self.dbscan_leaf_size = dbscan_leaf_size if dbscan_leaf_size is not None else self.dbscan_leaf_size
        self.dbscan_p = dbscan_p if dbscan_p is not None else self.dbscan_p
        self.dbscan_n_jobs = dbscan_n_jobs if dbscan_n_jobs is not None else self.dbscan_n_jobs

        # HDBSCAN
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size if hdbscan_min_cluster_size is not None else self.hdbscan_min_cluster_size
        self.hdbscan_min_samples = hdbscan_min_samples if hdbscan_min_samples is not None else self.hdbscan_min_samples
        self.hdbscan_cluster_selection_epsilon = hdbscan_cluster_selection_epsilon if hdbscan_cluster_selection_epsilon is not None else self.hdbscan_cluster_selection_epsilon
        self.hdbscan_max_cluster_size = hdbscan_max_cluster_size if hdbscan_max_cluster_size is not None else self.hdbscan_max_cluster_size
        self.hdbscan_metric = hdbscan_metric if hdbscan_metric is not None else self.hdbscan_metric
        self.hdbscan_metric_params = hdbscan_metric_params if hdbscan_metric_params is not None else self.hdbscan_metric_params
        self.hdbscan_alpha = hdbscan_alpha if hdbscan_alpha is not None else self.hdbscan_alpha
        self.hdbscan_algorithm = hdbscan_algorithm if hdbscan_algorithm is not None else self.hdbscan_algorithm
        self.hdbscan_leaf_size = hdbscan_leaf_size if hdbscan_leaf_size is not None else self.hdbscan_leaf_size
        self.hdbscan_n_jobs = hdbscan_n_jobs if hdbscan_n_jobs is not None else self.hdbscan_n_jobs
        self.hdbscan_cluster_selection_method = hdbscan_cluster_selection_method if hdbscan_cluster_selection_method is not None else self.hdbscan_cluster_selection_method
        self.hdbscan_allow_single_cluster = hdbscan_allow_single_cluster if hdbscan_allow_single_cluster is not None else self.hdbscan_allow_single_cluster
        self.hdbscan_store_centers = hdbscan_store_centers if hdbscan_store_centers is not None else self.hdbscan_store_centers
        self.hdbscan_copy = hdbscan_copy if hdbscan_copy is not None else self.hdbscan_copy

        # Affinity Propagation
        self.affprop_damping = affprop_damping if affprop_damping is not None else self.affprop_damping
        self.affprop_max_iter = affprop_max_iter if affprop_max_iter is not None else self.affprop_max_iter
        self.affprop_convergence_iter = affprop_convergence_iter if affprop_convergence_iter is not None else self.affprop_convergence_iter
        self.affprop_copy = affprop_copy if affprop_copy is not None else self.affprop_copy
        self.affprop_preference = affprop_preference if affprop_preference is not None else self.affprop_preference
        self.affprop_affinity = affprop_affinity if affprop_affinity is not None else self.affprop_affinity
        self.affprop_verbose = affprop_verbose if affprop_verbose is not None else self.affprop_verbose

        # Agglomerative Clustering
        self.agcluster_n_clusters = agcluster_n_clusters if agcluster_n_clusters is not None else self.agcluster_n_clusters
        self.agcluster_metric = agcluster_metric if agcluster_metric is not None else self.agcluster_metric
        self.agcluster_memory = agcluster_memory if agcluster_memory is not None else self.agcluster_memory
        self.agcluster_connectivity = agcluster_connectivity if agcluster_connectivity is not None else self.agcluster_connectivity
        self.agcluster_compute_full_tree = agcluster_compute_full_tree if agcluster_compute_full_tree is not None else self.agcluster_compute_full_tree
        self.agcluster_linkage = agcluster_linkage if agcluster_linkage is not None else self.agcluster_linkage
        self.agcluster_distance_threshold = agcluster_distance_threshold if agcluster_distance_threshold is not None else self.agcluster_distance_threshold
        self.agcluster_compute_distances = agcluster_compute_distances if agcluster_compute_distances is not None else self.agcluster_compute_distances

        # Feature Agglomeration
        self.featagg_n_clusters = featagg_n_clusters if featagg_n_clusters is not None else self.featagg_n_clusters
        self.featagg_metric = featagg_metric if featagg_metric is not None else self.featagg_metric
        self.featagg_memory = featagg_memory if featagg_memory is not None else self.featagg_memory
        self.featagg_connectivity = featagg_connectivity if featagg_connectivity is not None else self.featagg_connectivity
        self.featagg_compute_full_tree = featagg_compute_full_tree if featagg_compute_full_tree is not None else self.featagg_compute_full_tree
        self.featagg_linkage = featagg_linkage if featagg_linkage is not None else self.featagg_linkage
        self.featagg_pooling_func = featagg_pooling_func if featagg_pooling_func is not None else self.featagg_pooling_func
        self.featagg_distance_threshold = featagg_distance_threshold if featagg_distance_threshold is not None else self.featagg_distance_threshold
        self.featagg_compute_distances = featagg_compute_distances if featagg_compute_distances is not None else self.featagg_compute_distances

        # Birch
        self.birch_threshold = birch_threshold if birch_threshold is not None else self.birch_threshold
        self.birch_branching_factor = birch_branching_factor if birch_branching_factor is not None else self.birch_branching_factor
        self.birch_n_clusters = birch_n_clusters if birch_n_clusters is not None else self.birch_n_clusters
        self.birch_compute_labels = birch_compute_labels if birch_compute_labels is not None else self.birch_compute_labels
        self.birch_copy = birch_copy if birch_copy is not None else self.birch_copy

        # Mean Shift
        self.meanshift_bandwidth = meanshift_bandwidth if meanshift_bandwidth is not None else self.meanshift_bandwidth
        self.meanshift_seeds = meanshift_seeds if meanshift_seeds is not None else self.meanshift_seeds
        self.meanshift_bin_seeding = meanshift_bin_seeding if meanshift_bin_seeding is not None else self.meanshift_bin_seeding
        self.meanshift_min_bin_freq = meanshift_min_bin_freq if meanshift_min_bin_freq is not None else self.meanshift_min_bin_freq
        self.meanshift_cluster_all = meanshift_cluster_all if meanshift_cluster_all is not None else self.meanshift_cluster_all
        self.meanshift_n_jobs = meanshift_n_jobs if meanshift_n_jobs is not None else self.meanshift_n_jobs
        self.meanshift_max_iter = meanshift_max_iter if meanshift_max_iter is not None else self.meanshift_max_iter

        # OPTICS
        self.optics_min_samples = optics_min_samples if optics_min_samples is not None else self.optics_min_samples
        self.optics_max_eps = optics_max_eps if optics_max_eps is not None else self.optics_max_eps
        self.optics_metric = optics_metric if optics_metric is not None else self.optics_metric
        self.optics_p = optics_p if optics_p is not None else self.optics_p
        self.optics_metric_params = optics_metric_params if optics_metric_params is not None else self.optics_metric_params
        self.optics_cluster_method = optics_cluster_method if optics_cluster_method is not None else self.optics_cluster_method
        self.optics_eps = optics_eps if optics_eps is not None else self.optics_eps
        self.optics_xi = optics_xi if optics_xi is not None else self.optics_xi
        self.optics_predecessor_correction = optics_predecessor_correction if optics_predecessor_correction is not None else self.optics_predecessor_correction
        self.optics_min_cluster_size = optics_min_cluster_size if optics_min_cluster_size is not None else self.optics_min_cluster_size
        self.optics_algorithm = optics_algorithm if optics_algorithm is not None else self.optics_algorithm
        self.optics_leaf_size = optics_leaf_size if optics_leaf_size is not None else self.optics_leaf_size
        self.optics_memory = optics_memory if optics_memory is not None else self.optics_memory
        self.optics_n_jobs = optics_n_jobs if optics_n_jobs is not None else self.optics_n_jobs

        # Spectral Clustering
        self.spcluster_n_clusters = spcluster_n_clusters if spcluster_n_clusters is not None else self.spcluster_n_clusters
        self.spcluster_eigen_solver = spcluster_eigen_solver if spcluster_eigen_solver is not None else self.spcluster_eigen_solver
        self.spcluster_n_components = spcluster_n_components if spcluster_n_components is not None else self.spcluster_n_components
        self.spcluster_n_init = spcluster_n_init if spcluster_n_init is not None else self.spcluster_n_init
        self.spcluster_gamma = spcluster_gamma if spcluster_gamma is not None else self.spcluster_gamma
        self.spcluster_affinity = spcluster_affinity if spcluster_affinity is not None else self.spcluster_affinity
        self.spcluster_n_neighbors = spcluster_n_neighbors if spcluster_n_neighbors is not None else self.spcluster_n_neighbors
        self.spcluster_eigen_tol = spcluster_eigen_tol if spcluster_eigen_tol is not None else self.spcluster_eigen_tol
        self.spcluster_assign_labels = spcluster_assign_labels if spcluster_assign_labels is not None else self.spcluster_assign_labels
        self.spcluster_degree = spcluster_degree if spcluster_degree is not None else self.spcluster_degree
        self.spcluster_coef0 = spcluster_coef0 if spcluster_coef0 is not None else self.spcluster_coef0
        self.spcluster_kernel_params = spcluster_kernel_params if spcluster_kernel_params is not None else self.spcluster_kernel_params
        self.spcluster_n_jobs = spcluster_n_jobs if spcluster_n_jobs is not None else self.spcluster_n_jobs
        self.spcluster_verbose = spcluster_verbose if spcluster_verbose is not None else self.spcluster_verbose

        return self