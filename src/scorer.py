import pandas as pd
from graphic import Graphic
from ansi import ANSI

from sklearn.metrics import rand_score
from sklearn.metrics import v_measure_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import mutual_info_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import completeness_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics.cluster import contingency_matrix

class Scorer:
    
    def __init__(self):
        self.g = Graphic()
        self.print_cli = True
        self.decimals = 5
    
    def config(self, print_cli=None, decimals=None):
        self.print_cli = print_cli if print_cli is not None else self.print_cli
        self.decimals = decimals if decimals is not None else self.decimals
        return self

    #TODO implement this method
    def contingency_matrix(self, *args):
        pass

    def external_metrics(self, truth, predicted) -> dict:
        metrics_dict = {
            'rand_score': rand_score(truth, predicted),
            'v_measure_score': v_measure_score(truth, predicted),
            'mutual_info_score': mutual_info_score(truth, predicted),
            'homogeneity_score': homogeneity_score(truth, predicted),
            'completeness_score': completeness_score(truth, predicted),
            'adjusted_rand_score': adjusted_rand_score(truth, predicted),
            'fowlkes_mallows_score': fowlkes_mallows_score(truth, predicted),
        }
        if self.print_cli:
            a = ANSI()
            print(
                f'Model performance {a.b}{a.green}external metrics:{a.res}'
                f'{a.cyan}'
                f'\n\t* rand_score:             {metrics_dict["rand_score"]:.5}'
                f'\n\t* adjusted_rand_score:    {metrics_dict["adjusted_rand_score"]:.5}'
                f'\n\t* v_measure_score:        {metrics_dict["v_measure_score"]:.5}'
                f'\n\t* mutual_info_score:      {metrics_dict["mutual_info_score"]:.5}'
                f'\n\t* homogeneity_score:      {metrics_dict["homogeneity_score"]:.5}'
                f'\n\t* completeness_score:     {metrics_dict["completeness_score"]:.5}'
                f'\n\t* fowlkes_mallows_score:  {metrics_dict["fowlkes_mallows_score"]:.5}'
                f'{a.res}'
            )
        return metrics_dict

    def internal_metrics(self, data: pd.DataFrame, labels) -> dict:
        metrics_dict = {
            'silhouette_score': silhouette_score(data, labels),
            'davies_bouldin_score': davies_bouldin_score(data, labels),
            'calinski_harabasz': calinski_harabasz_score(data, labels),
        }
        if self.print_cli:
            a = ANSI()
            print(
                f'Model performance {a.b}{a.green}internal metrics:{a.res}'
                f'{a.cyan}'
                f'\n\t* silhouette_score:       {metrics_dict["silhouette_score"]:.5}'
                f'\n\t* davies_bouldin_score:   {metrics_dict["davies_bouldin_score"]:.5}'
                f'\n\t* calinski_harabasz:      {metrics_dict["calinski_harabasz"]:.5}'
                f'{a.res}'
            )
        return metrics_dict