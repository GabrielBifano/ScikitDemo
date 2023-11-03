from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.datasets import make_s_curve
from graphic import Graphic, Plot
import random

class Generator:
    def __init__(self, samp=300, feat=2, clusters=6, noise=0.05, factor=0.5):
        self.g = Graphic()
        self.samp = samp
        self.feat = feat
        self.clusters = clusters
        self.noise = noise
        self.factor = factor
        self.rand = lambda : random.randint(0, 0xFFFFFFFF - 1)

    def make_blobs(self, to_print: bool=False) -> tuple:
        X, y = make_blobs (
            n_samples=self.samp,
            n_features=self.feat,
            centers=self.clusters,
            random_state=self.rand()
        )
        if to_print:
            self.g.select(Plot.cluster).points_2D(X=X, y=y)
            self.g.select(Plot.indistinguishable).points_2D(X=X)
        return X, y


    def make_moons(self, to_print: bool=False) -> tuple:
        X, y = make_moons(
            n_samples=self.samp,
            noise=self.noise,
            random_state=self.rand()
        )
        if to_print:
            self.g.select(Plot.cluster).points_2D(X=X, y=y)
            self.g.select(Plot.indistinguishable).points_2D(X=X)
        return X, y


    def make_circles(self, to_print: bool=False) -> tuple:
        X, y = make_circles(
            n_samples=self.samp,
            factor=self.factor,
            noise=self.noise,
            random_state=self.rand()
        )
        if to_print:
            self.g.select(Plot.cluster).points_2D(X=X, y=y)
            self.g.select(Plot.indistinguishable).points_2D(X=X)
        return X, y


    def make_s_curve(self, to_print: bool=False) -> tuple:
        X, y = make_s_curve(
            n_samples=self.samp,
            noise=self.noise,
            random_state=self.rand()
        )
        if to_print:
            self.g.select(Plot.indistinguishable).points_2D(X=X[:,0], y=y)
        return X, y

    # TODO implement this function here and in the notebook
    def make_hard_config(self, n:int, to_print: bool=False) -> tuple:
        pass