from enum import Enum
import matplotlib.pyplot as plt

class Plot(Enum):
  default = 0
  indistinguishable = 1
  cluster = 3

class Graphic:
  def __init__(self):
    self.__first_color      = "blue"
    self.__second_color     = "orange"
    self.__third_color      = "purple"
    self.__default_color    = "black"
    self.__divisory_color   = "black"
    self.p_type = Plot.default
    self.plot_type = {
      Plot.default:             self.indistinguishable_clusters_2D,
      Plot.indistinguishable:   self.indistinguishable_clusters_2D,
      Plot.cluster:             self.distinguishable_clusters_2D,
    }

  def select(self, t: Plot) -> 'Graphic':
    self.p_type = t
    return self

  def points_2D(self, **args) -> None:    
    # switch case
    if self.p_type in self.plot_type:
      self.plot_type[self.p_type](**args)

  def distinguishable_clusters_2D(self, **args) -> None:
    X = args['X']
    y = args['y']
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', marker='.')
    plt.title("Scatter Plot of the clusters in the Dataset")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

  def indistinguishable_clusters_2D(self, **args) -> None:
    X = args['X']
    y = None
    if len(X.shape) < 2:
      y = args['y']
    else: 
      y = args['X'][:, 1]
      X = X[:,0]
    plt.scatter(X, y, color = self.__default_color, marker='.')
    plt.title("Scatter Plot of samples in the Dataset")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()