import matplotlib.pyplot as plt
import pandas as pd

class Graphic:
  def __init__(self):
    self.__first_color      = "red"
    self.__second_color     = "orange"
    self.__third_color      = "purple"
    self.__default_color    = "black"

  def sample_many(self, dataframes, nrows: int, ncols: int, figsize: tuple=(15, 10)):
    prop = ncols/nrows
    a_prop = nrows/ncols
    
    plt.style.use('dark_background')
    _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    axes_it = iter(axes.flatten())
    
    for key in dataframes:
      X = dataframes[key].values
      y = dataframes[key]['labels']
      ax = next(axes_it)
      
      ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', marker='.', edgecolors='black')  
      ax.grid(linewidth=0.5, linestyle='--', alpha=0.2)
      ax.set_xticks([])
      ax.set_yticks([])
      ax.set_title(key, fontsize=8)
    
    plt.show()

  def sample(self, X, y=[None]) -> None:
    if type(X) == pd.DataFrame: X = X.values
    if y.all() == None: y = [0]*len(X)
    
    plt.style.use('dark_background')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', marker='.', edgecolors='black')
    plt.grid(linewidth=0.5, linestyle='--', alpha=0.2)
    plt.title("Scatter plot of samples in dataset")
    plt.show()