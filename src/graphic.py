import matplotlib.pyplot as plt
import pandas as pd

class Graphic:
  def __init__(self):
    self.__first_color      = "red"
    self.__second_color     = "orange"
    self.__third_color      = "purple"
    self.__default_color    = "black"

  def sample(self, X, y=[None]) -> None:
    if type(X) == pd.DataFrame: X = X.values
    if y.all() == None: y = [0]*len(X)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', marker='.')
    plt.title("Scatter plot of samples in dataset")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()