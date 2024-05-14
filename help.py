import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Plt:
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def fig_size(self,x_size,y_size):
        self.x_size = x_size
        self.y_size = y_size 
        return self
    def plt_show(self,label = None ):
        plt.figure(figsize=(self.x_size, self.y_size))
        if label is not None:
            plt.title('y = '+ label)
        plt.plot(self.x, self.y)
        plt.show()
