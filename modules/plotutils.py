import numpy as np
import matplotlib.pyplot as plt

class PlotUtils:

    def __init__(self):
        pass

    def corr_func(x,y,**kwargs):
        r = np.corrcoef(x,y)[0][1]
        ax = plt.gca()
        ax.annotate("r:{:.2f}".format(r),xy=(.2,.8),xycoords=ax.transAxes,size=20)
