"""
Convergence tests for first derivative
"""

import numpy as np
import matplotlib.pyplot as plt
from num_calculus import first_derivative


def simple_plot(x,f,axis_txts,title_txt=None,legend_txt=None,save2file=False):
    fig = plt.figure()
    # - or, e.g., fig = plt.figure(figsize=(width, height))
    # - so-called golden ratio: width=height*(sqrt(5.0)+1.0)/2.0
    ax = fig.add_subplot(111)
    # plot and add label if legend desired
    ax.plot(x,f,label=legend_txt)
    if legend_txt!=None:
        # plot legend
        ax.legend(loc=0)
    # set axes labels and limits
    ax.set_xlabel(axis_txts[0])
    ax.set_ylabel(axis_txts[1])
    ax.set_xlim(x.min(), x.max())
    if title_txt!=None:
        ax.set_title(title_txt)
    fig.tight_layout(pad=1)
    if save2file:
        # save figure as pdf with 200dpi resolution
        fig.savefig('testfile.pdf',dpi=200)
    return ax,fig
    #plt.show()

def test_function(x):
    return np.sin(x)

def test_function_derivative(x):
    return np.cos(x)

def first_derivative_convergence():
    x = 0.3
    dx = np.linspace(0.001,0.5,10)
    df = []
    for i in range(len(dx)):
        df.append(abs(first_derivative(test_function,x,dx[i])-test_function_derivative(x)))
    df = np.array(df)
    simple_plot(dx,df,['$x$','$f(x)$'],None)
    plt.show()

def main():
    first_derivative_convergence()

if __name__=="__main__":
    main()
