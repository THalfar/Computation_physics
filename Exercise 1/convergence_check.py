import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
# - or, e.g., fig = plt.figure(figsize=(width, height))
# - so-called golden ratio: width=height*(sqrt(5.0)+1.0)/2.0
ax = fig.add_subplot(111)
x = np.linspace(0,np.pi/2,100)
f = np.sin(x)
g = np.exp(-x)
# plot and add label if legend desired
ax.plot(x,f,label=r'$f(x)=\sin(x)$')
ax.plot(x,g,label=r'$f(x)=\exp(-x)$')
# plot legend
ax.legend(loc=0)
# set axes labels and limits
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'f(x)')
ax.set_xlim(x.min(), x.max())
fig.tight_layout(pad=1)
# save figure as pdf with 200dpi resolution
#fig.savefig(’testfile.pdf’,dpi=200)
plt.show()


