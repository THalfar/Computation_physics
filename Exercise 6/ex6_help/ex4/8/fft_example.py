import numpy as np
import matplotlib.pyplot as plt


# ADD CODE: read in the data here 


dt = t[1]-t[0]
N=len(t)

# Fourier coefficients from numpy fft normalized by multiplication of dt
F = np.fft.fft(f)*dt

# frequencies from numpy fftfreq
freq = np.fft.fftfreq(len(F),d=dt)

# inverse Fourier with numpy ifft (normalization removed with division by dt)
iF = np.fft.ifft(F/dt)

# positive frequencies are given as
# freq[:N//2] from above or freq = np.linspace(0, 1.0/dt/2, N//2)

fig, ax = plt.subplots()
# plot over positive frequencies the Fourier transform
ax.plot(freq[:N//2], np.abs(F[:N//2]))
ax.xlabel(r'$f$ (Hz)')
ax.ylabel(r'$F(\omega/2\pi)$')
 
# plot the "signal" and test the inverse transform
fig, ax = plt.subplots()
ax.plot(t, f,t,iF.real,'r--')
ax.set_xlabel(r'$t$ (s)')
ax.set_ylabel(r'$f(t)$')

plt.show()
