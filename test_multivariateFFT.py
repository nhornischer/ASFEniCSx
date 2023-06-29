import numpy as np
import math
from time import sleep

import matplotlib.pyplot as plt


"""
Periodic Function
"""

L = 2*np.pi

x=np.linspace(0,L,100, endpoint=False)

X,Y = np.meshgrid(x,np.copy(x))

f = np.cos(X) + np.sin(Y)

f = np.sin(X) * np.cos(Y)

df_analytical = [-np.sin(X[:]), np.cos(Y[:])]

df_analytical = [np.cos(X[:]) * np.cos(Y[:]), -np.sin(X[:]) * np.sin(Y[:])]

F = np.fft.fftn(f)

f_back = np.fft.ifftn(F)

frequencies_x = frequencies_y = 2*np.pi*np.fft.fftfreq(len(f),x[1]-x[0])

kx , ky = np.meshgrid(frequencies_x,frequencies_y)

dFSx = 1j*kx*F

dFSy = 1j*ky*F

df = [np.fft.ifftn(dFSx), np.fft.ifftn(dFSy)]

print("Error in transform", np.linalg.norm(f_back - f)/np.linalg.norm(f))

print("Error in dx", np.linalg.norm(np.real(df[0]) - df_analytical[0])/np.linalg.norm(df_analytical[0]))
print("Error in dy", np.linalg.norm(np.real(df[1]) - df_analytical[1])/np.linalg.norm(df_analytical[1]))

plt.figure("Periodic Function")
plt.subplot(2,2,1)
plt.imshow(df_analytical[0])
plt.title('df/dx')
plt.subplot(2,2,2)
plt.imshow(df_analytical[1])
plt.title('df/dy')
plt.subplot(2,2,3)
plt.imshow(np.real(df[0]))
plt.title("F^-1(dF/dx)")
plt.subplot(2,2,4)
plt.imshow(np.real(df[1]))
plt.title("F^-1(dF/dy)")

plt.tight_layout()

"""
Non-Periodic Function
"""

x = np.linspace(0,L,100,endpoint=False)

X,Y = np.meshgrid(x,np.copy(x))

f = np.exp(0.7 *X[:] + 0.3*Y[:])

df_analytical = [0.7*f, 0.3*f]

F = np.fft.fftn(f)

f_back = np.fft.ifftn(F)

frequencies_x = frequencies_y = 2*np.pi*np.fft.fftfreq(len(f),x[1]-x[0])

kx , ky = np.meshgrid(frequencies_x,frequencies_y)

dFSx = 1j*kx*F

dFSy = 1j*ky*F

df = [np.fft.ifftn(dFSx), np.fft.ifftn(dFSy)]

print("Error in transform", np.linalg.norm(f_back - f)/np.linalg.norm(f))

print("Error in dx", np.linalg.norm(np.real(df[0]) - df_analytical[0])/np.linalg.norm(df_analytical[0]))
print("Error in dy", np.linalg.norm(np.real(df[1]) - df_analytical[1])/np.linalg.norm(df_analytical[1]))

plt.figure("Non-Periodic Function")
plt.subplot(2,2,1)
plt.imshow(df_analytical[0])
plt.title('df/dx')
plt.subplot(2,2,2)
plt.imshow(df_analytical[1])
plt.title('df/dy')
plt.subplot(2,2,3)
plt.imshow(np.real(df[0]))
plt.title("F^-1(dF/dx)")
plt.subplot(2,2,4)
plt.imshow(np.real(df[1]))
plt.title("F^-1(dF/dy)")

plt.tight_layout()

plt.show()