#!/usr/bin/env python
# coding: utf-8

# use esc + m to switch from code to markdown cell
# # 1D diffusion model

# Here we develop a one-dimensional model of diffusion. It assumes a constant diffusivity, uses a regular grid, and has fixed boundary conditions (these are simplifying assumptions)

# The diffusion equation:
# $$ \frac{\partial C}{\partial t} = D\frac{\partial^2 C}{\partial x^2} $$
# The discretized version of the diffusion equation that we'll solve with our model:
# $$ C^{t+1}_x = C^t_x + {D \Delta t \over \Delta x^2} (C^t_{x+1} - 2C^t_x + C^t_{x-1}) $$
# This is the explicit FTCS scheme as described in Slingerland and Kump (2011). (Or see Wikipedia.)

# - Jupyter integrates LaTeX formatting
# - FTCS = forward in time, centered in space

# Two libraries used to create this: Numpy for arrays, Matplotlib for plotting. These aren't part of base Python.

# At this point, switched from default kernel to CSDMS kernel for the preconfigured env.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


# Set two fixed model paramenters: diffusivity and size of model domain

# In[ ]:


D = 100 # diffusivity
Lx = 300 # domain size


# Then set up the model grid using a numpy array

# In[ ]:


dx = 0.5 # resolution of the x axis
x = np.arange(start=0, stop=Lx, step=dx) # use the arange function from numpy to initialize model grid with an array
nx = len(x) # this is the number of model steps


# In[ ]:


whos # prints variable window, similar to Rstudio viewer


# In[ ]:


x[0]  # prints first element of the array


# In[ ]:


x[nx] # trying to get the 600th index, will throw an error


# In[ ]:


x[nx-1]


# In[ ]:


x[-1]  # will give same result as prev cell


# In[ ]:


x[0:5]  # gives first 5 values of array, end of colon is not inclusive


# Set the initial concentration profile for the model. The concentration `C` is a step function with a high value on the left and a low value on the right. There's a step at the center of the domain.

# In[ ]:


C = np.zeros_like(x)  # zeros_like makes an array that is like another array, in this case, x
C_left = 500 # spatial constraint, left side
C_right = 0  # spatial constraint, right side
C[x <= Lx//2] = C_left  # C for values of x that are less than the domain length divided by two 
C[x > Lx//2] = C_right  # C for values of x that are more than the domain length divided by two
# note: double slash indicates integer division


# plotting the initial profile of the step function

# In[ ]:


plt.figure()  # initiates plotting process with matplotlib
plt.plot(x, C, "r")
plt.xlabel("x")
plt.ylabel("C")
plt.title("Initial concentration profile")


# Now set up the time component of the model:
# - start time
# - number of time steps
# - calculate a stable time step using stability criterion

# In[ ]:


time  = 0
nt = 5000  # number of time steps
dt = (0.5 * dx**2)w / D  # double asterisks indicates exponent


# In[ ]:


dt


# Loop over the time steps of the model, solving the diffusion equation using the FTCS explicit scheme described above.
# - Boundary conditions are fixed so reset them at each time step

# In[ ]:


for t in range(0, nt):
    C += D * dt/dx**2 * (np.roll(C, -1) - 2*C + np.roll(C,1))
    C[0] = C_left
    C[-1] = C_right

# this is the discretization equation from earlier
# += will add whatever to prev value of the thing
# the numpy roll function shifts the array as indicated by the second arg


# In[ ]:


# plot the result:

plt.figure()
plt.plot(x, C, "b")
plt.xlabel("x")
plt.ylabel("C")
plt.title("Final concentration profile")  #diffusion!


# In[ ]:




