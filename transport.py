"""
This code will likely be superseded by dwaq_transport.py

Use a combination of resolved hydrodynamics, aggregated hydrodynamics,
and observed tracer distributions (salinity, temp) to infer box-model
scale transport in SF Bay.  
"""

import os
import matplotlib.pyplot as plt

from stompy import utils
from stompy.grid import unstructured_grid
from stompy.spatial import field

import dem_tools

## 

dem=field.GdalGrid('/opt/data/bathy_dwr/joined-40m.tif')

##

g=unstructured_grid.UnstructuredGrid.from_shp('boxes-v02.shp')

##

# About 10 seconds
dem_tools.add_msl_geometry(g,dem,msl=1.0)

##

# The original grid has some spatial relevance, but we need an even
# simpler topological representation, which is essentially the dual.
gd=g.create_dual()

##         


# Need a method to take a scalar field at time t0, and transport it
# to time t1
# Aim for general exchange matrix.

# is this easier if S is "homogeneous", i.e. always carries around a unit
# value? doesn't really matter, maybe makes the code a bit cleaner not to?

# S(t+1) = M(dt)*S(t) + dt*J(t) + diric(t)

N=gd.Nnodes()

class BC(object):
    pass

class DirichletPoint(BC):
    def __init__(self,gd,x,fn_t):
        self.gd=gd
        self.n=self.gd.select_nodes_nearest(x)
        self.x=x
        self.fn=fn_t
    def apply(self,M,J,D,t):
        M[self.n,:] = 0.0 # no exchange
        J[self.n]=0.0 # not a flux
        D[self.n]=self.fn(t)

class FluxPoint(BC):
    def __init__(self,gd,x,fn_t):
        self.gd=gd
        self.n=self.gd.select_nodes_nearest(x)
        self.x=x
        self.fn=fn_t
    def apply(self,M,J,D,t):
        # fn(t) => m3/s
        J[self.n] += self.fn(t) 
        
ocean=lambda t: 34.0
bcs= [DirichletPoint(gd=gd,x=(500024, 4197981),fn_t=ocean),
      DirichletPoint(gd=gd,x=(508239, 4188818),fn_t=ocean),
      DirichletPoint(gd=gd,x=(512505, 4177600),fn_t=ocean),
      DirichletPoint(gd=gd,x=(515507, 4166225),fn_t=ocean),
      DirichletPoint(gd=gd,x=(525144, 4154218),fn_t=ocean)]

t=0

##

M=np.eye(N)# zeros((N,N), np.float64)
J=np.zeros(N,np.float64)
D=np.zeros(N,np.float64)

# Basic mixing:
K=0.2
dt=0.5
for j in gd.valid_edge_iter():
    n1,n2=gd.edges['nodes'][j]
    M[n1,n1] -= dt*K
    M[n2,n2] -= dt*K
    M[n2,n1] += dt*K
    M[n1,n2] += dt*K

for bc in bcs:
    bc.apply(M,J,D,t)

##


# basic testing

salt0=33*np.ones(gd.Nnodes())
salt=np.dot(M,salt0) + dt*J + D

##

plt.figure(1).clf()
fig,axs=plt.subplots(1,2,num=1)

ccoll0=g.plot_cells(values=salt0,ax=axs[0])

ccoll1=g.plot_cells(values=salt2,ax=axs[1])

for it in range(10):
    salt=np.dot(M,salt) + dt*J + D 
    ccoll1.set_array(salt)
    plt.draw()
    plt.pause(0.01)



##

# How can this be used to back out the transport model?

#  - using the existing aggregation code, add in depth averaging, maybe a longer
#    low-pass period.

#  - purely by tuning.
#    assume that only edges in the dual can have
#    coefficients.  need to solve for both advection and exchange.
#    apply the model to the salt field.  for each dual edge, there
#    are two degrees of freedom.  The advection terms must add to
#    continuity per-volume, but there is no additional constraint on the exchange
#    terms.  Everybody starts with a prior of zero advection and little
#    exchange.  There is probably a way to represent advection terms directly
#    by a potential field, to bake in the continuity constraint.
#    that's 386 degrees of freedom.

#  - extracting resolved model output, fluxes, tune dispersion.
