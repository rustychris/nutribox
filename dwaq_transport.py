"""
Recreate a basic version of 2D transport from DWAQ output which has been filtered
aggregated, and depth-integrated.
"""

import os
import matplotlib.pyplot as plt

from stompy import utils
from stompy.utils import add_to

from stompy.grid import unstructured_grid
from stompy.spatial import field

import stompy.model.delft.waq_scenario as waq


## 

hydro=waq.HydroFiles('hydro-wy2013c_adj-agg_lp_2d/com-wy2013c_adj_agg_lp_2d.hyd')

##

def set_variables(obj,kws):
    """
    Free form setting of instance variables, but check to be sure they
    exist first in order to avoid misspellings
    """
    for k, v in kws.items():
        try:
            getattr(obj,k)
            setattr(obj, k, v)
        except AttributeError:
            raise Exception("Unknown keyword option: %s=%s"%(k, v))

class DwaqTransport(object):
    # must be an indexable, iterable object with values assumed to be in seconds.
    times=None
    theta=0.50
    
    def __init__(self,hydro,**kws):
        self.hydro=hydro
        self.scalars=[]
        set_variables(self,kws)

        #  Perform pre-processing related to the source Hydro.
        self.setup_grid()
        self.setup_bcs()
        
    def setup_grid(self):
        """
        Initialize spatial information
        """
        self.g=hydro.grid()
        # The original grid has some spatial relevance, but we need an even
        # simpler topological representation, which is essentially the dual.
        self.gd=g.create_dual()
        self.N=gd.Nnodes() # number of computational elements
        
    def setup_bcs(self):
        """
        Initialize details of the boundaries in the hydro
        """
        self.N_boundary=np.sum( self.hydro.pointers[:,0]<0)
        bnds=self.hydro.read_bnd()
        self.bnd_map={} # negative BC index => name
        for bnd in bnds:
            for bc_link in bnd[1]['link']:
                self.bnd_map[bc_link]=bnd[0]

    def add_scalar(self,name,**kws):
        """
        Register a scalar field to be transport
        """
        scalar=Scalar(name,transport=self,**kws)
        self.scalars.append(scalar)
        return scalar
        
    def boundary_C_zero(self):
        """
        Return a vector of zeros the correct shape for specifying boundary concentrations.
        """
        return np.zeros(self.N_boundary,np.float64)
    def initial_C_zero(self):
        """
        Return a vector of zeros the correct shape for specifying initial concentrations.
        """
        return np.zeros(self.N,np.float64)

    def initialize(self):
        """ Prepare initial conditions.  Call before beginning time stepping
        """
        self.t_idx=0
        self.t=self.times[self.t_idx]
            
        for scal in self.scalars:
            scal.initialize(t=self.t)

    def loop(self):
        """
        Iterate over all time steps defined in self.times,
        Assumes that current scalar state is consistent with model time
        self.t
        """
        while self.t_idx+1 < len(self.times):
            # Set exchange matrix entries based on DWAQ data
            t0=times[self.t_idx]
            t1=times[self.t_idx+1]
            dt_s=t1-t0

            # Write in terms of concentrations
            # M=np.eye(N,dtype=np.float64)
            M=np.zeros((self.N,self.N), dtype=np.float64)

            J=np.zeros(self.N,np.float64)
            D=np.zeros(self.N,np.float64)

            flows=self.hydro.flows(t0)
            # vol=hydro.volumes(t0)
            vol=self.hydro.volumes(t1)

            boundary_C_per_scalar=[scal.boundary_C(t1)
                                   for scal in self.scalars]
            J_per_scalar=[J.copy() for scal in self.scalars]
            
            for j,(seg_from,seg_to) in enumerate(self.hydro.pointers[:,:2]-1):
                if 1: # Advection, including boundaries
                    # upwind advective flux
                    if flows[j]>0:
                        seg_src=seg_from # may be a boundary!
                    else:
                        seg_src=seg_to

                    if seg_from>=0:
                        assert seg_src>=0
                        M[seg_from,seg_src] -= dt_s*flows[j] / vol[seg_from]

                    if seg_src>=0:
                        assert seg_to>=0
                        M[seg_to,seg_src]   += dt_s*flows[j] / vol[seg_to]
                    else:
                        bc_idx=(seg_src+1) # just undoing the -1 from above
                        # Probably there is a more efficient way to do this:
                        for scal_i,(J,boundary_C) in enumerate(zip(J_per_scalar,boundary_C_per_scalar)):
                            # print("Boundary flux j=%d bc_idx=%d seg_src=%d name=%s"%(j,bc_idx,seg_src+1,bnd_map[bc_idx]))
                            J[seg_to] += boundary_C[-bc_idx-1] * flows[j] / vol[seg_to]
                            
                if 1: # Exchange
                    if seg_from>=0:
                        # Basic mixing:
                        # HERE - adjust to take into account length scales, areas
                        # This is a stand-in formulation for testing
                        K=0.5 # fraction exchange per day of the smaller volume.
                        dV=dt_s/86400. * K * min(vol[seg_from],vol[seg_to])

                        M[seg_from,seg_from] -= dV / vol[seg_from]
                        M[seg_to,seg_to]     -= dV / vol[seg_to]
                        M[seg_from,seg_to] += dV / vol[seg_from]
                        M[seg_to,seg_from] += dV / vol[seg_to]

            I=np.eye(M.shape[0])

            for scal,J in zip(self.scalars,J_per_scalar):
                state0=scal.state
                if theta==0: # explicit
                    scal.state[:]=state0 + np.dot(M,state0) + dt_s*J + D
                elif theta==1: # implicit
                    # scal1=scal0 + np.dot(M,scal1) + dt_s*J + D
                    # scal1-np.dot(M,scal1) = scal0 + dt_s*J + D
                    rhs=state0+dt_s*J+D
                    scal.state[:]=np.linalg.solve(I-M, rhs)
                else:
                    # y1 = y0 + dt*(theta*f(t1,y1) + (1-theta)*f(t0,y0))
                    # y1-theta*dt*f(t1,y1) = y0 + (1-theta)*dt*f(t0,y0)
                    # explicit terms, and explicit portion of M
                    rhs=state0+dt_s*J+D+np.dot((1-theta)*M,state0)
                    scal.state[:]=np.linalg.solve(I-theta*M,rhs)

                scal.record_state(t1)

            self.t_idx+=1
    
class Scalar(object):
    """
    Track state, history, boundary conditions for a scalar
    """
    boundary_C_fn=lambda t,x:x
    initial_C_fn=lambda t,x:x
    state=None
    
    def __init__(self,name,transport,**kws):
        self.history=[]
        self.name=name
        self.transport=transport

        set_variables(self,kws)
        
    def boundary_C(self,t):
        return self.boundary_C_fn(t,self.transport.boundary_C_zero())
    def initial_C(self,t):
        return self.initial_C_fn(t,self.transport.initial_C_zero())

    def initialize(self,t):
        """
        This is called by Transport after the scalar is fully configured, and just prior
        to time stepping.  This is where initial conditions are used to populate model state,
        and that initial state is recorded in the history
        """
        self.state=self.initial_C(t)
        self.record_state(t)
        
    def record_state(self,t):
        self.history.append( (t,self.state.copy()) )
    
def zero_vec(t,x):
    x[:]=0
    return x
def unit_vec(t,x):
    x[:]=1.0
    return x
        
transport=DwaqTransport(hydro,
                        times=hydro.t_secs[1000:1000+125*48:2*48])

salt=transport.add_scalar("salinity",initial_C_fn=zero_vec)

@utils.add_to(salt)
def boundary_C(self,t):
    x=self.transport.boundary_C_zero()
    x[:]=0.0 # freshwater sources
    for bc_elt in self.transport.bnd_map:
        bc_idx=-bc_elt-1
        name=bnd_map[bc_elt]
        if 'Sea' in name:
            x[bc_idx]=34
    return x
    
cont=transport.add_scalar("continuity",
                          boundary_C_fn=unit_vec,
                          initial_C_fn=unit_vec)

transport.initialize()


transport.loop()

##

plt.figure(1).clf()
fig,axs=plt.subplots(1,2,num=1)

ccoll0=g.plot_cells(values=salt.history[0][1],ax=axs[0],cmap='jet')

ccoll1=g.plot_cells(values=salt.history[-1][1],ax=axs[1],cmap='jet')

plt.colorbar(ccoll0,ax=axs[0])
plt.colorbar(ccoll1,ax=axs[1])


##

salt=hydro.parameters()['salinity']

plt.clf()
g.plot_cells(values=salt.evaluate(t=t).data)

## 

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

