"""
Recreate a basic version of 2D transport from DWAQ output which has been filtered
aggregated, and depth-integrated.
"""

import os
import matplotlib.pyplot as plt
from matplotlib import collections
import numpy as np 

from stompy import utils
from stompy.utils import add_to

from stompy.grid import unstructured_grid
from stompy.spatial import field, wkb2shp

import stompy.model.delft.waq_scenario as waq
from stompy.model.delft import dfm_grid

import logging
logger=logging.getLogger('transport')

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
        self.g=self.hydro.grid()
        # The original grid has some spatial relevance, but we need an even
        # simpler topological representation, which is essentially the dual.
        self.gd=self.g.create_dual(center='centroid')
        self.N=self.gd.Nnodes() # number of computational elements
        
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

    def flows(self,t0,t1):
        """
        Return the volumetric flow rates (m3/s) for each exchange which
        represent the transport between time t0 and t1.
        per DWAQ file formatting, hydro.flows(t0) represents the fluxes
        integrated in time between t0 and t1, consistent with the change
        in volumes hydro.volume(t1) - hydro.volume(t0).
        
        This handles averaging fluxes when the hydro time step is a fraction
        of our time step.
        """
        t_secs=self.hydro.t_secs
        t0_i,t1_i=self.hydro.t_sec_to_index([t0,t1])
        flows=np.zeros(self.hydro.n_exch,'f4')

        for t_i in range(t0_i,t1_i):
            flow_i=self.hydro.flows(t_secs[t_i],memmap=True)
            flows+=flow_i
        flows/=t1_i-t0_i # average
        return flows
    
    def loop(self):
        """
        Iterate over all time steps defined in self.times,
        Assumes that current scalar state is consistent with model time
        self.t
        """
        while self.t_idx+1 < len(self.times):
            # Set exchange matrix entries based on DWAQ data
            t0=self.times[self.t_idx]
            t1=self.times[self.t_idx+1]
            dt_s=t1-t0

            # Formulated in terms of concentrations
            M=np.zeros((self.N,self.N), dtype=np.float64)

            J=np.zeros(self.N,np.float64)

            flows=self.flows(t0,t1)
            vol0=self.hydro.volumes(t0)
            vol1=self.hydro.volumes(t1)

            boundary_C_per_scalar=[scal.boundary_C(t1)
                                   for scal in self.scalars]
            J_per_scalar=[J.copy() for scal in self.scalars]


            for J,scal in zip(J_per_scalar,self.scalars):
                boundary_J=scal.boundary_J(t1)
                if boundary_J is not None:
                    J[:] += boundary_J[:] / vol1[:]
            
            for j,(seg_from,seg_to) in enumerate(self.hydro.pointers[:,:2]-1):
                # This is an assumption on the pointers -- boundaries are always
                # seg_from
                assert seg_to>=0
                    
                if 1: # Advection, including boundaries
                    # upwind advective flux
                    # The upwind segment (or boundary) is used to determine the concentration
                    if flows[j]>0:
                        seg_upwind=seg_from # may be a boundary!
                    else:
                        seg_upwind=seg_to

                    if seg_from>=0: # internal exchange
                        assert seg_upwind>=0
                        M[seg_from,seg_upwind] -= dt_s*flows[j] / vol1[seg_from]
                    else:
                        pass # no entry for updating a boundary

                    # this is debatable in the case of a boundary with seg_from<0
                    # in that case, flow is out of the domain.  Do we advect out
                    # the upwind concentration, or the prescribed concentration?
                    # depends.  
                    always_bc_concentration=False
                    
                    if seg_from<0 and (always_bc_concentration or seg_upwind<0):
                        # It's a boundary exchange, and we should use the boundary
                        # concentration, either because it's the upwind element, or
                        # because we've decided to always use boundary concentrations
                        bc_idx=(seg_from+1) # just undoing the -1 from above
                        # there is a more efficient way to do this:
                        for scal_i,(J,boundary_C) in enumerate(zip(J_per_scalar,
                                                                   boundary_C_per_scalar)):
                            if boundary_C is not None:
                                # print("Boundary flux j=%d bc_idx=%d seg_upwind=%d name=%s"
                                #        %(j,bc_idx,seg_upwind+1,bnd_map[bc_idx]))
                                J[seg_to] += boundary_C[-bc_idx-1] * flows[j] / vol1[seg_to]
                    else:
                        M[seg_to,seg_upwind]   += dt_s*flows[j] / vol1[seg_to]
                            
                if 1: # Exchange 
                    if seg_from>=0:
                        # Basic mixing:
                        # HERE - adjust to take into account length scales, areas
                        # This is a stand-in formulation for testing
                        K=0.5 # fraction exchange per day of the smaller volume.
                        dV=dt_s/86400. * K * min(vol0[seg_from],vol0[seg_to])

                        M[seg_from,seg_from] -= dV / vol1[seg_from]
                        M[seg_to,seg_to]     -= dV / vol1[seg_to]
                        M[seg_from,seg_to] += dV / vol1[seg_from]
                        M[seg_to,seg_from] += dV / vol1[seg_to]

            I=np.eye(M.shape[0])
            Vratio=vol0/vol1

            for scal,J in zip(self.scalars,J_per_scalar):
                state0=scal.state
                # Vratio here is important and captures the dilution/concentration
                # of the existing scalar mass due to change in volume in the absence
                # of fluxes.
                rhs=state0*Vratio + dt_s*J
                if self.theta==0: # explicit
                    # is this the fix-- Vratio here?
                    scal.state[:]= rhs + np.dot(M,state0)
                elif self.theta==1: # implicit
                    # scal1=(scal0*Vratio + dt_s*J) + np.dot(M,scal1) 
                    # scal1-np.dot(M,scal1) = scal0 + dt_s*J
                    scal.state[:]=np.linalg.solve(I-M, rhs)
                else:
                    # y1 = y0 + dt*(theta*f(t1,y1) + (1-theta)*f(t0,y0))
                    # y1-theta*dt*f(t1,y1) = y0 + (1-theta)*dt*f(t0,y0)
                    # explicit terms, and explicit portion of M
                    rhs += np.dot((1-self.theta)*M,state0)
                    scal.state[:]=np.linalg.solve(I-self.theta*M,rhs)

                scal.record_state(t1)
                scal.J=J

            self.t_idx+=1
    
class Scalar(object):
    """
    Track state, history, boundary conditions for a scalar
    """
    state=None
    
    def __init__(self,name,transport,**kws):
        self.history=[]
        self.name=name
        self.transport=transport

        set_variables(self,kws)
        
    def boundary_C(self,t):
        return None # same as self.transport.boundary_C_zero()
    def boundary_J(self,t):
        """
        Mass flux
        """
        return None # same as self.transport.boundary_C_zero()
    def initial_C(self,t):
        return self.transport.initial_C_zero()

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

