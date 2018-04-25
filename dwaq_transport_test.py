import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from stompy import utils

import stompy.model.delft.waq_scenario as waq
import dwaq_transport
from stompy.plot import plot_utils

##

# This does not integrate over the timesteps, but dwaq_transport will handle that.
hydro=waq.HydroFiles('hydro-wy2013c_adj-agg_lp_2d/com-wy2013c_adj_agg_lp_2d.hyd')

##
six.moves.reload_module(dwaq_transport)

transport=dwaq_transport.DwaqTransport(hydro,
                                       times=hydro.t_secs[::2*48][50:150])

# check_conservation_incr reports that, at least, hydro.time 100:200
# is pretty good, with typical losses per time step of 0.0001 m
#transport=dwaq_transport.DwaqTransport(hydro,
#                                       times=hydro.t_secs[100:400])

# salt=transport.add_scalar("salinity")
# 
# @utils.add_to(salt)
# def boundary_C(self,t):
#     x=self.transport.boundary_C_zero()
#     x[:]=0.0 # freshwater sources
#     for bc_elt in self.transport.bnd_map:
#         bc_idx=-bc_elt-1
#         name=transport.bnd_map[bc_elt]
#         if 'Sea' in name:
#             x[bc_idx]=34
#     return x
    
cont=transport.add_scalar("continuity")

@utils.add_to(cont)
def initial_C(self,t):
    x=self.transport.initial_C_zero()
    x[:]=1.0
    return x

if 0: # The previous way - hardcode boundary concentrations
    @utils.add_to(cont)
    def boundary_C(self,t):
        x=self.transport.boundary_C_zero()
        x[:]=1.0
        return x
else:
    # Load boundary info for continuity:
    # try the derived BC data, but using concentration
    scalar_bc_data=xr.open_dataset('scalar_data/continuity-with_bc2.nc')
    exch_is_bc=(transport.hydro.pointers[:,0]<0)
    bc_from=transport.hydro.pointers[exch_is_bc,0] # -1, ... -48
    bc_from_idx=-(bc_from+1)
    # Could be relaxed, but it seems that the standard is for boundary elements to be
    # numbered consecutive with their appearance in boundary exchanges
    assert np.all( bc_from_idx==np.arange(len(bc_from)) )
    bc_to  =transport.hydro.pointers[exch_is_bc,1]
    elt_per_bc_idx=bc_to-1 # adjust to 0-based
    @utils.add_to(cont)
    def boundary_C(self,t):
        # t comes in as seconds since self.transport.hydro.time0 (which is a datetime)
        t_dt64=utils.to_dt64(self.transport.hydro.time0) + t*np.timedelta64(1,'s')
        t_idx=np.searchsorted(scalar_bc_data.time.values,t_dt64)
        per_element=scalar_bc_data.bc_conc_inflow.isel(time=t_idx).values
        # have to transform this back to per-boundary -- return to this.
        # per_element: 141 values (one per element...)
        # goal: 48 values (one per boundary exchange)
        return per_element[elt_per_bc_idx]

transport.initialize()

transport.loop()

##

plt.figure(1).clf()
fig,ax=plt.subplots(num=1)

tidx=0
ccoll0=transport.g.plot_cells(values=cont.history[tidx][1],ax=ax,cmap='seismic')
ccoll0.set_clim([0.7,1.3])

dt64=np.datetime64(transport.hydro.time0) + transport.times[tidx]*np.timedelta64(1,'s')
time_str=str(dt64)[:16]
txt=ax.set_title(time_str)
plot_utils.cbar(ccoll0,ax=ax)

ax.xaxis.set_visible(0)
ax.yaxis.set_visible(0)
ax.set_facecolor("0.8")

## 
for tidx in range(0,len(cont.history),2):
    ccoll0.set_array(cont.history[tidx][1])
    dt64=np.datetime64(transport.hydro.time0) + transport.times[tidx]*np.timedelta64(1,'s')
    time_str=str(dt64)[:16]
    txt=ax.set_title(time_str)
    
    plt.draw()
    plt.pause(0.5)

##

salt=hydro.parameters()['salinity']

plt.clf()
g.plot_cells(values=salt.evaluate(t=t).data)

##

# For now, use 2D centroid of the cells, which are already the nodes of
# the dual
self=transport

L=self.gd.edges_length() # 243
Aflux=transport.hydro.areas(transport.hydro.t_secs[1]) # 291

# After updating the waq code, there are now 291 exchanges in the hydro,
# vs. 48 boundary exchanges + 243 edges in the dual.  Finally they match!

# the dual has 141 nodes, and 243 edges.
# the original grid has 578 edges, of which 324 are between real cells

##

