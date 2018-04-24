import numpy as np
import matplotlib.pyplot as plt

import stompy.model.delft.waq_scenario as waq
import dwaq_transport
from stompy.plot import plot_utils

##

def zero_vec(t,x):
    x[:]=0
    return x
def unit_vec(t,x):
    x[:]=1.0
    return x

# Note! this does not integrate over the timesteps yet.
# taking these long steps means that there is some loss
# of continuity.
hydro=waq.HydroFiles('hydro-wy2013c_adj-agg_lp_2d/com-wy2013c_adj_agg_lp_2d.hyd')

##
six.moves.reload_module(dwaq_transport)

transport=dwaq_transport.DwaqTransport(hydro,
                                       times=hydro.t_secs[1000:1000+125*48:2*48])

salt=transport.add_scalar("salinity")

@utils.add_to(salt)
def boundary_C(self,t):
    x=self.transport.boundary_C_zero()
    x[:]=0.0 # freshwater sources
    for bc_elt in self.transport.bnd_map:
        bc_idx=-bc_elt-1
        name=transport.bnd_map[bc_elt]
        if 'Sea' in name:
            x[bc_idx]=34
    return x
    
cont=transport.add_scalar("continuity")
@utils.add_to(cont)
def initial_C(self,t):
    x=self.transport.initial_C_zero()
    x[:]=1.0
    return x


if 0: # The previous way:
    @utils.add_to(cont)
    def boundary_C(self,t):
        x=self.transport.boundary_C_zero()
        x[:]=1.0
        return x
if 1:
    # Load boundary info for continuity:
    # HERE
    # the ocean is mostly below the expected unity, most of the Bay
    # is above the expected (up to 10 or so).
    # 
    scalar_bc_data=xr.open_dataset('scalar_data/continuity-with_bc.nc')
    @utils.add_to(cont)
    def boundary_J(self,t):
        # t comes in as seconds since self.transport.hydro.time0 (which is a datetime)
        t_dt64=utils.to_dt64(self.transport.hydro.time0) + t*np.timedelta64(1,'s')
        t_idx=np.searchsorted(scalar_bc_data.time,t_dt64)
        return scalar_bc_data.bc_inflow.isel(time=t_idx).values

# There isn't boundary information for salt since it wasn't run in DWAQ

# This has data once a day for each of 141 faces/segments/elements
# data consists of concentrations, and mass inflow.

transport.initialize()

transport.loop()

##

plt.figure(1).clf()
fig,axs=plt.subplots(1,2,num=1)

ccoll0=transport.g.plot_cells(values=cont.history[0][1],ax=axs[0],cmap='jet')
ccoll1=transport.g.plot_cells(values=cont.history[-1][1],ax=axs[1],cmap='jet')

plot_utils.cbar(ccoll0,ax=axs[0])
plot_utils.cbar(ccoll1,ax=axs[1])

for ax in axs:
    ax.xaxis.set_visible(0)
    ax.yaxis.set_visible(0)

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

poi=transport.hydro.pointers.copy()

# internal exchanges
poi0_int = poi[ poi[:,0]>=0, :2 ] - 1
segs=transport.gd.nodes['x'][poi0_int]

## 
hydro_unagg=waq.HydroFiles("/opt/data/delft/sfb_dfm_v2/runs/wy2013c/DFM_DELWAQ_wy2013c_adj/wy2013c.hyd")
g_orig=hydro_unagg.grid()


