"""
Load BCs and aggregated output for a conservative scalar run, and start 
to look at how to tune exchange to reproduce those results.
"""

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



## 

transport=dwaq_transport.DwaqTransport(hydro,
                                       times=hydro.t_secs[::2*48][50:100])

# Load boundary info for continuity:
exch_is_bc=(transport.hydro.pointers[:,0]<0)
bc_from=transport.hydro.pointers[exch_is_bc,0] # -1, ... -48
bc_from_idx=-(bc_from+1)
assert np.all( bc_from_idx==np.arange(len(bc_from)) )
bc_to  =transport.hydro.pointers[exch_is_bc,1]
elt_per_bc_idx=bc_to-1 # adjust to 0-based

class DwaqScalar(dwaq_transport.Scalar):
    scalar_data=None


all_scalars=[
    #'continuity', # gradients too small
    # 'sea', # is this properly set up?
    'novato','valero','millbrae_burlingame','tesoro',
    'calistoga','ddsd','delta','shell','san_jose','benicia',
    'south_bayside','treasure_island','petaluma','san_mateo','pinole',
    'ebda','american','sfo','sunnyvale','mt_view','vallejo',
    'rodeo','cccsd','west_county_richmond','sausalito','napa','fs',
    'st_helena','palo_alto','ebmud','central_marin','phillips66','ch',
    'lg','marin5','sonoma_valley','stormwater','south_sf','sf_southeast','chevron',
    'yountville']

for scalar_name in all_scalars:
    scalar_name="ebda"
    scalar_data_fn="scalar_data/%s-with_bc2.nc"%scalar_name
    scalar_data=xr.open_dataset(scalar_data_fn)
    scalar_data['t_sec']=('time',), (scalar_data.time.values - utils.to_dt64(transport.hydro.time0))/np.timedelta64(1,'s')
    scal=DwaqScalar(name=scalar_name,transport=transport,scalar_data=scalar_data)
    transport.add_scalar(scalar=scal)

    @utils.add_to(scal)
    def initial_C(self,t):
        scal_t_idx=np.searchsorted(self.scalar_data.t_sec.values,t)
        return self.scalar_data.scalar.values[scal_t_idx,:].copy()

    @utils.add_to(scal)
    def boundary_C(self,t):
        t_idx=np.searchsorted(self.scalar_data.t_sec.values,t)
        per_element=self.scalar_data.bc_conc_inflow.isel(time=t_idx).values
        # per_element: 141 values (one per element...)
        # goal: 48 values (one per boundary exchange)
        concs=per_element[elt_per_bc_idx]
        # either need to fix or rerun agg_scalars - there shouldn't be nan concentrations
        # there.
        concs[np.isnan(concs)]=0.0
        return concs

transport.initialize()

transport.loop()

##

plt.figure(1).clf()
fig,(ax,ax_res)=plt.subplots(1,2,sharex=True,sharey=True,num=1,
                             subplot_kw=dict(adjustable='box', aspect='equal'))

tidx=1
ccoll0=transport.g.plot_cells(values=scal.history[tidx][1],ax=ax,cmap='jet')
ccoll0.set_clim([0.0,0.01])

ccoll1=transport.g.plot_cells(values=scal.initial_C(scal.history[tidx][0]),ax=ax_res,cmap='jet')
ccoll1.set_clim([0.0,0.01])

dt64=np.datetime64(transport.hydro.time0) + transport.times[tidx]*np.timedelta64(1,'s')
time_str=str(dt64)[:16]
txt=ax.set_title(time_str)
# plot_utils.cbar(ccoll0,ax=ax)

for a in [ax,ax_res]:
    a.xaxis.set_visible(0)
    a.yaxis.set_visible(0)
    a.set_facecolor("0.8")

ax.axis(transport.g.bounds())
fig.tight_layout()

## 
for tidx in range(0,len(scal.history)):
    ccoll0.set_array(scal.history[tidx][1])
    ccoll1.set_array(scal.initial_C(scal.history[tidx][0]))
    dt64=np.datetime64(transport.hydro.time0) + transport.times[tidx]*np.timedelta64(1,'s')
    time_str=str(dt64)[:16]
    txt=ax.set_title(time_str)
    
    plt.draw()
    plt.pause(0.01)

##

def scalar_constraints(scal,tidx0,tidx1):
    t0=scal.history[tidx0][0]
    t1=scal.history[tidx1][0]
    C_pred0=scal.history[tidx0][1]
    C_pred1=scal.history[tidx1][1]
    C_real0=scal.initial_C(t0).copy()
    C_real1=scal.initial_C(t1)

    vol1=transport.hydro.volumes(t1)

    C_err0=C_pred0-C_real0
    assert np.all(C_err0==0) # risky with floating point, but they really should be identical
    C_err1=C_pred1-C_real1

    # The scalar mass error in each cell, which forms the right hand side
    error_vector=(C_real1-C_pred1)*vol1

    scal_M_block=np.zeros( (transport.N, transport.hydro.n_exch - transport.N_boundary), 'f8')

    for exch_j in range(transport.gd.Nedges()):
        a,b=transport.gd.edges[exch_j]['nodes']
        G_ab=C_pred0[a] - C_pred0[b]

        # positive G_ab means that diffusion will decrease the mass in a
        scal_M_block[a,exch_j] += -G_ab
        scal_M_block[b,exch_j] +=  G_ab

    # Normalization - with widely varying concentrations, least squares will
    # heavily weight the strongest concentration field.  Normalize by volume
    # and mean concentration
    if 1:
        norm_factors=C_real0*vol1
        error_vector[:] /= norm_factors
        scal_M_block[:] /= norm_factors[:,None]

    if 0: # Dev only tests, since they will fail when gradients are zero.
        # each edge, and thus each diffusion, should get used twice in the matrix,
        # once for each of the two cells it can affect.  
        assert np.all( (scal_M_block!=0).sum(axis=0) == 2 )
        # Each cell presumably connects to at least one other cell
        assert np.all( (scal_M_block!=0).sum(axis=1) > 0 )

    return scal_M_block,error_vector


M_blocks=[]
error_vectors=[]


for scal in transport.scalars:
    scal_M_block,error_vector=scalar_constraints(scal,tidx0=0,tidx1=1)
    M_blocks.append(scal_M_block)
    error_vectors.append(error_vector)

combined_M=np.concatenate(M_blocks,axis=0)
combined_error=np.concatenate(error_vectors)
    
##

# Even with 6 scalars, this is pretty bad.
# Going with all of them, still not great.
# Going for a longer period makes it worse.
# Could try multiple short periods, maybe
# that would smooth out some transient issues
# Probably lots of options for better normalization, too
D_lsq,residuals,rank,s=np.linalg.lstsq(combined_M,combined_error)

from scipy import optimize
D_lsq_nn,rnorm=optimize.nnls(combined_M,combined_error)

##

plt.figure(2).clf()

transport.g.plot_cells(lw=0.5)
ecoll=transport.gd.plot_edges(values=np.log10(D_lsq_nn.clip(1,np.inf)))
ecoll.set_cmap('jet')
ecoll.set_clim([0,8])

