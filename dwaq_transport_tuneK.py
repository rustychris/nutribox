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
from stompy.grid import unstructured_grid
six.moves.reload_module(unstructured_grid)
from stompy.model.delft import dfm_grid
six.moves.reload_module(dfm_grid)
six.moves.reload_module(dwaq_transport)


## 

transport=dwaq_transport.DwaqTransport(hydro,
                                       times=hydro.t_secs[::48][50:150])

transport.history=[]

@utils.add_to(transport)
def record_transport(self,**kws):
    self.history.append(kws)
    
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

# While testing:
all_scalars=['stormwater','valero','vallejo','cccsd','fs']

for scalar_name in all_scalars:
    scalar_data_fn="scalar_data_agg_his/%s.nc"%scalar_name
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
        low_end= norm_factors[ norm_factors>0 ].min()
        norm_factors += 0.1*low_end # avoid division by zero!
        error_vector[:] /= norm_factors
        scal_M_block[:] /= norm_factors[:,None]

    if 0: # Dev only tests, since they will fail when gradients are zero.
        # each edge, and thus each diffusion, should get used twice in the matrix,
        # once for each of the two cells it can affect.  
        assert np.all( (scal_M_block!=0).sum(axis=0) == 2 )
        # Each cell presumably connects to at least one other cell
        assert np.all( (scal_M_block!=0).sum(axis=1) > 0 )

    assert np.all( np.isfinite(scal_M_block) )
    assert np.all( np.isfinite(error_vector) )
    return scal_M_block,error_vector

## 

M_blocks=[]
error_vectors=[]

for t_idx0 in range(0,100,1):
    print("Looping from t_idx=%d"%t_idx0)
    # How much of this can be re-done?    
    transport.initialize(t_idx=t_idx0)
    transport.loop(t_idx_end=t_idx0+3) # don't go for too long

    for scal in transport.scalars:
        scal_M_block,error_vector=scalar_constraints(scal,tidx0=0,tidx1=1)
        M_blocks.append(scal_M_block)
        error_vectors.append(error_vector)

combined_M=np.concatenate(M_blocks,axis=0)
combined_error=np.concatenate(error_vectors)
    
##

# still more mottled than ideal.
# D,residuals,rank,s=np.linalg.lstsq(combined_M,combined_error,rcond=None)

from scipy import optimize
D,rnorm=optimize.nnls(combined_M,combined_error)


##

# Show the inferred dispersions
fig=plt.figure(20)
fig.clf()

transport.g.plot_cells(lw=0.5)
ecoll=transport.gd.plot_edges(values=np.log10(D.clip(1,np.inf)))
ecoll.set_cmap('jet')
ecoll.set_clim([0,10])

fig.axes[0].axis('equal')

##

# Take a look at the error fields
storm_idx=0 # was 36 in old run.

scal_i=storm_idx
scal=transport.scalars[scal_i]

plt.figure(3).clf()
fig,axs=plt.subplots(2,3,sharex=True,sharey=True,num=3,
                     subplot_kw=dict(adjustable='box-forced', aspect=1.0))

conc_clim=[0.0,0.2]

for ax_row,tidx in zip(axs,[0,1]):
    dt64=np.datetime64(transport.hydro.time0) + transport.times[tidx]*np.timedelta64(1,'s')
    time_str=str(dt64)[:16]

    scal_pred=scal.history[tidx][1]
    ccoll0=transport.g.plot_cells(values=scal_pred,ax=ax_row[0],cmap='jet')
    ccoll0.set_clim(conc_clim)
    ax_row[0].set_title("pred %s\n%s"%(scal.name,time_str))
    plt.colorbar(ccoll0,ax=ax_row[0])

    scal_orig=scal.initial_C(scal.history[tidx][0])
    ccoll1=transport.g.plot_cells(values=scal_orig,ax=ax_row[1],cmap='jet')
    ccoll1.set_clim(conc_clim)
    ax_row[1].set_title("orig %s\n%s"%(scal.name,time_str))
    plt.colorbar(ccoll1,ax=ax_row[1])

    scal_diff=scal_pred-scal_orig
    ccoll2=transport.g.plot_cells(values=scal_diff,ax=ax_row[2],cmap='seismic')
    ccoll2.set_clim( [-np.abs(scal_diff).max(),
                      np.abs(scal_diff).max()] )
    plt.colorbar(ccoll2,ax=ax_row[2])

for ax in axs.flat:
    ax.xaxis.set_visible(0)
    ax.yaxis.set_visible(0)
    ax.set_facecolor("0.8")

axs[0,0].axis(transport.g.bounds())
    
fig.tight_layout()

# ebda: there is a distinct difference in where alameda and coyote creek come in
#   looks like the original data shows a weird dip in concentration there?
#   predictions slowly climb, while the original data hovers, declines slightly.
#   stormwater: both climb, with the original much spikier
#   Also, at this 24 hour time scale, seems like the greatest errors are in location
#   of the plume, not spreading of the plume.  The original data show the plume advecting
#   seaward compared to the predictions.

# stormwater: alameda, petaluma, sonoma are low.
# napa, coyote are high, and the delta is significantly high.
# Delta is maybe the easiest to debug - it goes from basically 0 to 0.15 stormwater
#  concentration in one day.
# After fixing the aggregation process when there are multiple bc inputs to the same
# element, the Delta is behaving.  But Alameda is not. What gives?  It looks much worse
# in fact, basically trending to zero.

##

# What is the time series of concentration at Alameda Creek look like?

# Run for more time steps than above
transport.initialize(t_idx=2)
transport.loop(t_idx_end=100)

## 
# alameda=transport.gd.select_nodes_nearest( plt.ginput()[0] )
alameda=29
cell=alameda

def plot_budget(cell,scalar_name,cell_name):
    origs=[]
    preds=[]

    scal=[ s
           for s in transport.scalars
           if s.name==scalar_name ][0]

    times=[]

    pred_times=[]
    orig_times=[]

    def sec_to_dt(sec):
        return np.datetime64(transport.hydro.time0) + sec*np.timedelta64(1,'s')

    M_sample=transport.history[-1]['M']
    sel_cols=np.nonzero( np.abs(M_sample[alameda,:]) + np.abs(M_sample[:,alameda]) )[0]
    
    for hidx in range(len(scal.history)):
        preds.append( scal.history[hidx][1][alameda] )
        pred_times.append( sec_to_dt(scal.history[hidx][0]) )
        
        origs.append( scal.initial_C(scal.history[hidx][0])[alameda] )
        orig_times.append( sec_to_dt(scal.history[hidx][0]) )

        # And the flux terms from M:
        M=transport.history[hidx]['M']
        delta_c=np.dot(M,scal.history[hidx][1])
        

    fig=plt.figure(5)
    fig.clf()
    fig,(ax,ax_Q)=plt.subplots(2,1,num=5,sharex=True,sharey=True)

    ax.plot( utils.to_dnum(np.array(pred_times)), preds, label='pred %s at %s'%(scal.name,cell_name))
    ax.plot( utils.to_dnum(np.array(orig_times)), origs, label='orig %s at %s'%(scal.name,cell_name))
    ax.legend()
    ax.xaxis.axis_date()

    fig.autofmt_xdate()

plot_budget(cell=alameda,scalar_name='stormwater',cell_name='alameda')

    
## 

# So what's up with the budget for predicting stormwater at Alameda?
plt.figure(25).clf()
M=transport.history[-1]['M']

fig,axs=plt.subplots(1,2,sharex=True,sharey=True,num=25)

transport.g.plot_cells(values=M[alameda,:],ax=axs[0])
transport.g.plot_cells(values=M[:,alameda],ax=axs[1])
axs[0].axis('equal')

## 
# Backtracking to find where that J_stormwater into the Delta
# came from
scalar_name='stormwater'
scalar_data_fn="scalar_data/%s-with_bc3.nc"%scalar_name
scalar_data=xr.open_dataset(scalar_data_fn)
scalar_data['t_sec']=('time',), (scalar_data.time.values - utils.to_dt64(transport.hydro.time0))/np.timedelta64(1,'s')
# scal=DwaqScalar(name=scalar_name,transport=transport,scalar_data=scalar_data)
# transport.add_scalar(scalar=scal)

t=scal.history[0][0]
t_idx=np.searchsorted(scalar_data.t_sec.values,t)

# this now has non-zero values in the right places, but the concentrations
# are often too low.  aggregated elements which should have only stormwater
# coming in are showing concentrations much lower than 1.0 -- max conc.
# is 0.1.  Actually, eerily close to 0.1, which sounds like a layer problem?
# okay - it was a multiple agg_elements issue.  Regenerating...
per_element=scalar_data.bc_conc_inflow.isel(time=t_idx).values

# flows based on scalar_data:
# [g/s] / [g/m3] => m3/s
# These concentrations are now all 0.0 -- that's bad!
# similarly, water_inflow is all zero.
Q=(scalar_data.bc_mass_inflow / scalar_data.bc_conc_inflow).isel(time=t_idx).values
Q[ np.isnan(Q) ] =0.0


# per_element: 141 values (one per element...)
# goal: 48 values (one per boundary exchange)
concs=per_element[elt_per_bc_idx]
# either need to fix or rerun agg_scalars - there shouldn't be nan concentrations
# there.
concs[np.isnan(concs)]=0.0
return concs
