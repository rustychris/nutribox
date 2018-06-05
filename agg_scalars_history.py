"""
Similar to agg_scalars, but using the history output which does most of the aggregation
during the DWAQ run (just not any lowpass filter).
"""

import os
import glob
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from stompy.spatial import wkb2shp
from stompy.grid import unstructured_grid
from stompy import filters

##
agg_shp_fn='boxes-v02.shp'

if 0: # dev
    g=unstructured_grid.UnstructuredGrid.from_shp('boxes-v02.shp')

    scalar='stormwater'

    region_idxs=np.arange(1,1+g.Ncells())

    # Line up the regions
    scalar_time_region = his_nc.bal.isel(region=region_idxs).sel(field=scalar)

    fig=plt.figure(1)
    fig.clf()
    ax=fig.add_subplot(1,1,1)

    values=scalar_time_region.isel(time=-1).values

    ccoll=g.plot_cells(ax=ax,values=np.log10(values.clip(1e-5,1.)) )

    ax.axis('equal')

    for tidx in range(1000,2000,2):
        values=scalar_time_region.isel(time=tidx).values
        ccoll.set_array(values)
        plt.draw()
        plt.pause(0.05)

##

agg_dir="scalar_data_agg_his" # where to write the aggregated, lowpassed data
agg_bc_dir="scalar_data"

os.path.exists(agg_dir) or os.makedirs(agg_dir)
g_agg=unstructured_grid.UnstructuredGrid.from_shp(agg_shp_fn)

lp_secs=86400*3
dt_secs=np.median(np.diff(his_nc.time))/np.timedelta64(1,'s')

n_pad=int(5*lp_secs / dt_secs)
lp_pad=np.ones(n_pad)

def lowpass(data):
    # padding with first/last value is different than in waq_scenario.
    # but feels better. right?
    flow_padded=np.concatenate( ( lp_pad*data[0], data, lp_pad*data[-1]) )
    lp_flows=filters.lowpass(flow_padded, cutoff=lp_secs, dt=dt_secs)
    lp_flows=lp_flows[n_pad:-n_pad] # trim the pad
    return lp_flows


def add_bcs(scalar,ds):
    """
    Find the right aggregated bc data, and add it to ds, and assert that times line up
    """
    bc_nc_fn=os.path.join(agg_bc_dir,scalar+"-with_bc3.nc")
    bc_nc=xr.open_dataset(bc_nc_fn)

    shorter=min(len(bc_nc.time.values),
                len(ds.time.values))
    assert np.all( bc_nc.time.values[:shorter] == ds.time.values[:shorter] )
    
    # ds['time']=('time',),bc_nc.time.values
    
    for fld in ['bc_mass_inflow','bc_conc_inflow','bc_water_inflow']:
        ds[fld]=bc_nc[fld].isel(time=slice(None,shorter))
    return ds

def process_one(his_nc,scalar,overwrite=False):
    # fragile, may have to match by name in the future.
    region_idxs=np.arange(1,1+g_agg.Ncells())
    
    agg_nc_fn=os.path.join(agg_dir,'%s.nc'%scalar)
    if os.path.exists(agg_nc_fn):
        if overwrite:
            os.unlink(agg_nc_fn)
        else:
            print("%s exists - skip"%agg_nc_fn)
            return
    
    ds=g_agg.write_to_xarray()

    values=his_nc.bal.isel(region=region_idxs).sel(field=scalar).values
    for i in range(values.shape[1]):
        values[:,i] = lowpass(values[:,i])

    output_stride=slice(None,None,int(round(86400/dt_secs)))

    ds['time']=('time',),his_nc.time.values[output_stride]    
    ds['scalar']=('time','face'),values[output_stride,:]    
    ds.attrs['name']=scalar

    add_bcs(scalar,ds)
    
    ds.to_netcdf( agg_nc_fn )
    
def process(ds,overwrite=False):
    for scalar in ds.field.values:
        if scalar.lower() in ['localdepth','surf']:
            continue
        print("Processing scalar %s"%scalar)

        process_one(ds,scalar,overwrite=overwrite)
        
for his_nc_fn in glob.glob("histories/*.nc"):
    ds=xr.open_dataset(his_nc_fn)
    process(ds,overwrite=True)

##

# Make it look like the aggregated scalar data already written out
ds_old=xr.open_dataset('scalar_data/stormwater-with_bc3.nc')
