"""
Read in passive scalar runs, aggregate to the same spatial configuration
as the hydro
"""

import os

import xarray as xr 
from stompy import utils
import aggregate_wy2013c as agg
import stompy.model.delft.io as dio
import stompy.model.delft.waq_scenario as waq
from stompy.model.delft import dfm_grid
import glob

##

# SLOW! - just do this once
aggregator=agg.get_hydro()

##

g_agg=aggregator.grid()
Nagg=1+aggregator.elt_global_to_agg_2d.max()

# Get a consistent unaggregated grid -- the one from scalar_map is bad.
# g_waq=dfm_grid.DFMGrid(scalar_map) # BAD

# This grid matches both the scalar fields coming out of WAQ, and
# the aggregation mapping
waq_hyd_dir='/opt/data/delft/sfb_dfm_v2/runs/wy2013c/DFM_DELWAQ_wy2013c_adj'
g_waq=dfm_grid.DFMGrid(os.path.join(waq_hyd_dir,'wy2013c_waqgeom.nc'))
# 
hyd=waq.HydroFiles(os.path.join(waq_hyd_dir,"wy2013c.hyd"))


##
scalar_dir='scalar_data'
os.path.exists(scalar_dir) or os.makedirs(scalar_dir)

runs=glob.glob("/opt/data/dwaq/sfbay_constracer/runs/wy2013c-*")

for run_dir in runs:
    print(run_dir)
    scalar_name=run_dir.split('-')[-1]
    nc_fn=os.path.join( scalar_dir,scalar_name+".nc")
    if os.path.exists(nc_fn):
        print("  exists")
        continue
    
    scalar_map=dio.read_map( fn=os.path.join(run_dir,"sfb_dfm_v2.map"),
                             hyd=os.path.join(run_dir,"com-sfb_dfm_v2.hyd") )
    
    scalar=scalar_map[scalar_name]
    volumes=scalar_map['volume']

    agg_scalar=np.zeros( (len(scalar_map.time),Nagg) )

    for t_idx,t in enumerate(scalar_map.time):
        print("Time: %s"%utils.to_datetime(t.values).strftime('%Y-%m-%d'))
        # enforce ordering of dimensions for safety
        scalar_value=scalar.isel(time=t_idx).transpose('layer','face').values
        volume=volumes.isel(time=t_idx).transpose('layer','face').values

        # need to filter out the -999 values first.
        valid=(scalar_value!=-999)
        num=(scalar_value*volume*valid).sum(axis=0)
        den=(volume*valid).sum(axis=0)

        #empty=(den<=0)
        #num[empty]=0
        #den[empty]=1.0
        #scalar_2d= num/den

        # And aggregate
        for agg_idx,members in utils.enumerate_groups(aggregator.elt_global_to_agg_2d):
            # print(".",end='')
            agg_num=num[members].sum()
            agg_den=den[members].sum()
            if agg_den<=0.0:
                agg_scalar[t_idx,agg_idx]=0.0
            else:
                agg_scalar[t_idx,agg_idx] = agg_num/agg_den

    ds=g_agg.write_to_xarray()
    ds['time']=('time',),scalar_map.time.values
    ds['scalar']=('time','face'),agg_scalar
    ds.attrs['name']=scalar_name
    ds.to_netcdf( nc_fn )

## 

# The other factor here is boundary conditions.
#   For each time step, basically need to know what the effective boundary concentration
#   was, i.e. what is consistent with the flow coming into this cell.
#   First step there is knowing what the water flows coming in are.

hydro_2d=waq.HydroFiles('hydro-wy2013c_adj-agg_lp_2d/com-wy2013c_adj_agg_lp_2d.hyd')

##

# Say we have san jose scalar.
# The boundaries are already aggregated, such that COYOTEd, EBAYS, and san_jose
# are combined.  Bummer.  Is it possible to reach back to the run setup?

scalar_name='san_jose'

orig_waq_dir='/opt/data/dwaq/sfbay_constracer/runs/wy2013c-20180404-%s'%scalar_name
inp_fn=os.path.join(orig_waq_dir,'sfb_dfm_v2.inp')

# forcing data is just a concentration on the SJ flow.
conc=1

# Which exchanges participate in this scalar's source?
bcs,items=dio.parse_boundary_conditions(inp_fn) # get 2480 bcs

# luckily I know that the boundary groups are named by the scalar
hits=[ idx
       for idx,bc in enumerate(bcs)
       if bc[2]==scalar_name ]

poi=hyd.pointers

# The goal here is to get a time series flow rate on a per-2D-aggregated segment
# basis.

hit_exchs=[np.nonzero(poi[:,0]==-(hit+1))[0][0]
           for hit in hits]

# Aggregate just this flow information
# hyd.infer_2d_elements()
hyd.infer_2d_links()

hit_links=hyd.exch_to_2d_link[hit_exchs]
assert np.all(hit_links['sgn']==1)
hit_links=hit_links['link']

# Now map those links to the internal, unaggregated element
hit_elements=hyd.links[hit_links,1]

# Now how do those 2D links map to the aggregated grid?
agg_elements=aggregator.elt_global_to_agg_2d[hit_elements]

element_volume_influx=np.zeros( (len(hyd.t_secs),Nagg), 'f8')

# This is pretty slow - maybe 10 minutes?
# Any chance of speeding it up with memmap?
for t_idx,t in enumerate(hyd.t_secs):
    flo=hyd.flows(hyd.t_secs[t_idx])

    for hit_exch,agg_elt in zip(hit_exchs,agg_elements):
        element_volume_influx[t_idx,agg_elt] += flo[hit_exch]

    if t_idx%100==0:
        print("%d/%d max flux = %s"%(t_idx,len(hyd.t_secs),element_volume_influx[t_idx,:].max()))
## 

# Still need to lowpass.  These will be written out with the same
# time resolution as the aggregated hydro, which has not been decimated
# in time.
import lowpass_wy2013c
lp_secs=lowpass_wy2013c.lp_secs
lp_hyd=lowpass_wy2013c.get_hydro()

for i in range(Nagg):
    element_volume_influx[:,i] = lp_hyd.lowpass(element_volume_influx[:,i])

##

# Data looks okay --  HERE - need to write that out

