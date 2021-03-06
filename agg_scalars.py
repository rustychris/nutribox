"""
Read in passive scalar runs, aggregate to the same spatial configuration
as the hydro
"""

import os
import numpy as np
import xarray as xr 
from stompy import utils
import aggregate_wy2013c as agg
import stompy.model.delft.io as dio
import stompy.model.delft.waq_scenario as waq
from stompy.model.delft import dfm_grid
import glob
from collections import defaultdict

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


# Not needed, right?
agg_lp_2d_dir='hydro-wy2013c_adj-agg_lp_2d'
hydro_2d=waq.HydroFiles('hydro-wy2013c_adj-agg_lp_2d/com-wy2013c_adj_agg_lp_2d.hyd')


import lowpass_wy2013c
lp_secs=lowpass_wy2013c.lp_secs
lp_hyd=lowpass_wy2013c.get_hydro()

## 

npad=int(5*lp_secs / 86400.)
daily_pad=np.zeros(npad)
from stompy import filters
def lowpass_daily(data):
    """
    Replicate as much as possible the lowpass from lowpass_wy2013c, but
    applied to daily data.
    """
    flow_padded=np.concatenate( ( daily_pad, 
                                  data,
                                  daily_pad) )
    lp_flows=filters.lowpass(flow_padded,
                             cutoff=lp_secs,dt=86400.)
    lp_flows=lp_flows[npad:-npad] # trim the pad
    return lp_flows

        
##
scalar_dir='scalar_data_agg_lp_2d'
os.path.exists(scalar_dir) or os.makedirs(scalar_dir)

runs=glob.glob("/opt/data/dwaq/sfbay_constracer/runs/wy2013c-*")

for run_dir in runs:
    print("Processing run %s"%run_dir)
    scalar_name=run_dir.split('-')[-1]

    #--- Integrate concentrations from the model output
    nc_fn=os.path.join( scalar_dir,scalar_name+".nc")
    if os.path.exists(nc_fn):
        print("%s data exists"%run_dir)
    else:
        print(run_dir)

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

    # -with_bc2: the 2 indicates that it has concentrations, too.
    # -with_bc3: testing issues with delta stormwater
    # -with_bc4: lowpassing the concentrations
    nc_bc_fn=nc_fn.replace('.nc','-with_bc4.nc')

    if os.path.exists(nc_bc_fn) and (os.stat(nc_bc_fn).st_mtime >= os.stat(nc_fn).st_mtime):
        print("BC data already in place")
    else:
        # The other factor here is boundary conditions.
        #   For each time step, basically need to know what the effective boundary concentration
        #   was, i.e. what is consistent with the flow coming into this cell.
        #   First step there is knowing what the water flows coming in are.

        # The boundaries are already aggregated, such that COYOTEd, EBAYS, and san_jose
        # are combined.  Bummer.  Is it possible to reach back to the run setup?

        inp_fn=os.path.join(run_dir,'sfb_dfm_v2.inp')

        # Which exchanges participate in this scalar's source?
        boundaries,bc_blocks=dio.parse_boundary_conditions(inp_fn) # get 2480 boundaries

        # Does not yet allow for time-varying BC concentrations!
        boundary_conc=np.zeros(len(boundaries),'f8')

        # Partially robust approach
        # Scan the individual item blocks
        for bc_block in bc_blocks:
            # rows/cols can be either scalars/bc_links or bc_links/scalars
            # and this dictates the form of the matrix
            rows,cols,data=bc_block
            if rows[0]=='concentration':
                concs=rows[1]
                items=cols[1]
                conc_item=data
            else:
                concs=cols[1]
                items=rows[1]
                conc_item=data[1].T
            if scalar_name in concs:
                conc_idx=list(concs).index(scalar_name.lower())
                scalar_data=conc_item[conc_idx,:]
                for item,datum in zip(items,scalar_data):
                    for idx,(boundary_id,boundary_name,boundary_group) in enumerate(boundaries):
                        #  
                        if ( (idx+1==item) or
                             ( item.lower() in [ boundary_id.lower(),
                                                 boundary_name.lower(),
                                                 boundary_group.lower() ] ) ):
                            boundary_conc[idx]=datum
        # here hits is zero-based, positive indices into the list of boundary exchanges,
        # should be same as previous, fragile code
        hits=np.nonzero(boundary_conc)[0]
        hit_concs=boundary_conc[ boundary_conc!=0.0 ]
            
        poi=hyd.pointers

        # The goal here is to get a time series flow rate on a per-2D-aggregated segment
        # basis.

        hit_exchs=[np.nonzero(poi[:,0]==-(hit+1))[0][0]
                   for hit in hits]

        # Aggregate just this flow information
        hyd.infer_2d_links()

        hit_links=hyd.exch_to_2d_link[hit_exchs]
        assert np.all(hit_links['sgn']==1)
        hit_links=hit_links['link']

        # Now map those links to the internal, unaggregated element
        hit_elements=hyd.links[hit_links,1]

        # Now how do those 2D links map to the aggregated grid?
        agg_elements=aggregator.elt_global_to_agg_2d[hit_elements]

        element_mass_influx=np.zeros( (len(hyd.t_secs),Nagg), 'f8')
        # This will first be used to sum flows, then used to normalize the
        # the mass influx to get resulting concentration
        element_water_influx=np.zeros( (len(hyd.t_secs),Nagg), 'f8')

        # Given an aggregated element index, what is the list of unaggregated
        # boundary exchanges which contribute to it
        exchs_for_agg_elt=defaultdict(list)
        # Limit to boundary exchanges
        all_bc_exchs=np.nonzero(poi[:,0]<0)[0]
        for bc_exch in all_bc_exchs:
            assert hyd.exch_to_2d_link[bc_exch]['sgn']==1
            bc_link=hyd.exch_to_2d_link[bc_exch]['link']
            bc_elt=hyd.links[bc_link,1]
            agg_elt=aggregator.elt_global_to_agg_2d[bc_elt]
            exchs_for_agg_elt[agg_elt].append(bc_exch)

        #for agg_elt in np.unique(agg_elements):
        #    print("Agg element %d has exchanges %s"%(agg_elt,exchs_for_agg_elt[agg_elt]))
            
        # This is pretty slow - maybe 10 minutes without memmap, 2 minutes with the memmap code.
        # When there are many exchanges (e.g. stormwater) it's that much slower.
        for t_idx,t in enumerate(hyd.t_secs):
            flo=hyd.flows(hyd.t_secs[t_idx],memmap=True)

            # 2018-05-05: it is wrong here to loop over only the hits.  The hits
            # are a subset of unaggregated boundaries which carry a concentration
            # of the current scalar.  But here we need to dilute those by the sum
            # of *all* BC flows into this aggregated element.
            for hit_exch,hit_conc,agg_elt in zip(hit_exchs,hit_concs,agg_elements):
                # Clip this to inflow only, to be consistent with influx below.
                element_mass_influx[t_idx,agg_elt] += flo[hit_exch].clip(0,np.inf) * hit_conc
                
                # Used to do this, but as noted above it only captures a subset
                # of the flows into this element.
                # element_conc_influx[t_idx,agg_elt] += flo[hit_exch]
            for agg_elt in np.unique(agg_elements):
                # sum BC flows over all boundary exchanges hitting this agg_elt
                # NOTE: this could be negative, and there could be offsetting fluxes here.
                # What's the "right" way to sum these?  When all flows are positive, it
                # doesn't matter.  If all flows are negative, we'll be taking the upwind
                # concentration anyway, which is inside the domain, so concentrations here
                # don't matter. if the net flow into the element is zero, then advection
                # isn't going to cut it.  The safest is to take the sum of inflows, and ignore
                # outflows. This might be a problem if there are odd combinations of boundary
                # flows for a single element!
                element_water_influx[t_idx,agg_elt] += flo[exchs_for_agg_elt[agg_elt]].clip(0,np.inf).sum()

            if t_idx%100==0:
                max_flux=element_mass_influx[t_idx,:].max()
                sum_flux=element_mass_influx[t_idx,:].sum()
                print("%d/%d flux max=%s sum=%s"%(t_idx,len(hyd.t_secs),max_flux,sum_flux))

        # Still need to lowpass.  These will be written out with the same
        # time resolution as the aggregated hydro, which has not been decimated
        # in time.
        print("Lowpassing...")
        for i in range(Nagg):
            element_mass_influx[:,i] = lp_hyd.lowpass(element_mass_influx[:,i])
            element_water_influx[:,i] = lp_hyd.lowpass(element_water_influx[:,i])

        # Finally, normalize to concentration here:
        zero_flow=np.abs(element_water_influx)<1e-6
        element_conc_influx=np.zeros_like(element_water_influx)
        element_conc_influx[~zero_flow] = element_mass_influx[~zero_flow] / element_water_influx[~zero_flow]
        # could be nice to know a forced concentration even with zero flow, but not today
        element_conc_influx[zero_flow] = 0.0
            
        # Data looks okay. Write that out

        # concentrations are daily
        # element_mass_influx is 0.5h.  It's been lowpassed, so good enough to
        # just decimate

        conc=xr.open_dataset(nc_fn)
        conc_dnum=utils.to_dnum(conc.time.values)
        time_sel=np.searchsorted(lp_hyd.t_dn, conc_dnum)

        lp_scalar=conc.scalar.values
        for i in range(Nagg):
            lp_scalar[:,i] = lp_hyd.lowpass( lp_scalar[:,i] )
        conc['scalar']=conc.scalar.dims, lp_scalar

        # This assumes that all inflows enter with the same concentration.  That
        # won't be true for salinity, but okay for the conservative tracers.
        conc['bc_mass_inflow']= ('time','face'), element_mass_influx[time_sel,:]
        conc['bc_conc_inflow']= ('time','face'), element_conc_influx[time_sel,:]
        conc['bc_water_inflow']= ('time','face'), element_water_influx[time_sel,:]
        conc.to_netcdf(nc_bc_fn)
        
