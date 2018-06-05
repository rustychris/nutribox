"""
Look at the aggregated and lowpass data - is it actually lowpassing?
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
scalar_name="stormwater"
scalar3_data_fn="scalar_data/%s-with_bc3.nc"%scalar_name
scalar4_data_fn="scalar_data_agg_lp_2d/%s-with_bc4.nc"%scalar_name

scalar3_nc=xr.open_dataset(scalar3_data_fn)
scalar4_nc=xr.open_dataset(scalar4_data_fn)

##

g=unstructured_grid.UnstructuredGrid.from_ugrid(scalar_nc)

##

# looks fine.
plt.clf() ; g.plot_edges() ; plt.axis('equal')

##

alameda=29 # g.select_cells_nearest( plt.ginput(1)[0] )


fig=plt.figure(6)
fig.clf()
ax=fig.gca()
ax.plot(scalar3_nc.time,scalar3_nc['scalar'].isel(face=29),label='bc3')
ax.plot(scalar3_nc.time,lowpass_daily(scalar3_nc['scalar'].isel(face=29)),label='lp(bc3)')
ax.plot(scalar4_nc.time,scalar4_nc['scalar'].isel(face=29),label='bc4')
ax.legend()
