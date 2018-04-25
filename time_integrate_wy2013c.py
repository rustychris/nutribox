"""
PLACEHOLDER -- this is being done for the moment on-demand in dwaq_transport.py

waq.TimeIntegrator has not been written yet.

2018-04-05: Take the aggregated, lowpassed, 2D data generated in depth_integrate_wy2013c.py,
and integrate daily.  
"""

import six
import os
import shutil
import datetime

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

import stompy.model.delft.waq_scenario as waq

##

hydro_in=waq.HydroFiles("hydro-wy2013c_adj-agg_lp_2d/com-wy2013c_adj_agg_lp_2d.hyd")

class TimeIntegrator(waq.TimeIntegrator):
    enable_write_symlink=False
    sparse_layers=False
    agg_boundaries=False

stride_secs=86400*1

def get_hydro():
    # Trying a 3 day cutoff
    return TimeIntegrator(hydro_in,stride_secs=stride_secs)

class Scen(waq.Scenario):
    name="wy2013c_adj_agg_lp"
    desc=('wy2013c_adj_agg_lp',
          'wy2013',
          'agg,lp,2d,daily')
    base_path='hydro-wy2013c_adj-agg_lp_2d_daily'

    def cmd_default(self):
        self.cmd_write_hydro()

def get_scen():        
    hydro=get_hydro()
    sec=datetime.timedelta(seconds=1)

    start_time=hydro.time0+hydro.t_secs[ 0]*sec
    stop_time =hydro.time0+hydro.t_secs[-1]*sec

    scen=Scen(hydro=hydro,
              start_time=start_time,
              stop_time=stop_time)
    return scen

if __name__=='__main__':
    scen=get_scen()
    os.path.exists(scen.base_path) and shutil.rmtree(scen.base_path)
    # assert not os.path.exists(scen.base_path)

    scen.cmd_default()

    
