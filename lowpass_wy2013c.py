"""
2018-04-05: Take the aggregated data generated in aggregate_wy2013c.py,
and perform the lowpass filter.
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

hydro_agg=waq.HydroFiles("hydro-wy2013c_adj-agg/com-wy2013c_adj_agg.hyd")

class Detide(waq.FilterAll):
    enable_write_symlink=False
    sparse_layers=False
    agg_boundaries=False

lp_secs=86400*3

def get_hydro():
    # Trying a 3 day cutoff
    return Detide(hydro_agg,lp_secs=lp_secs)

class Scen(waq.Scenario):
    name="wy2013c_adj_agg_lp"
    desc=('wy2013c_adj_agg_lp',
          'wy2013',
          'agg,lp')
    base_path='hydro-wy2013c_adj-agg_lp'

    def cmd_default(self):
        self.cmd_write_hydro()

def get_scen():        
    hydro=get_hydro()
    sec=datetime.timedelta(seconds=1)

    if 0:
        # short run for testing: start after some hydro spinup:
        start_time=hydro.time0+sec*hydro.t_secs[100]
        # and run for 1.5 days..
        stop_time=start_time + 4*24*3600*sec
    else:
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

    
