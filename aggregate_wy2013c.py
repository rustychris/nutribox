"""
2018-04-05:  Revisiting aggregated model using wy2013c DFM hydro output (adjusted for
continuity issues).

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

hydro_unagg=waq.HydroFiles("/opt/data/delft/sfb_dfm_v2/runs/wy2013c/DFM_DELWAQ_wy2013c_adj/wy2013c.hyd")

class Aggregate(waq.HydroAggregator):
    enable_write_symlink=False
    sparse_layers=False
    agg_boundaries=False

agg_shp='boxes-v02.shp'

def get_hydro():
    return Aggregate(hydro_in=hydro_unagg,
                     agg_shp=agg_shp)

class Scen(waq.Scenario):
    name="wy2013c_adj_agg"
    desc=('wy2013c_adj_agg',
          'wy2013',
          'agg')
    base_path='hydro-wy2013c_adj-agg'

    def cmd_default(self):
        self.cmd_write_hydro()

sec=datetime.timedelta(seconds=1)

def get_scen():
    if 0:
        # short run for testing: start after some hydro spinup:
        start_time=hydro.time0+sec*hydro.t_secs[100]
        # and run for 1.5 days..
        stop_time=start_time + 4*24*3600*sec
        Scen.base_path+='_short'
    else:
        start_time=hydro.time0+hydro.t_secs[ 0]*sec
        stop_time =hydro.time0+hydro.t_secs[-1]*sec

    hydro=get_hydro()
    scen=Scen(hydro=hydro,
              start_time=start_time,
              stop_time=stop_time)

if __name__=='__main__':
    scen=get_scen()

    # during dev:
    os.path.exists(scen.base_path) and shutil.rmtree(scen.base_path)
    # safer
    # assert not os.path.exists(scen.base_path)

    scen.cmd_default()

