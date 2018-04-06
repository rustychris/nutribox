"""
2018-04-05: Take the aggregated data generated in aggregate_wy2013c.py,
and perform the lowpass, depth integration, and daily integration to 
get compact subtidal, 2D transport.
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

six.moves.reload_module(waq)

hydro_lp=waq.HydroFiles(hyd_path='hydro-wy2013c_adj-agg_lp/com-wy2013c_adj_agg_lp.hyd')

# Depth-integrate

class DepthAggregator(waq.HydroAggregator):
    
    def __init__(self,**kwargs):
        super(DepthAggregator,self).__init__(merge_only=True,**kwargs)

    def find_maxima(self):
        super(DepthAggregator,self).find_maxima()
        self.n_agg_layers=1
        
    def get_agg_k(self,proc,k,seg):
        return 0 # all layers to 1
        
hydro=DepthAggregator(hydro_in=hydro_lp)

class Scen(waq.Scenario):
    name="wy2013c_adj_agg_lp_2d"
    desc=('wy2013c_adj_agg_lp_2d',
          'wy2013',
          'agg,lp,2d')
    base_path='hydro-wy2013c_adj-agg_lp_2d'

    def cmd_default(self):
        self.cmd_write_hydro()

sec=datetime.timedelta(seconds=1)

start_time=hydro.time0+hydro.t_secs[ 0]*sec
stop_time =hydro.time0+hydro.t_secs[-1]*sec

scen=Scen(hydro=hydro,
          start_time=start_time,
          stop_time=stop_time)

# os.path.exists(scen.base_path) and shutil.rmtree(scen.base_path)
assert not os.path.exists(scen.base_path)

scen.cmd_default()



        
    
    