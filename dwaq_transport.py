"""
Recreate a basic version of 2D transport from DWAQ output which has been filtered
aggregated, and depth-integrated.
"""

import os
import matplotlib.pyplot as plt

from stompy import utils
from stompy.grid import unstructured_grid
from stompy.spatial import field

import stompy.model.delft.waq_scenario as waq

## 

hydro=waq.HydroFiles('hydro-wy2013c_adj-agg_lp_2d/com-wy2013c_adj_agg_lp_2d.hyd')

##

g=hydro.grid()

# The original grid has some spatial relevance, but we need an even
# simpler topological representation, which is essentially the dual.
gd=g.create_dual()

##

N=gd.Nnodes()
bnds=hydro.read_bnd()
bnd_map={} # negative BC index => name
for bnd in bnds:
    for bc_link in bnd[1]['link']:
        bnd_map[bc_link]=bnd[0]

##

# Would like to do this for continuity, and for salinity
# as a test

#scal_name='continuity'
scal_name='salinity'
N_boundary=np.sum( hydro.pointers[:,0]<0)
boundary_C=np.zeros(N_boundary,np.float64) 

if scal_name=='continuity':
    boundary_C[:]=1.0 # for continuity
    scal=np.ones(N,np.float64)
elif scal_name=='salinity':
    boundary_C[:]=0.0 # freshwater sources
    for bc_elt in bnd_map:
        bc_idx=-bc_elt-1
        name=bnd_map[bc_elt]
        if 'Sea' in name:
            boundary_C[bc_idx]=34
    scal=np.ones(N,np.float64)
    
results=[scal]

# Even at this time scale, some of the ocean segments
# have negative values on the diagonal
# It's not clear at the moment whether volume should be at the new or old
# timestep - too complicated to work out here, need to go to pencil and paper.

times=hydro.t_secs[1000:1000+125*48:8]

for t_i in range(len(times)-1):
    # Set exchange matrix entries based on DWAQ data
    t0=times[t_i]
    t1=times[t_i+1]
    dt_s=t1-t0

    # Write in terms of concentrations
    M=np.eye(N,dtype=np.float64)

    J=np.zeros(N,np.float64)
    D=np.zeros(N,np.float64)

    flows=hydro.flows(t0)
    # vol=hydro.volumes(t0)
    vol=hydro.volumes(t1)

    for j,(seg_from,seg_to) in enumerate(hydro.pointers[:,:2]-1):
        if 1: # Advection, including boundaries
            # upwind advective flux
            if flows[j]>0:
                seg_src=seg_from # may be a boundary!
            else:
                seg_src=seg_to

            if seg_from>=0:
                assert seg_src>=0
                M[seg_from,seg_src] -= dt_s*flows[j] / vol[seg_from]

            if seg_src>=0:
                assert seg_to>=0
                M[seg_to,seg_src]   += dt_s*flows[j] / vol[seg_to]
            else:
                bc_idx=(seg_src+1) # just undoing the -1 from above
                # print("Boundary flux j=%d bc_idx=%d seg_src=%d name=%s"%(j,bc_idx,seg_src+1,bnd_map[bc_idx]))

                J[seg_to] += boundary_C[-bc_idx-1] * flows[j] / vol[seg_to]
        if 1: # Exchange
            if seg_from>=0:
                # Basic mixing:
                # HERE - adjust to take into account length scales, areas
                K=0.5 # fraction exchange per day of the smaller volume.
                dV=dt_s/86400. * K * min(vol[seg_from],vol[seg_to])
                
                M[seg_from,seg_from] -= dV / vol[seg_from]
                M[seg_to,seg_to]     -= dV / vol[seg_to]
                M[seg_from,seg_to] += dV / vol[seg_from]
                M[seg_to,seg_from] += dV / vol[seg_to]

    # Could this easily be made explicit or theta?
    # would help with stability
    if 1: # explicit
        # scal1=np.dot(M,scal0) + dt_s*J + D
        scal=np.dot(M,scal) + dt_s*J + D
    else:
        # implicit
        # scal1=np.dot(M,scal1) + dt_s*J + D
        # np.dot( (M-I), scal1 ) = -dt_s*J-D
        # HERE
        scal=np.linalg.solve( (M-np.eye(M.shape[0])), -dt_s*J-D )

    results.append(scal)
    

# #

plt.figure(1).clf()
fig,axs=plt.subplots(1,2,num=1)

ccoll0=g.plot_cells(values=results[0],ax=axs[0],cmap='jet')

ccoll1=g.plot_cells(values=results[-1],ax=axs[1],cmap='jet')

plt.colorbar(ccoll0,ax=axs[0])
plt.colorbar(ccoll1,ax=axs[1])


##

salt=hydro.parameters()['salinity']

plt.clf()
g.plot_cells(values=salt.evaluate(t=t).data)

## 

# Basic mixing:
K=0.2
dt=0.5
for j in gd.valid_edge_iter():
    n1,n2=gd.edges['nodes'][j]
    M[n1,n1] -= dt*K
    M[n2,n2] -= dt*K
    M[n2,n1] += dt*K
    M[n1,n2] += dt*K

for bc in bcs:
    bc.apply(M,J,D,t)

