import numpy as np

def add_msl_geometry(g,dem,msl=1.0):
    """
    g: UnstructuredGrid instance
    dem: field.SimpleGrid/GdalGrid instance
    msl: elevation to use as mean sea level relative to datum of dem.

    Adds dem_area, dem_volume, dem_depth as cell fields.
    """
    
    g.add_cell_field('dem_area',np.zeros(g.Ncells()),on_exists='overwrite')
    g.add_cell_field('dem_volume',np.zeros(g.Ncells()),on_exists='overwrite')

    pix_A=dem.dx * dem.dy

    for c in g.valid_cell_iter():
        poly=g.cell_polygon(c)
        dem_mask=(dem.F<msl) & dem.polygon_mask(poly) # => 2D bool array
        
        g.cells['dem_area'][c]   = pix_A*dem_mask.sum()
        g.cells['dem_volume'][c] = (msl-dem.F[dem_mask]).sum() * pix_A

    g.add_cell_field('dem_depth',g.cells['dem_volume'] / g.cells['dem_area'],
                     on_exists='overwrite')
