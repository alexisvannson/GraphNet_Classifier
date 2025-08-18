import numpy
import os

def _write_numpy2d_to_file_png(tab, filename):
    """Non applicable.
    """
    import matplotlib.pyplot
 
    imghsv = _transform_numpy2d_to_hsv(tab)
 
    fig, ax = matplotlib.pyplot.subplots()
    imgrgb = matplotlib.colors.hsv_to_rgb(imghsv)
    img = ax.imshow(imgrgb)
    ax.autoscale()
 
    (root, ext) = os.path.splitext(filename)
    if ext == '.bmp' or ext == '.jpg' or ext == '.png' or ext == 'gif' or ext == 'tiff':
        output_file = root + '_imsavegrey' + ext
        matplotlib.pyplot.imsave(output_file, tab, cmap='Greys')
 
        output_file = root + '_imsaverainbow' + ext
        matplotlib.pyplot.imsave(output_file, tab, cmap='rainbow')
        matplotlib.pyplot.close(fig)
 
    else:
        matplotlib.pyplot.show()
        matplotlib.pyplot.close(fig)
 
    return
 
 
def _transform_numpy2d_to_hsv(tab):
    """Non applicable.
    """
    imgval = numpy.zeros((tab.shape[0], tab.shape[1], 3), dtype=numpy.float64)
    for i in range(0, tab.shape[0]):
        for j in range(0, tab.shape[1]):
            if tab[i, j] == 1:
                imgval[i, j, 0] = 1  # h = elem2subdom/nsubdoms
                imgval[i, j, 1] = 1  # s
                imgval[i, j, 2] = 1  # v
            else:
                imgval[i, j, 0] = 1  # h
                imgval[i, j, 1] = 1  # s
                imgval[i, j, 2] = 0  # v
 
    return imgval
 
 
def _map_ab_to_cd(a, b, c, d, x):
    """Non applicable.
    """
    # Map $x \in [a,b]$ to $x \in [c,d]$.
    alpha = (c - d) / (a - b)
    beta = c - alpha * a
 
    return alpha * x + beta
 
 
def _transform_numpy2d_to_mrg(tab, selection):
    """Non applicable.
    """
    ny = tab.shape[0]
    nx = tab.shape[1]
 
    out_mesh = mesh4u._mesh.Mesh()
    out_mesh.spacedim = 3
    nnodes = (nx + 1) * (ny + 1)
    nodes_per_elem = 4
    if selection == None:
        nelems = nx * ny
    elif selection == 0:
        nelems = numpy.count_nonzero(tab==0)
    else:
        nelems = numpy.count_nonzero(tab)
    out_mesh.nelems = nelems
    out_mesh.p_elem2nodes = numpy.empty(out_mesh.nelems + 1, dtype=numpy.int64)
    out_mesh.p_elem2nodes[0] = 0
    for i in range(0, out_mesh.nelems):
        out_mesh.p_elem2nodes[i + 1] = out_mesh.p_elem2nodes[i] + nodes_per_elem
    out_mesh.elem2nodes = numpy.empty(out_mesh.nelems * nodes_per_elem, dtype=numpy.int64)
 
    ## nx*ny quad elements contains nodes: (i,j), (i+1,j), (i+1,j+1), (i,j+1)
    k = 0
    for j in range(0, ny):
        jj = (ny - 1) - j  ## (0,0) of numpy tab in top-left, (0,0) of mesh in bottom-left
        for i in range(0, nx):
            if selection == None or tab[jj, i] == selection:
                out_mesh.elem2nodes[k + 0] = j * (nx + 1) + i
                out_mesh.elem2nodes[k + 1] = j * (nx + 1) + i + 1
                out_mesh.elem2nodes[k + 2] = (j + 1) * (nx + 1) + i + 1
                out_mesh.elem2nodes[k + 3] = (j + 1) * (nx + 1) + i
                k += nodes_per_elem
    out_mesh.elem_type = numpy.empty(out_mesh.nelems, dtype=numpy.int64)
    out_mesh.elem_type[:] = mesh4u._mesh.VTK_QUAD
 
    # coordinates of (nx+1)*(ny+1) nodes of cartesian grid
    node_coords = numpy.empty((nnodes, out_mesh.spacedim), dtype=numpy.float64)
    xmin = 0
    xmax = nx
    ymin = 0
    ymax = ny
    xymax = max([xmax, ymax])
    xmax = _map_ab_to_cd(xmin, xmax, xmin, 1.*xmax/xymax, xmax)
    ymax = _map_ab_to_cd(ymin, ymax, ymin, 1.*ymax/xymax, ymax)
 
    k = 0
    for j in range(0, ny + 1):
        yy = ymin + (j * (ymax - ymin) / ny)
        for i in range(0, nx + 1):
            xx = xmin + (i * (xmax - xmin) / nx)
            node_coords[k, :] = xx, yy, 0.
            k += 1
 
    if selection == None:
        out_mesh.nnodes = (nx + 1) * (ny + 1)
        out_mesh.node_coords = numpy.empty((out_mesh.nnodes, out_mesh.spacedim), dtype=numpy.float64)
        for i in range(0, out_mesh.nnodes):
            out_mesh.node_coords[i, :] = node_coords[i, :]
        # local to global numbering
        out_mesh_l2g = numpy.arange(0, out_mesh.nnodes, 1, dtype=numpy.int64)
        out_mesh.node_l2g = out_mesh_l2g
 
    else:
        node_mask = numpy.zeros(nnodes, dtype=numpy.int64)
        node_mask[out_mesh.elem2nodes] = 1
        node_l2g = numpy.nonzero(node_mask)[0]
 
        # local to global numbering
        out_mesh.node_l2g = node_l2g
 
        out_mesh.nnodes = numpy.count_nonzero(node_mask)
        node_g2l = numpy.zeros(nnodes, dtype=numpy.int64)
        for i in range(0, out_mesh.nnodes):
            node_g2l[node_l2g[i]] = i
 
        for i in range(0, out_mesh.nelems):
            for ii in range(out_mesh.p_elem2nodes[i], out_mesh.p_elem2nodes[i+1]):
                out_mesh.elem2nodes[ii] = node_g2l[out_mesh.elem2nodes[ii]]
 
        out_mesh.node_coords = numpy.empty((out_mesh.nnodes, out_mesh.spacedim), dtype=numpy.float64)
        #out_mesh.node_coords = node_coords[node_g2l,:]
        for i in range(0, out_mesh.nnodes):
            out_mesh.node_coords[i, :] = node_coords[node_l2g[i], :]
 
    return out_mesh, out_mesh.node_l2g