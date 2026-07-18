import ufl
from basix.ufl import element
from mpi4py import MPI
import numpy as np
from dolfinx.mesh import create_unit_square, create_mesh, CellType, GhostMode
from dolfinx.graph import adjacencylist
from dolfinx.io import VTKFile
import dolfinx.cpp as cpp


def create_ghost_layer(mesh):
    tdim = mesh.topology.dim
    num_cells_local = mesh.topology.index_map(tdim).size_local

    tag = 12345
    vertex_destinations = mesh.topology.index_map(0).index_to_dest_ranks(tag)
    c_to_v = mesh.topology.connectivity(tdim, 0)

    rank = mesh.comm.rank

    dests = np.array([], dtype=np.int32)
    offsets = [0]

    for c in range(num_cells_local):
        dests = np.concatenate((dests, [rank]), dtype=np.int32)
        cdests = []
        for v in c_to_v.links(c):
            vdest = vertex_destinations.links(v)
            for d in vdest:
                cdests.append(d)
        cdests = np.unique(np.array(cdests, dtype=np.int32), sorted=True)
        dests = np.concatenate((dests, cdests))
        offsets.append(len(dests))

    offsets = np.array(offsets, dtype=np.int32)

    imap = mesh.geometry.index_map()
    num_vertices = imap.size_local
    x = mesh.geometry.x[:num_vertices, :]

    dofmap = mesh.geometry.dofmaps[0][:num_cells_local, :]
    dofmap_global = imap.local_to_global(dofmap.flatten()).reshape(dofmap.shape)


    def partitioner(comm, nparts, local_graph, num_ghost_nodes):
        return adjacencylist(dests, offsets)._cpp_object


    return create_mesh(mesh.comm, dofmap_global,
                       mesh.geometry.cmaps[0], x, partitioner)

mesh = create_unit_square(MPI.COMM_WORLD, 12, 12, ghost_mode=GhostMode.none)

r = VTKFile(mesh.comm, "mesh0.pvd", "w")
r.write_mesh(mesh)
r.close()

for i in range(1, 4):
    mesh = create_ghost_layer(mesh)
    r = VTKFile(mesh.comm, f"mesh{i}.pvd", "w")
    r.write_mesh(mesh)
    r.close()
