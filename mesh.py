from pathlib import Path

from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType  # type: ignore

import numpy as np

import ufl
from dolfinx import fem, io, mesh, plot
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc

msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (2.0, 1.0)),
    n=(32, 32),
    cell_type=mesh.CellType.triangle,
)


tdim = 2
gdim = 2

ncells = msh.topology.index_map(tdim).size_local
num_vertices = msh.topology.index_map(0).size_local
# print(idx_map.size_local)
print(ncells)
print(num_vertices)

vertex_destinations = msh.topology.index_map(0).index_to_dest_ranks(0)

c_to_v = msh.topology.connectivity(tdim, 0)

i = 0
print(vertex_destinations)
