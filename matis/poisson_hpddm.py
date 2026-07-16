# Solving Poisson via HPDDM in FEniCSx

# Run with mpirun -np 4 python poisson.py

from pathlib import Path

from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType  # type: ignore

import scipy.sparse as sp
import numpy as np
import functools

import ufl
from dolfinx import fem, io, mesh, plot
from dolfinx.fem.petsc import (
    assemble_matrix,
    assemble_vector,
    apply_lifting,
    set_bc,
    create_matrix,
)
from dolfinx import cpp
from utils import _dm_create_matrix, _dm_create_field_decomposition

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (2.0, 1.0)),
    n=(32, 32),
    ghost_mode=mesh.GhostMode.none,
    cell_type=mesh.CellType.quadrilateral,
)

for i in range(2):
    msh = mesh.ghost_layer_mesh(msh)

V = fem.functionspace(msh, ("Lagrange", 1))

tdim = msh.topology.dim
fdim = tdim - 1
facets = mesh.locate_entities_boundary(
    msh,
    dim=fdim,
    marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 2.0),
)

dofs = fem.locate_dofs_topological(V=V, entity_dim=fdim, entities=facets)

bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(msh)
f = 10 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
g = ufl.sin(5 * x[0])

# Must explicitly assemble over local + ghost cells as default behaviour
# assembled over local cells
# HPDDM wants a Neumann matrix that has been assembled over ghost cells as well.

num_cells_local = msh.topology.index_map(tdim).size_local
num_ghost_cells = msh.topology.index_map(tdim).num_ghosts
all_cells = np.arange(num_cells_local + num_ghost_cells, dtype=np.int32)
dx_is = ufl.Measure("dx", domain=msh, subdomain_data=[(1, all_cells)])

a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
a_is = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_is(1)
L = ufl.inner(f, v) * ufl.dx + ufl.inner(g, v) * ufl.ds

# Compile forms
a_form = fem.form(a_is)
a_lift_form = fem.form(a)
L_form = fem.form(L)

# Assemble matrix with BCs applied
A = assemble_matrix(a_form, bcs=[bc], kind="is")
A.assemble()

# Assemble RHS vector with lifting for inhomogeneous BCs
b = assemble_vector(L_form)
apply_lifting(b, [a_lift_form], bcs=[[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
set_bc(b, [bc])

# Set up the KSP solver explicitly
ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A)
ksp.setOptionsPrefix("demo_poisson_")
opts = PETSc.Options()
opts.prefixPush("demo_poisson_")

opts["pc_hpddm_levels_1_sub_pc_type"] = "lu"
opts["pc_hpddm_levels_1_eps_nev"] = 10
opts["pc_hpddm_levels_2_p"] = 1
opts["pc_hpddm_levels_2_sub_pc_type"] = "lu"
opts["pc_hpddm_levels_2_eps_nev"] = 10
opts["pc_hpddm_coarse_p"] = 1
opts["pc_hpddm_coarse_mat_type"] = "baij"

opts["ksp_error_if_not_converged"] = True
opts.prefixPop()
ksp.setFromOptions()


def monitor(ksp, its, rnorm):
    PETSc.Sys.Print(f"  iter {its:4d}  residual = {rnorm:.6e}", comm=msh.comm)


pc = ksp.getPC()
pc.setType(PETSc.PC.Type.HPDDM)
pc.setOperators(A)  # global operator

# No need as A is a IS matrix
# pc.setHPDDMAuxiliaryMat(is_local, A_local)  # local IS + Neumann mat
# pc.setHPDDMHasNeumannMat(True)  # tell it aux_mat is a true Neumann mat

pc.setFromOptions()
pc.setUp()

ksp.setType(PETSc.KSP.Type.GMRES)
ksp.setFromOptions()
ksp.setUp()

ksp.setMonitor(monitor)

uh = fem.Function(V)

# Attach PETSc DM
dm = ksp.getDM()
dm.setCreateMatrix(functools.partial(_dm_create_matrix, A))
dm.setCreateFieldDecomposition(functools.partial(_dm_create_field_decomposition, uh, L))
ksp.getPC().setDM(dm)

# Solve
ksp.solve(b, uh.x.petsc_vec)
uh.x.scatter_forward()

# ksp.view()


out_folder = Path("out_poisson")
out_folder.mkdir(parents=True, exist_ok=True)
with io.XDMFFile(msh.comm, out_folder / "poisson.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(uh)

# try:
#     import pyvista

#     cells, types, x = plot.vtk_mesh(V)
#     grid = pyvista.UnstructuredGrid(cells, types, x)
#     grid.point_data["u"] = uh.x.array.real
#     grid.set_active_scalars("u")
#     plotter = pyvista.Plotter()
#     plotter.add_mesh(grid, show_edges=True)
#     warped = grid.warp_by_scalar()
#     plotter.add_mesh(warped)
#     if pyvista.OFF_SCREEN:
#         plotter.screenshot(out_folder / "uh_poisson.png")
#     else:
#         plotter.show()
# except ModuleNotFoundError:
#     print("'pyvista' is required to visualise the solution.")
#     print("To install pyvista with pip: 'python3 -m pip install pyvista'.")
