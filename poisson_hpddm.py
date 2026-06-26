# Solving Poisson via Domain Decomposition in FEniCSx

# -- ASM (Addititve Schwarz Method) pc_asm_type basic
#    -- produces symettric preconditioner (R = R^T)
# -- RAS (Restricted Additive Schwarz) pc_asm_type restrict
#    -- produces nonsymmetric preconditioner (R != R^T)

# Run with mpirun -np 4 python poisson.py

from pathlib import Path

from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType  # type: ignore

import scipy.sparse as sp
import numpy as np

import ufl
from dolfinx import fem, io, mesh, plot
from dolfinx.fem.petsc import (
    assemble_matrix,
    assemble_vector,
    apply_lifting,
    set_bc,
    create_matrix,
)

msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (2.0, 1.0)),
    n=(10, 10),
    cell_type=mesh.CellType.quadrilateral,
)
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
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx + ufl.inner(g, v) * ufl.ds

# Compile forms
a_form = fem.form(a)
L_form = fem.form(L)

# Assemble matrix with BCs applied
A = assemble_matrix(a_form, bcs=[bc])
A.assemble()

# A_local = create_matrix(a_form, kind=PETSc.Mat.Type.SEQAIJ)

# im = V.dofmap.index_map
# bs = V.dofmap.index_map_bs
# n_local_ghosted = (im.size_local + im.num_ghosts) * bs

# A_local = PETSc.Mat().createAIJ(
#     size=((n_local_ghosted, n_local_ghosted), (n_local_ghosted, n_local_ghosted)),
#     comm=PETSc.COMM_SELF,
# )

# print(A_local.getSize())
# assemble_matrix(A_local, a_form, bcs=[])

# Assemble RHS vector with lifting for inhomogeneous BCs
b = assemble_vector(L_form)
apply_lifting(b, [a_form], bcs=[[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
set_bc(b, [bc])

# Set up the KSP solver explicitly
ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A)
ksp.setOptionsPrefix("demo_poisson_")
opts = PETSc.Options()
opts.prefixPush("demo_poisson_")
# opts["ksp_type"] = "cg"
# opts["pc_type"] = "asm"
# opts["pc_asm_type"] = "basic"
# opts["sub_pc_type"] = "cholesky"
opts["pc_hpddm_levels_1_eps_nev"] = 4
opts["pc_hpddm_levels_1_st_pc_type"] = "cholesky"
opts["pc_hpddm_levels_1_st_pc_type"] = "lu"
opts["pc_hpddm_levels_1_st_pc_factor_shift_type"] = "nonzero"

opts["ksp_error_if_not_converged"] = True
opts.prefixPop()
ksp.setFromOptions()


def monitor(ksp, its, rnorm):
    PETSc.Sys.Print(f"  iter {its:4d}  residual = {rnorm:.6e}", comm=msh.comm)


# ai, aj, av = A.getValuesCSR()
# M = sp.csr_matrix((av, aj, ai), shape=A.getLocalSize())  # full n x n
# arr = M.toarray()

# np.set_printoptions(linewidth=200, precision=3, suppress=True)
# print(arr)
# print(arr.shape)

# # Get the local-to-global column map (includes ghosts)
# lgmap = A.getLGMap()[1]  # column local-to-global mapping
# global_cols = lgmap.getIndices()  # global indices this rank sees (owned + ghost)

# # Build an IS of those global indices
# is_local = PETSc.IS().createGeneral(global_cols, comm=PETSc.COMM_SELF)


# # Extract the submatrix: local rows x (local+ghost cols)
# A_local = A.createSubMatrices(is_local, iscols=is_local)[0]
# A_local.zeroEntries()

# assemble_matrix(A_local, a_form, bcs=[])
# A_local.assemble()

A_matis = A.convert(PETSc.Mat.Type.IS)

A_local = A_matis.getISLocalMat()

lgmap = A_matis.getLGMap()[1]
global_cols = lgmap.getIndices()

is_local = PETSc.IS().createGeneral(global_cols, comm=PETSc.COMM_SELF)

# A_local.zeroEntries()
assemble_matrix(A_local, a_form, bcs=[])
A_local.assemble()

# A.zeroEntries()
# A = assemble_matrix(a_form, bcs=[bc])
# A.assemble()
# exit()
# A_local.view()


# ai, aj, av = A_local.getValuesCSR()
# M = sp.csr_matrix((av, aj, ai), shape=A_local.getLocalSize())  # full n x n
# arr = M.toarray()

# np.set_printoptions(linewidth=200, precision=3, suppress=True)
# print(arr)
# print(arr.shape)

pc = ksp.getPC()
pc.setType(PETSc.PC.Type.HPDDM)
pc.setOperators(A)  # global operator
pc.setHPDDMAuxiliaryMat(is_local, A_local)  # local IS + Neumann mat
pc.setHPDDMHasNeumannMat(True)  # tell it aux_mat is a true Neumann mat
pc.setFromOptions()
pc.setUp()

ksp.setType(PETSc.KSP.Type.GMRES)
ksp.setFromOptions()
ksp.setUp()

ksp.setMonitor(monitor)

# Solve
uh = fem.Function(V)
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
