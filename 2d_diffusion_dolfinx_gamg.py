from pathlib import Path

from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType  # type: ignore

import numpy as np

import ufl
from dolfinx import fem, io, mesh, plot
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc

# Global unit-square mesh, matching FreeFem's square(40, 40)
msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (1.0, 1.0)),
    n=(600, 600),
    cell_type=mesh.CellType.triangle,
)
V = fem.functionspace(msh, ("Lagrange", 1))  # P1

tdim = msh.topology.dim
fdim = tdim - 1

# FreeFem applies on(1, u=0): boundary label 1 of square() is the bottom edge y=0
facets = mesh.locate_entities_boundary(
    msh,
    dim=fdim,
    marker=lambda x: np.isclose(x[1], 0.0),
)

dofs = fem.locate_dofs_topological(V=V, entity_dim=fdim, entities=facets)
bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# vPb(u, v) = int2d(grad(u)'*grad(v)) + int2d(v)
# -> stiffness + constant unit source f = 1, no Neumann term
f = fem.Constant(msh, ScalarType(1.0))
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx

# Compile forms
a_form = fem.form(a)
L_form = fem.form(L)

# Assemble matrix with BCs applied
A = assemble_matrix(a_form, bcs=[bc])
A.assemble()

# nullspace_vec = A.createVecLeft()
# nullspace_vec.set(1.0)
# nullspace_vec.normalize()
# nsp = PETSc.NullSpace().create(vectors=[nullspace_vec], comm=msh.comm)

# A.setNearNullSpace(nsp)

# A_matis = A.convert(PETSc.Mat.Type.IS)

# A_local = A_matis.getISLocalMat()

# lgmap = A_matis.getLGMap()[1]
# global_cols = lgmap.getIndices()

# is_local = PETSc.IS().createGeneral(global_cols, comm=PETSc.COMM_SELF)

# # A_local.zeroEntries()
# assemble_matrix(A_local, a_form, bcs=[])
# A_local.assemble()

# Assemble RHS vector with lifting (homogeneous here, but kept for correctness)
b = assemble_vector(L_form)
apply_lifting(b, [a_form], bcs=[[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
set_bc(b, [bc])

# Direct solve (LU), matching the A^-1 solve in the FreeFem script
ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A)
opts = PETSc.Options()


# pc = ksp.getPC()
# pc.setType(PETSc.PC.Type.GAMG)
# pc.setOperators(A)  # global operator
# pc.setHPDDMAuxiliaryMat(is_local, A_local)  # local IS + Neumann mat
# pc.setHPDDMHasNeumannMat(True)  # tell it aux_mat is a true Neumann mat

# opts["pc_hpddm_levels_1_sub_pc_type"] = "cholesky"
# opts["pc_hpddm_levels_1_sub_pc_factor_mat_solver_type"] = "mumps"
# opts["pc_hpddm_has_neumann"] = None
# opts["pc_hpddm_define_subdomains"] = None
# opts["pc_hpddm_levels_1_pc_asm_type"] = "basic"
# opts["pc_hpddm_coarse_correction"] = "additive"
# opts["pc_hpddm_levels_1_sub_mat_mumps_cntl_4"] = 0.1
# opts["pc_hpddm_levels_1_eps_use_inertia"] = None
# opts["pc_hpddm_levels_1_eps_threshold"] = 0.5
# opts["pc_hpddm_levels_1_st_share_sub_ksp"] = None
# opts["ksp_type"] = "cg"
# opts["ksp_monitor_singular_value"] = None

# opts["pc_hpddm_levels_1_eps_nev"] = 10
# opts["pc_hpddm_coarse_pc_type"] = "asm"

# opts["pc_hpddm_levels_1_pc_type"] = "asm"

# pc.setFromOptions()
# pc.setUp()

ksp.setType(PETSc.KSP.Type.CG)
ksp.getPC().setType(PETSc.PC.Type.GAMG)

ksp.setFromOptions()
ksp.setUp()


def monitor(ksp, its, rnorm):
    PETSc.Sys.Print(f"  iter {its:4d}  residual = {rnorm:.6e}", comm=msh.comm)


ksp.setMonitor(monitor)

# Solve
uh = fem.Function(V)
ksp.solve(b, uh.x.petsc_vec)
uh.x.scatter_forward()

ksp.view()

out_folder = Path("out_diffusion")
out_folder.mkdir(parents=True, exist_ok=True)
with io.XDMFFile(msh.comm, out_folder / "diffusion.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(uh)
