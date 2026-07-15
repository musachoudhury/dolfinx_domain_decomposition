# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# # Elasticity using BDDC precondtioner
#
# Copyright © 2020-2022 Garth N. Wells and Michal Habera
#
# ```{admonition} Download sources
# :class: download
# * {download}`Python script <./demo_elasticity.py>`
# * {download}`Jupyter notebook <./demo_elasticity.ipynb>`
# ```
# This demo solves the equations of static linear elasticity using
# a smoothed aggregation algebraic multigrid solver.
# It illustrates how to:
# - Use a smoothed aggregation algebraic multigrid solver
# - Use {py:class}`Expression <dolfinx.fem.Expression>` to compute
#   derived quantities of a solution
#
# The required modules are first imported:


# +
from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
from numpy import typing as npt
import ufl
from dolfinx import la
from dolfinx.fem import (
    Expression,
    Function,
    FunctionSpace,
    dirichletbc,
    form,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import apply_lifting, assemble_matrix, assemble_vector
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, GhostMode, create_box, locate_entities_boundary
from dolfinx import fem

from utils import _dm_create_matrix, _dm_create_field_decomposition
import functools

dtype = PETSc.ScalarType
# -

# ## Create the operator near-nullspace
#
# Smooth aggregation algebraic multigrid solvers require the so-called
# 'near-nullspace', which is the nullspace of the operator in the
# absence of boundary conditions. The below function builds a
# `PETSc.NullSpace` object for a 3D elasticity problem. The nullspace is
# spanned by six vectors -- three translation modes and three rotation
# modes.


def build_nullspace(V: FunctionSpace):
    """Build PETSc nullspace for 3D elasticity."""
    # Create vectors that will span the nullspace
    bs = V.dofmap.index_map_bs
    length0 = V.dofmap.index_map.size_local
    basis = [la.vector(V.dofmap.index_map, bs=bs, dtype=dtype) for i in range(6)]
    b = [b.array for b in basis]

    # Get dof indices for each subspace (x, y and z dofs)
    dofs = [V.sub(i).dofmap.list.flatten() for i in range(3)]

    # Set the three translational rigid body modes
    for i in range(3):
        b[i][dofs[i]] = 1.0

    # Set the three rotational rigid body modes
    x = V.tabulate_dof_coordinates()
    dofs_block = V.dofmap.list.flatten()
    x0, x1, x2 = x[dofs_block, 0], x[dofs_block, 1], x[dofs_block, 2]
    b[3][dofs[0]] = -x1
    b[3][dofs[1]] = x0
    b[4][dofs[0]] = x2
    b[4][dofs[2]] = -x0
    b[5][dofs[2]] = x1
    b[5][dofs[1]] = -x2

    la.orthonormalize(basis)

    basis_petsc = [
        PETSc.Vec().createWithArray(x[: bs * length0], bsize=3, comm=V.mesh.comm)
        for x in b
    ]
    return PETSc.NullSpace().create(vectors=basis_petsc)


# ## Problem definition

# Create a {py:func}`box mesh<dolfinx.mesh.create_box>`:


msh = create_box(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0, 0.0]), np.array([2.0, 1.0, 1.0])],
    (4, 4, 4),
    CellType.tetrahedron,
    ghost_mode=GhostMode.none,
)

# Create a centripetal source term $f = \rho \omega^2 [x_0, \, x_1]$:

ω, ρ = 300.0, 10.0
x = ufl.SpatialCoordinate(msh)
f = ufl.as_vector((ρ * ω**2 * x[0], ρ * ω**2 * x[1], 0.0))

# Define the elasticity parameters and create a function that computes
# an expression for the stress given a displacement field.

# +
E = 1.0e9
ν = 0.3
μ = E / (2.0 * (1.0 + ν))
λ = E * ν / ((1.0 + ν) * (1.0 - 2.0 * ν))


def σ(v):
    """Return an expression for the stress σ given a displacement field."""
    return 2.0 * μ * ufl.sym(ufl.grad(v)) + λ * ufl.tr(
        ufl.sym(ufl.grad(v))
    ) * ufl.Identity(len(v))


# -

# A function space is created and the elasticity variational
# problem defined:

gdim = msh.geometry.dim
V = functionspace(msh, ("Lagrange", 1, (gdim,)))
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

a = form(ufl.inner(σ(u), ufl.grad(v)) * ufl.dx)
L = form(ufl.inner(f, v) * ufl.dx)

# A homogeneous (zero) boundary condition is created on $x_0 = 0$ and
# $x_1 = 1$ by finding all facets on these boundaries, and then creating
# a Dirichlet boundary condition object.
tdim = msh.topology.dim
fdim = tdim - 1
facets = locate_entities_boundary(
    msh, dim=fdim, marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[1], 1.0)
)
bc = dirichletbc(
    np.zeros(gdim, dtype=dtype),
    locate_dofs_topological(V, entity_dim=fdim, entities=facets),
    V=V,
)

uh = Function(V)

# Linear problem


# ksp = problem.solver
# ksp.setMonitor(
#     lambda _, its, rnorm: PETSc.Sys.Print(
#         f"  iteration: {its:>4d}, residual: {rnorm:.3e}"
#     )
# )

# problem.solve()

# converged_reason = problem.solver.getConvergedReason()
# assert converged_reason > 0, (
#     f"Krylov solver for has not converged, reason: {converged_reason}."
# )

# ## Assemble and solve
#
# The bilinear form `a` is assembled into a matrix `A`, with
# modifications for the Dirichlet boundary conditions. The call
# `A.assemble()` completes any parallel communication required to
# compute the matrix.

# +

A = fem.petsc.create_matrix(a, kind="is")
A.zeroEntries()
assemble_matrix(A, a, bcs=[bc])  # type: ignore[arg-type, misc]
A.assemble()
# -

# The linear form `L` is assembled into a vector `b`, and then modified
# by {py:func}`apply_lifting <dolfinx.fem.petsc.apply_lifting>` to
# account for the Dirichlet boundary conditions. After calling
# {py:func}`apply_lifting <dolfinx.fem.petsc.apply_lifting>`, the method
# `ghostUpdate` accumulates entries on the owning rank, and this is
# followed by setting the boundary values in `b`.

# +
b = assemble_vector(L)
apply_lifting(b, [a], bcs=[[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
bc.set(b.array_w)
# -

# Create the near-nullspace and attach it to the PETSc matrix:

# ns = build_nullspace(V)
# A.setNearNullSpace(ns)
# A.setOption(PETSc.Mat.Option.SPD, True)

# Set PETSc solver options, create a PETSc Krylov solver, and attach the
# matrix `A` to the solver:

# +
# Set solver options
opts = PETSc.Options()
opts["ksp_rtol"] = 1e-8
opts["ksp_view"] = None

# local_solver = "lu"
# local_solver_type = "superlu"
# coarse_solver = "lu"
# coarse_solver_type = "superlu_dist"

local_solver = "lu"
local_solver_type = "mumps"
coarse_solver = "lu"
coarse_solver_type = "mumps"

# opts["ksp_type"] = "gmres"
opts["pc_type"] = "bddc"
opts["pc_bddc_use_local_mat_graph"] = False
opts["pc_bddc_benign_trick"] = None
opts["pc_bddc_nonetflux"] = None
opts["pc_bddc_detect_disconnected"] = None
opts["pc_bddc_dirichlet_pc_type"] = local_solver
opts["pc_bddc_dirichlet_pc_factor_mat_solver_type"] = local_solver_type
opts["pc_bddc_neumann_pc_type"] = local_solver
opts["pc_bddc_neumann_pc_factor_mat_solver_type"] = local_solver_type
opts["pc_bddc_coarse_pc_type"] = coarse_solver
opts["pc_bddc_coarse_pc_factor_mat_solver_type"] = coarse_solver_type

# opts["pc_bddc_dirichlet_mat_mumps_icntl_14"] = 200
# opts["pc_bddc_neumann_mat_mumps_icntl_14"] = 200
# opts["pc_bddc_coarse_mat_mumps_icntl_14"] = 200

opts["ksp_error_if_not_converged"] = None

# Create PETSc Krylov solver and turn convergence monitoring on
solver = PETSc.KSP().create(msh.comm)
solver.setFromOptions()

# Set matrix operator
solver.setOperators(A)

# Attach problem information
dm = solver.getDM()
dm.setCreateMatrix(functools.partial(_dm_create_matrix, A))
dm.setCreateFieldDecomposition(functools.partial(_dm_create_field_decomposition, uh, L))
solver.getPC().setDM(dm)
# -

# Create a solution {py:class}`Function<dolfinx.fem.Function>` `uh` and
# solve:

#

# Set a monitor, solve linear system, and display the solver
# configuration
solver.setMonitor(
    lambda _, its, rnorm: print(f"Iteration: {its}, rel. residual: {rnorm}")
)
solver.solve(b, uh.x.petsc_vec)
solver.view()

converged_reason = solver.getConvergedReason()
print(converged_reason)
# Scatter forward the solution vector to update ghost values
uh.x.scatter_forward()
# -

# ## Post-processing
#
# The computed solution is now post-processed. Expressions for the
# deviatoric and Von Mises stress are defined:

# +
sigma_dev = σ(uh) - (1 / 3) * ufl.tr(σ(uh)) * ufl.Identity(len(uh))
sigma_vm = ufl.sqrt((3 / 2) * ufl.inner(sigma_dev, sigma_dev))
# -

# Next, the Von Mises stress is interpolated in a piecewise-constant
# space by creating an {py:class}`Expression<dolfinx.fem.Expression>`
# that is interpolated into the
# {py:class}`Function<dolfinx.fem.Function>` `sigma_vm_h`.

# +
W = functionspace(msh, ("Discontinuous Lagrange", 0))
sigma_vm_expr = Expression(sigma_vm, W.element.interpolation_points)
sigma_vm_h = Function(W)
sigma_vm_h.interpolate(sigma_vm_expr)
# -

# Save displacement field `uh` and the Von Mises stress `sigma_vm_h` in
# XDMF format files.

# +
with XDMFFile(msh.comm, "out_elasticity/displacements.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(uh)

# Save solution to XDMF format
with XDMFFile(msh.comm, "out_elasticity/von_mises_stress.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(sigma_vm_h)
# -

# Finally, we compute the $L^2$ norm of the displacement solution
# vector. This is a collective operation (i.e., the method
# {py:func}`norm<dolfinx.la.norm>` must be called from all MPI ranks),
# but we print the norm only on rank 0.

# +
unorm = la.norm(uh.x)
if msh.comm.rank == 0:
    print("Solution vector norm:", unorm)
# -

# The solution vector norm can be a useful check that the solver is
# computing the same result when running in serial and in parallel.
