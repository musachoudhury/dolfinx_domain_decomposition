# poisson_hpddm.py
#
# Solve the Poisson problem  -Delta u = 1  on the unit square with
# homogeneous Dirichlet boundary conditions, discretized with P1 finite
# elements on a structured triangulation (each of the m x m grid cells is
# split into two triangles).
#
# The preconditioner is PCHPDDM (GenEO-type multilevel overlapping Schwarz).
# Because the global Pmat is an assembled MPIAIJ matrix, we must supply for
# each MPI process:
#   * an IS with the global indices of its overlapping subdomain, and
#   * the *unassembled* local operator on that subdomain, i.e. the "Neumann"
#     matrix: the sum of the element matrices of the elements inside the
#     overlapping subdomain only (no contributions from elements owned by
#     the neighbors are added at the subdomain interface, which is exactly
#     what "natural/Neumann boundary condition on the artificial interface"
#     means algebraically).
# These are passed with PCHPDDMSetAuxiliaryMat / pc.setHPDDMAuxiliaryMat().
#
# Requirements: PETSc configured with --download-hpddm --download-slepc
#               (and petsc4py, e.g. --with-petsc4py).
#
# Run, e.g.:
#   mpiexec -n 4 python poisson_hpddm.py -m 120 -overlap 1 \
#       -ksp_monitor -ksp_converged_reason -pc_hpddm_levels_1_eps_nev 15
#
# The domain decomposition here is a simple 1D strip decomposition in the
# y-direction (each rank owns a contiguous band of grid rows); this keeps
# the index bookkeeping short while still exercising the full PCHPDDM
# machinery (overlap, GenEO eigenproblems with the Neumann matrix, coarse
# space, ...).

import numpy as np
from petsc4py import PETSc


def set_default(opts, key, value):
    """Set an option only if the user did not already provide it."""
    if not opts.hasName(key):
        opts[key] = value


def main():
    comm = PETSc.COMM_WORLD
    rank = comm.getRank()
    size = comm.getSize()
    opts = PETSc.Options()

    m = opts.getInt("m", 80)  # number of cells per direction
    overlap = max(1, opts.getInt("overlap", 1))  # overlap (in element layers), >= 1

    if size > m:
        raise RuntimeError("Use at most m MPI processes (one grid row each)")

    h = 1.0 / m
    n_node_rows = m + 1  # node rows j = 0..m
    N = n_node_rows * n_node_rows  # global number of nodes

    def nid(i, j):
        """Global (lexicographic) index of node (i, j)."""
        return j * (m + 1) + i

    # ------------------------------------------------------------------
    # 1D strip decomposition: rank owns node rows [jstart, jend)
    # ------------------------------------------------------------------
    per = n_node_rows // size
    rem = n_node_rows % size
    jstart = rank * per + min(rank, rem)
    jend = jstart + per + (1 if rank < rem else 0)
    nlocal = (jend - jstart) * (m + 1)  # owned matrix rows

    # element row j lives between node rows j and j+1; assign it to the
    # owner of node row j  =>  every element row 0..m-1 has a unique owner
    estart, eend = jstart, min(jend, m)

    # P1 element stiffness matrix for a right triangle with legs of length h,
    # vertex order: (right-angle vertex, x-neighbor, y-neighbor).
    # (Independent of h in 2D.)
    Ke = 0.5 * np.array([[2.0, -1.0, -1.0], [-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
    fe = h * h / 6.0  # P1 load per vertex, f = 1

    def cell_triangles(i, j):
        """The two triangles of cell (i, j), right-angle vertex first."""
        n00 = nid(i, j)
        n10 = nid(i + 1, j)
        n01 = nid(i, j + 1)
        n11 = nid(i + 1, j + 1)
        return ([n00, n10, n01], [n11, n01, n10])

    # ------------------------------------------------------------------
    # Global (assembled, "Dirichlet") matrix A and right-hand side b
    # ------------------------------------------------------------------
    A = PETSc.Mat().create(comm=comm)
    A.setSizes(((nlocal, N), (nlocal, N)))
    A.setType(PETSc.Mat.Type.AIJ)
    A.setPreallocationNNZ((9, 9))
    A.setOption(PETSc.Mat.Option.SYMMETRIC, True)

    b = A.createVecRight()
    x = A.createVecRight()

    for j in range(estart, eend):
        for i in range(m):
            for tri in cell_triangles(i, j):
                A.setValues(tri, tri, Ke, addv=PETSc.InsertMode.ADD_VALUES)
                b.setValues(tri, [fe, fe, fe], addv=PETSc.InsertMode.ADD_VALUES)
    A.assemble()
    b.assemble()

    # homogeneous Dirichlet BC on the whole boundary: zero rows *and*
    # columns (keeps symmetry), put 1 on the diagonal, fix b accordingly
    bnd = []
    for j in range(jstart, jend):
        if j == 0 or j == m:
            bnd.extend(nid(i, j) for i in range(m + 1))
        else:
            bnd.extend((nid(0, j), nid(m, j)))
    is_bnd = PETSc.IS().createGeneral(np.asarray(bnd, dtype=PETSc.IntType), comm=comm)
    x.set(0.0)
    A.zeroRowsColumns(is_bnd, diag=1.0, x=x, b=b)
    is_bnd.destroy()

    # ------------------------------------------------------------------
    # Overlapping subdomain of this rank + local *Neumann* matrix on it
    # ------------------------------------------------------------------
    # element rows of the overlapping subdomain
    pel = max(0, estart - overlap)
    peh = min(m, eend + overlap)  # exclusive
    # node rows covered by those elements: [pel, peh + 1)
    nbase = pel * (m + 1)  # first global node index of the patch
    nloc = (peh + 1 - pel) * (m + 1)  # size of the local subdomain

    # IS with the global numbering of the subdomain unknowns; since the patch
    # is a contiguous block of node rows, this is just a stride. It must be
    # created on PETSC_COMM_SELF (it is a purely local object) and must
    # contain all rows owned by this process (it does, by construction).
    is_sub = PETSc.IS().createStride(nloc, first=nbase, step=1, comm=PETSc.COMM_SELF)

    # Unassembled local operator: sum element matrices of *local* elements
    # only. Nothing is added from the other side of the artificial interface
    # => natural (Neumann) boundary condition there.
    Aneu = PETSc.Mat().create(comm=PETSc.COMM_SELF)
    Aneu.setSizes((nloc, nloc))
    Aneu.setType(PETSc.Mat.Type.AIJ)  # could also be SBAIJ
    Aneu.setPreallocationNNZ(9)
    Aneu.setOption(PETSc.Mat.Option.SYMMETRIC, True)

    for j in range(pel, peh):
        for i in range(m):
            for tri in cell_triangles(i, j):
                loc = [g - nbase for g in tri]  # patch-local numbering
                Aneu.setValues(loc, loc, Ke, addv=PETSc.InsertMode.ADD_VALUES)
    Aneu.assemble()

    # The *physical* Dirichlet boundary must be treated in the Neumann matrix
    # exactly as in the global matrix (only the artificial interfaces get the
    # natural condition):
    lbnd = []
    for j in range(pel, peh + 1):
        if j == 0 or j == m:
            lbnd.extend(nid(i, j) - nbase for i in range(m + 1))
        else:
            lbnd.extend((nid(0, j) - nbase, nid(m, j) - nbase))
    is_lbnd = PETSc.IS().createGeneral(
        np.asarray(lbnd, dtype=PETSc.IntType), comm=PETSc.COMM_SELF
    )
    Aneu.zeroRowsColumns(is_lbnd, diag=1.0)
    is_lbnd.destroy()

    # ------------------------------------------------------------------
    # KSP + PCHPDDM
    # ------------------------------------------------------------------
    ksp = PETSc.KSP().create(comm=comm)
    ksp.setOperators(A)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.HPDDM)
    pc.setHPDDMAuxiliaryMat(is_sub, Aneu)  # <- the Neumann matrix
    pc.setHPDDMHasNeumannMat(True)  # it *is* the true Neumann matrix

    # sensible defaults, all overridable on the command line
    set_default(opts, "ksp_type", "fgmres")
    set_default(opts, "ksp_rtol", "1e-8")
    set_default(opts, "pc_hpddm_define_subdomains", "true")
    set_default(opts, "pc_hpddm_levels_1_eps_nev", "10")  # GenEO modes/subdomain
    set_default(opts, "pc_hpddm_levels_1_st_share_sub_ksp", "true")
    set_default(opts, "pc_hpddm_levels_1_sub_pc_type", "cholesky")
    set_default(opts, "pc_hpddm_levels_1_sub_pc_factor_shift_type", "inblocks")
    set_default(opts, "pc_hpddm_coarse_pc_type", "cholesky")
    ksp.setFromOptions()

    ksp.solve(b, x)

    its = ksp.getIterationNumber()
    r = b.duplicate()
    A.mult(x, r)
    r.axpy(-1.0, b)
    PETSc.Sys.Print(f"m = {m}, ndofs = {N}, {size} subdomain(s), overlap = {overlap}")
    PETSc.Sys.Print(
        f"converged reason = {ksp.getConvergedReason()}, "
        f"iterations = {its}, ||Ax-b|| = {r.norm():.3e}, "
        f"max(u) = {x.max()[1]:.6f}"
    )

    for obj in (r, x, b, A, Aneu, is_sub, ksp):
        obj.destroy()


if __name__ == "__main__":
    main()
