# Demo of increase boundary layer with ghost_layer_mesh
from mpi4py import MPI
from dolfinx import io, mesh

msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (2.0, 1.0)),
    n=(20, 20),
    ghost_mode=mesh.GhostMode.none,
    cell_type=mesh.CellType.quadrilateral,
)
with io.VTKFile(msh.comm, "msh0.pvd", "w") as file:
    file.write_mesh(msh)

msh = mesh.ghost_layer_mesh(msh)

with io.VTKFile(msh.comm, "msh1.pvd", "w") as file:
    file.write_mesh(msh)

msh = mesh.ghost_layer_mesh(msh)

with io.VTKFile(msh.comm, "msh2.pvd", "w") as file:
    file.write_mesh(msh)
