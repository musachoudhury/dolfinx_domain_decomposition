
#pragma once

#include <basix/finite-element.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/utils.h>
#include <map>
#include <span>

template <std::floating_point T>
void print_mesh_details(dolfinx::mesh::Mesh<T> &mesh) {
  MPI_Comm comm = mesh.comm();
  std::size_t size = dolfinx::MPI::size(comm);
  std::size_t rank = dolfinx::MPI::rank(comm);
  const std::size_t tdim = mesh.topology()->dim();
  // constexpr int gdim = 2;
  std::size_t ncells = mesh.topology()->index_map(tdim)->size_local();
  // std::size_t num_vertices = mesh.topology()->index_map(0)->size_local();

  // Find which local vertices are ghosted elsewhere
  auto vertex_destinations =
      mesh.topology()->index_map(0)->index_to_dest_ranks();

  // cell to vertex AdjacencyList
  auto c_to_v = mesh.topology()->connectivity(tdim, 0);

  // Loops over cells
  for (std::size_t r = 0; r < size; r++) {
    if (rank == r) {
      std::cout << "Rank " << rank << std::endl;
      for (std::size_t c = 0; c < ncells; ++c) {
        std::cout << "Cell " << c << ", vertex: ";
        for (auto v : c_to_v->links(c)) {
          {
            std::cout << v;
            auto vdest = vertex_destinations.links(v);
            for (int dest : vdest)
              std::cout << "(" << dest << ")";
            std::cout << ", ";
          }
        }
        std::cout << "\n";
      }
      std::cout << "*********************************" << std::endl;
    }

    MPI_Barrier(comm);
  }
}

/// @brief Create a new mesh with an extra boundary layer, such that all
/// cells
/// on other processes which share a vertex with this process are ghosted.
/// @param mesh Input mesh
/// @param coord_element A coordinate element for the new mesh. This may be
/// tensor product ordering.

template <std::floating_point T>
dolfinx::mesh::Mesh<T> ghost_layer_mesh(
    dolfinx::mesh::Mesh<T> &mesh,
    dolfinx::fem::CoordinateElement<T> coord_element,
    const std::function<std::vector<std::int32_t>(
        const dolfinx::graph::AdjacencyList<std::int32_t> &)> &reorder_fn =
        dolfinx::graph::reorder_gps) {

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const std::size_t tdim = mesh.topology()->dim();
  // const std::size_t gdim = mesh.geometry().dim();
  std::size_t ncells = mesh.topology()->index_map(tdim)->size_local();
  std::size_t num_vertices = mesh.topology()->index_map(0)->size_local();

  // Find which local vertices are ghosted elsewhere
  auto vertex_destinations =
      mesh.topology()->index_map(0)->index_to_dest_ranks();

  // cell to vertex AdjacencyList
  auto c_to_v = mesh.topology()->connectivity(tdim, 0);

  print_mesh_details(mesh);

  // Map from any local cells to processes where they should be ghosted
  std::map<int, std::vector<int>> cell_to_dests;

  // Loops over cells
  // Assigns destinations to each cell
  std::vector<int> cdests;

  // MPI_Barrier(MPI_COMM_WORLD);

  for (std::size_t c = 0; c < ncells; ++c) {
    cdests.clear();
    for (auto v : c_to_v->links(c)) {
      auto vdest = vertex_destinations.links(v);
      for (int dest : vdest)
        cdests.push_back(dest);
    }
    std::sort(cdests.begin(), cdests.end());
    cdests.erase(std::unique(cdests.begin(), cdests.end()), cdests.end());
    if (!cdests.empty())
      cell_to_dests[c] = cdests;
  }

  // for (std::size_t c = 0; c < ncells; ++c) {
  //   if (!cdests.empty())
  //     cell_to_dests[c] = {rank};
  // }
  spdlog::info("cell_to_dests= {}, ncells = {}", cell_to_dests.size(), ncells);

  // Builds a cell-to-rank adjacency list
  auto partitioner = [cell_to_dests, ncells](
                         MPI_Comm comm, int /*nparts*/,
                         const std::vector<dolfinx::mesh::CellType> &,
                         const std::vector<std::span<const std::int64_t>> &) {
    int rank = dolfinx::MPI::rank(comm);
    std::vector<std::int32_t> dests;
    std::vector<std::int32_t> offsets = {0};
    for (std::size_t c = 0; c < ncells; ++c) {
      dests.push_back(rank);
      if (auto it = cell_to_dests.find(c); it != cell_to_dests.end())
        dests.insert(dests.end(), it->second.begin(), it->second.end());

      // Ghost to other processes
      offsets.push_back(dests.size());
    }
    return dolfinx::graph::AdjacencyList<std::int32_t>(std::move(dests),
                                                       std::move(offsets));
  };

  // FIXME: mesh.geometry().x().data() always stored with 3 components per
  // points
  // Bugged version:
  // std::array<std::size_t, 2> xshape = {num_vertices, gdim};
  //
  std::array<std::size_t, 2> xshape = {num_vertices, 3};
  std::span<T> x(mesh.geometry().x().data(), xshape[0] * xshape[1]);

  auto dofmap = mesh.geometry().dofmaps().front();
  auto imap = mesh.geometry().index_map();
  std::vector<std::int32_t> permuted_dofmap;
  std::optional<std::vector<int>> perm = basix::lex_dof_ordering(
      basix::element::family::P,
      dolfinx::mesh::cell_type_to_basix_type(coord_element.cell_shape()),
      coord_element.degree(), coord_element.variant(),
      basix::element::dpc_variant::unset, false);

  for (std::size_t c = 0; c < dofmap.extent(0); ++c) {
    auto cell_dofs = std::submdspan(dofmap, c, std::full_extent);
    for (std::size_t i = 0; i < dofmap.extent(1); ++i) {
      permuted_dofmap.push_back(cell_dofs(perm.value()[i]));
    }
  }

  std::vector<std::int64_t> permuted_dofmap_global(permuted_dofmap.size());
  imap->local_to_global(permuted_dofmap, permuted_dofmap_global);

  std::optional<std::int32_t> max_facet_to_cell_links; // = 2;
  auto new_mesh = dolfinx::mesh::create_mesh(
      mesh.comm(), mesh.comm(), std::span(permuted_dofmap_global),
      coord_element, mesh.comm(), x, xshape, partitioner,
      max_facet_to_cell_links, reorder_fn);

  print_mesh_details(new_mesh);

  spdlog::info("** NEW MESH num_ghosts_cells = {}",
               new_mesh.topology()->index_map(tdim)->num_ghosts());
  spdlog::info("** NEW MESH num_local_cells = {}",
               new_mesh.topology()->index_map(tdim)->size_local());

  return new_mesh;
}

/// @brief Compute two lists of cell indices:
/// 1. cells which are "local", i.e. the dofs on
/// these cells are not shared with any other process.
/// 2. cells which share dofs with other processes.
///
template <typename T>
std::pair<std::vector<std::int32_t>, std::vector<std::int32_t>>
compute_boundary_cells(std::shared_ptr<dolfinx::fem::FunctionSpace<T>> V) {
  auto mesh = V->mesh();
  auto topology = mesh->topology_mutable();
  int tdim = topology->dim();
  int fdim = tdim - 1;
  topology->create_connectivity(fdim, tdim);

  int ncells_local = topology->index_map(tdim)->size_local();
  int ncells_ghost = topology->index_map(tdim)->num_ghosts();
  int ndofs_local = V->dofmap()->index_map->size_local();

  std::vector<std::uint8_t> cell_mark(ncells_local + ncells_ghost, 0);
  for (int i = 0; i < ncells_local; ++i) {
    auto cell_dofs = V->dofmap()->cell_dofs(i);
    for (auto dof : cell_dofs)
      if (dof >= ndofs_local)
        cell_mark[i] = 1;
  }
  for (int i = ncells_local; i < ncells_local + ncells_ghost; ++i)
    cell_mark[i] = 1;

  std::vector<int> local_cells;
  std::vector<int> boundary_cells;
  for (std::size_t i = 0; i < cell_mark.size(); ++i) {
    if (cell_mark[i])
      boundary_cells.push_back(i);
    else
      local_cells.push_back(i);
  }

  spdlog::debug("lcells:{}, bcells:{}", local_cells.size(),
                boundary_cells.size());

  return {std::move(local_cells), std::move(boundary_cells)};
}