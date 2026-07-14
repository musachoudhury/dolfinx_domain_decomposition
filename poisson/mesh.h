
#pragma once

#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/utils.h>
#include <map>
#include <span>

/// @brief Create a new mesh with an extra boundary layer, such that all
/// cells
/// on other processes which share a vertex with this process are ghosted.
/// @param mesh Input mesh
/// @param coord_element A coordinate element for the new mesh. This may be
/// tensor product ordering.

template <std::floating_point T>
dolfinx::mesh::Mesh<T> ghost_layer_mesh(
    dolfinx::mesh::Mesh<T>& mesh,
    const std::function<std::vector<std::int32_t>(
        const dolfinx::graph::AdjacencyList<std::int32_t>&)>& reorder_fn
    = dolfinx::graph::reorder_gps)
{

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const std::size_t tdim = mesh.topology()->dim();
  std::size_t num_cells_local = mesh.topology()->index_map(tdim)->size_local();

  // Find which local vertices are ghosted elsewhere
  auto vertex_destinations
      = mesh.topology()->index_map(0)->index_to_dest_ranks();

  // cell to vertex AdjacencyList
  auto c_to_v = mesh.topology()->connectivity(tdim, 0);

  // Loops over cells
  // Assigns destinations to each cell
  std::vector<int> dests;
  std::vector<int> cdests;
  std::vector<int> offsets = {0};

  for (std::size_t c = 0; c < num_cells_local; ++c)
  {
    dests.push_back(rank);
    cdests.clear();
    for (auto v : c_to_v->links(c))
    {
      auto vdest = vertex_destinations.links(v);
      for (int d : vdest)
        cdests.push_back(d);
    }
    std::sort(cdests.begin(), cdests.end());
    cdests.erase(std::unique(cdests.begin(), cdests.end()), cdests.end());
    dests.insert(dests.end(), cdests.begin(), cdests.end());
    offsets.push_back(dests.size());
  }

  // Builds a cell-to-rank adjacency list
  auto partitioner
      = [&dests, &offsets](MPI_Comm, int /*nparts*/,
                           const std::vector<dolfinx::mesh::CellType>&,
                           const std::vector<std::span<const std::int64_t>>&)
  {
    spdlog::debug("Partitioner: Adjacency list with {} entries",
                  offsets.size() - 1);
    return dolfinx::graph::AdjacencyList<std::int32_t>(std::move(dests),
                                                       std::move(offsets));
  };

  std::size_t num_vertices = mesh.geometry().index_map()->size_local();
  std::array<std::size_t, 2> xshape = {num_vertices, 3};
  std::span<T> x(mesh.geometry().x().data(), xshape[0] * xshape[1]);

  spdlog::debug("num vertices = {}", num_vertices);

  auto dofmap = mesh.geometry().dofmaps().front();
  auto imap = mesh.geometry().index_map();

  std::vector<std::int64_t> dofmap_global(num_cells_local * dofmap.extent(1));
  imap->local_to_global(std::span(dofmap.data_handle(), dofmap_global.size()),
                        dofmap_global);

  spdlog::debug("Call create_mesh");

  std::optional<std::int32_t> max_facet_to_cell_links; // = 2;
  auto new_mesh = dolfinx::mesh::create_mesh(
      mesh.comm(), mesh.comm(), std::span(dofmap_global),
      mesh.geometry().cmaps().front(), mesh.comm(), x, xshape, partitioner,
      max_facet_to_cell_links, reorder_fn);

  spdlog::info("** NEW MESH num_ghosts_cells = {}",
               new_mesh.topology()->index_map(tdim)->num_ghosts());
  spdlog::info("** NEW MESH num_local_cells = {}",
               new_mesh.topology()->index_map(tdim)->size_local());

  return new_mesh;
}
