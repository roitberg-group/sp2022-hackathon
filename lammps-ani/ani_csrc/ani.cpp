#include <torch/script.h>
#include <torch/torch.h>
#include <tuple>
#include <vector>
#include <iostream>
#include <cstdint>
#include "ani.h"

ANI::ANI(const std::string& model_file, int local_rank) : device(local_rank == -1 ? torch::kCPU: torch::kCUDA, local_rank) {
  std::cout << device << std::endl;
  try {
    model = torch::jit::load(model_file, device);
    std::cout << "Successfully loaded the model on " << device << std::endl;
  }
  catch (const c10::Error &e) {
    std::cerr << "Error loading the model on " << device << std::endl;
  }
}

// For simplicity, the accumulated energy will be saved into eng_vdwl,
// instead of writing out per atom energy.
void ANI::compute(double& out_energy, std::vector<float>& out_force,
                  std::vector<int64_t>& species, std::vector<float>& coordinates,
                  std::vector<int64_t>& atom_index12, std::vector<float>& diff_vector,
                  std::vector<float>& distances, std::vector<int64_t>& ghost_index) {
  int ntotal = species.size();
  int nghost = ghost_index.size();
  int npairs_half = distances.size();

  // output tensor
  auto out_force_t = torch::from_blob(out_force.data(), {1, ntotal, 3}, torch::dtype(torch::kFloat32));
  // input tensor
  auto species_t = torch::from_blob(species.data(), {1, ntotal}, torch::dtype(torch::kLong));
  auto coordinates_t = torch::from_blob(coordinates.data(), {1, ntotal, 3}, torch::dtype(torch::kFloat32));
  auto atom_index12_t = torch::from_blob(atom_index12.data(), {2, npairs_half}, torch::dtype(torch::kLong));
  auto diff_vector_t = torch::from_blob(diff_vector.data(), {npairs_half, 3}, torch::dtype(torch::kFloat32));
  auto distances_t = torch::from_blob(distances.data(), {npairs_half}, torch::dtype(torch::kFloat32));
  auto ghost_index_t = torch::from_blob(ghost_index.data(), {nghost}, torch::dtype(torch::kLong));

  // pack forward inputs
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(species_t.to(device));
  inputs.push_back(coordinates_t.to(device).requires_grad_(true));
  inputs.push_back(atom_index12_t.to(device));
  inputs.push_back(diff_vector_t.to(device));
  inputs.push_back(distances_t.to(device));
  inputs.push_back(ghost_index_t.to(device));

  std::cout << "ghost_index_t: " << ghost_index_t.sizes() << std::endl;
  // run ani model
  auto energy_force = model.forward(inputs).toTuple();

  // extract energy and force
  auto energy = energy_force->elements()[0].toTensor();
  auto force = energy_force->elements()[1].toTensor();

  // write energy and force out
  out_energy = energy.item<double>();
  out_force_t.copy_(force);

}
