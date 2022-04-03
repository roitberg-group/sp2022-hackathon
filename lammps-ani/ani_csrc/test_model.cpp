#include <torch/script.h>
#include <torch/torch.h>
#include <string>
#include <iostream>
#include "ani.h"

int main(int argc, const char *argv[]) {

  // Get the path of the model and the device type
  if (argc != 3) {
    std::cerr << "usage: test_model <model.pt> <cpu/cuda>\n";
    return -1;
  }
  std::string model_file = argv[1];
  std::string device_str = argv[2];
  int local_rank;
  if (device_str != "cpu" && device_str != "cuda") {
    std::cerr << "2nd argument must be <cpu/cuda>\n";
    return -1;
  }
  local_rank = device_str == "cpu" ? -1 : 0;
  ANI ani {model_file, local_rank};

  torch::Tensor coords = torch::tensor(
      {{{0.03192167, 0.00638559, 0.01301679},
        {-0.83140486, 0.39370209, -0.26395324},
        {-0.66518241, -0.84461308, 0.20759389},
        {0.45554739, 0.54289633, 0.81170881},
        {0.66091919, -0.16799635, -0.91037834}},
       {{-4.1862600, 0.0575700, -0.0381200},
        {-3.1689400, 0.0523700, 0.0200000},
        {-4.4978600, 0.8211300, 0.5604100},
        {-4.4978700, -0.8000100, 0.4155600},
        {0.00000000, -0.00000000, -0.00000000}}},
      torch::TensorOptions().requires_grad(true).dtype(torch::kFloat32));

  torch::Tensor species = torch::tensor(
      {{1, 0, 0, 0, 0}, {2, 0, 0, 0, -1}},
      torch::TensorOptions().requires_grad(false).dtype(torch::kLong));

  // Define the input variables
  coords = coords.to(ani.device);
  species = species.to(ani.device);
  std::vector<torch::jit::IValue> inputs;
  std::vector<torch::jit::IValue> tuple;
  tuple.push_back(species);
  tuple.push_back(coords);
  inputs.push_back(torch::ivalue::Tuple::create(tuple));

  // Run the model
  auto aev = ani.model.forward(inputs).toTuple()->elements()[1].toTensor();
  std::cout << "First call:  " << aev.sizes() << std::endl;
  auto aev1 = ani.model.forward(inputs).toTuple()->elements()[1].toTensor();
  std::cout << "Second call: " << aev1.sizes() << std::endl;
}
