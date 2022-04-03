#include <torch/script.h>
#include <torch/torch.h>
#include <tuple>
#include <vector>
#include <iostream>
#include "ani.h"

ANI::ANI(const std::string& model_file, int local_rank) : device(local_rank == -1 ? torch::kCPU: torch::kCUDA, local_rank) {
  std::cout << device << std::endl;
  try {
    model = torch::jit::load(model_file, device);
    std::cout << "successfully loaded the model\n";
  }
  catch (const c10::Error &e) {
    std::cerr << "error loading the model\n";
  }
  // model.to(device);
  // std::cout << model.device().type() << std::endl;
}
