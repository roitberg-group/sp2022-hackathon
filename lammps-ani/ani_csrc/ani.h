#include <torch/script.h>
#include <torch/torch.h>

class ANI {
public:
  torch::jit::script::Module model;
  torch::Device device;

  ANI(const std::string& model_file, int local_rank);
};
