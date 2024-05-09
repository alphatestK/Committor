/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Copyright (c) 2022 of Luigi Bonati and Enrico Trizio.

The pytorch module is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The pytorch module is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with plumed.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */

// #ifdef __PLUMED_HAS_LIBTORCH
// convert LibTorch version to string
//#define STRINGIFY(x) #x
//#define TOSTR(x) STRINGIFY(x)
//#define LIBTORCH_VERSION TO_STR(TORCH_VERSION_MAJOR) "." TO_STR(TORCH_VERSION_MINOR) "." TO_STR(TORCH_VERSION_PATCH)

#include "core/PlumedMain.h"
#include "function/Function.h"
#include "function/ActionRegister.h"

#include <torch/torch.h>
#include <torch/script.h>

#include <fstream>
#include <cmath>


// We have to do a backward compatability hack for <1.10
// https://discuss.pytorch.org/t/how-to-check-libtorch-version/77709/4
// Basically, the check in torch::jit::freeze
// (see https://github.com/pytorch/pytorch/blob/dfbd030854359207cb3040b864614affeace11ce/torch/csrc/jit/api/module.cpp#L479)
// is wrong, and we have ro "reimplement" the function
// to get around that...
// it's broken in 1.8 and 1.9
// BUT the internal logic in the function is wrong in 1.10
// So we only use torch::jit::freeze in >=1.11
// credits for this implementation of the hack to the NequIP guys
#if (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR <= 10)
  #define DO_TORCH_FREEZE_HACK
  // For the hack, need more headers:
  #include <torch/csrc/jit/passes/freeze_module.h>
  #include <torch/csrc/jit/passes/frozen_graph_optimizations.h>
#endif

using namespace std;

namespace PLMD {
namespace function {
namespace pytorch {

class PytorchModelBias :
  public Function
{
  unsigned _n_in;
  unsigned _n_out;
  double lambda;
  double epsilon;
  torch::jit::Module _model;
  torch::Device device = torch::kCPU;
public:
  explicit PytorchModelBias(const ActionOptions&);
  void calculate();
  static void registerKeywords(Keywords& keys);

  std::vector<float> tensor_to_vector(const torch::Tensor& x);
};

PLUMED_REGISTER_ACTION(PytorchModelBias,"PYTORCH_MODEL_BIAS")

void PytorchModelBias::registerKeywords(Keywords& keys) {
  Function::registerKeywords(keys);
  keys.use("ARG");
  keys.add("optional","FILE","Filename of the PyTorch compiled model");
  keys.add("optional","LAMBDA","Prefactor of the bias, default 1");
  keys.add("optional","EPSILON","Numerical regularization term in the logarithm, default 1e-6");
  keys.addOutputComponent("node", "default", "Model outputs");
  keys.addOutputComponent("bias", "default", "Model outputs");
}

std::vector<float> PytorchModelBias::tensor_to_vector(const torch::Tensor& x) {
  return std::vector<float>(x.data_ptr<float>(), x.data_ptr<float>() + x.numel());
}

PytorchModelBias::PytorchModelBias(const ActionOptions&ao):
  Action(ao),
  Function(ao)
{ //print pytorch version

  //number of inputs of the model
  _n_in=getNumberOfArguments();

  //parse model name
  std::string fname="model.ptc";
  parse("FILE",fname);

  //parse params
  lambda = 1.0;
  parse("LAMBDA", lambda);

  epsilon = 1e-6;
  parse("EPSILON", epsilon);

  // we create the metatdata dict 
  std::unordered_map<std::string, std::string> metadata = {
    {"_jit_bailout_depth", ""},
    {"_jit_fusion_strategy", ""}
  };

  //deserialize the model from file
  try {
    _model = torch::jit::load(fname, device, metadata);
  } 

  //if an error is thrown check if the file exists or not
  catch (const c10::Error& e) {
    std::ifstream infile(fname);
    bool exist = infile.good();
    infile.close();
    if (exist) {
      // print libtorch version
      std::stringstream ss;
      ss << TORCH_VERSION_MAJOR << "." << TORCH_VERSION_MINOR << "." << TORCH_VERSION_PATCH;
      std::string version;
      ss >> version; // extract into the string.
      plumed_merror("Cannot load FILE: '"+fname+"'. Please check that it is a Pytorch compiled model (exported with 'torch.jit.trace' or 'torch.jit.script') and that the Pytorch version matches the LibTorch one ("+version+").");
    }
    else {
      plumed_merror("The FILE: '"+fname+"' does not exist.");
    }
  }
  checkRead();

  // // recover the descriptors' statistics from model buffers
  // // this has to be done before freezing or optimizing for inference, otherwise they are unaccessible
  // torch::Tensor stat = torch::zeros({3,_n_in});
  // int c = 0;
  //   for (const torch::Tensor& b : _model.buffers()) {
  //           stat[c] = b;
  //           // std::cout << buff << std::endl;
  //           // std::cout << stat[c] << std::endl;
  //           c++;
  //   }

  // max_bias = stat[0][0];
  // mean = stat[1];
  // std = stat[2];
  // max_bias = max_bias.unsqueeze(0);
  // mean = mean.unsqueeze(0);
  // std = std.unsqueeze(0);
  // // std::cout << max_bias << std::endl;
  // // std::cout << mean << std::endl;
  // // std::cout << std << std::endl;

  // Optimize model 
  _model.eval();
  #ifdef DO_TORCH_FREEZE_HACK
      // Do the hack
      // Copied from the implementation of torch::jit::freeze,
      // except without the broken check
      // See https://github.com/pytorch/pytorch/blob/dfbd030854359207cb3040b864614affeace11ce/torch/csrc/jit/api/module.cpp
      bool optimize_numerics = true;  // the default
      // the {} is preserved_attrs
      auto out_mod = torch::jit::freeze_module(
        _model, {}
      );
      // See 1.11 bugfix in https://github.com/pytorch/pytorch/pull/71436
      auto graph = out_mod.get_method("forward").graph();
      OptimizeFrozenGraph(graph, optimize_numerics);
      _model = out_mod;
    #else
      // Do it normally
     _model = torch::jit::freeze(_model);
    #endif

  #if (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR <= 10)
    // Set JIT bailout to avoid long recompilations for many steps
    size_t jit_bailout_depth;
    if (metadata["_jit_bailout_depth"].empty()) {
      // This is the default used in the Python code
      jit_bailout_depth = 1;
    } else {
      jit_bailout_depth = std::stoi(metadata["_jit_bailout_depth"]);
    }
    torch::jit::getBailoutDepth() = jit_bailout_depth;
  #else
    // In PyTorch >=1.11, this is now set_fusion_strategy
    torch::jit::FusionStrategy strategy;
    if (metadata["_jit_fusion_strategy"].empty()) {
      // This is the default used in the Python code
      strategy = {{torch::jit::FusionBehavior::DYNAMIC, 0}};
    } else {
      std::stringstream strat_stream(metadata["_jit_fusion_strategy"]);
      std::string fusion_type, fusion_depth;
      while(std::getline(strat_stream, fusion_type, ',')) {
        std::getline(strat_stream, fusion_depth, ';');
        strategy.push_back({fusion_type == "STATIC" ? torch::jit::FusionBehavior::STATIC : torch::jit::FusionBehavior::DYNAMIC, std::stoi(fusion_depth)});
      }
    }
    torch::jit::setFusionStrategy(strategy);
  #endif

// TODO check torch::jit::optimize_for_inference() for more complex models
// This could speed up the code, it was not available on LTS 
  #if (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 10)
  _model = torch::jit::optimize_for_inference(_model);
  #endif
// END -> Optimize model 


 //check the dimension of the output
  log.printf("Checking output dimension:\n");
  std::vector<float> input_test (_n_in);
  torch::Tensor single_input = torch::tensor(input_test).view({1,_n_in});
  single_input = single_input.to(device);
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back( single_input );
  //std::cout << 'Input: ' << inputs << std::endl;
  //std::cout << 'Input: '<< inputs[0] << std::endl;
  torch::Tensor output = _model.forward( inputs ).toTensor();
  //std::cout << 'Output: '<< output << std::endl;
  vector<float> cvs = this->tensor_to_vector (output);
  //std::cout << "CV: " << cvs << std::endl;
  _n_out=cvs.size();

//create components of output
  for(unsigned j=0; j<_n_out; j++) {
    string name_comp = "node-"+std::to_string(j);
    addComponentWithDerivatives( name_comp );
    componentIsNotPeriodic( name_comp );
  }
  
  for(unsigned j=0; j<_n_out; j++) {
    string name_comp = "bias-"+std::to_string(j);
    addComponentWithDerivatives( name_comp );
    componentIsNotPeriodic( name_comp );
  }

  //print log
  //log.printf("Pytorch Model Loaded: %s \n",fname);
  log.printf("Number of input: %d \n",_n_in);
  log.printf("Number of outputs: %d \n",_n_out);
  log.printf("  Bibliography: ");
  log<<plumed.cite("Bonati, Rizzi and Parrinello, J. Phys. Chem. Lett. 11, 2998-3004 (2020)");
  log<<plumed.cite("Trizio and Parrinello, J. Phys. Chem. Lett. 12, 8621-8626 (2021)");
  log.printf("\n");
}


void PytorchModelBias::calculate() {

// retrieve arguments
vector<float> current_S(_n_in);
for(unsigned i=0; i<_n_in; i++)
  current_S[i]=getArgument(i);
//convert to tensor
torch::Tensor input_S = torch::tensor(current_S).view({1,_n_in}).to(device);
input_S.set_requires_grad(true);
//convert to Ivalue
std::vector<torch::jit::IValue> inputs;
inputs.push_back( input_S );
//calculate output
torch::Tensor output = _model.forward( inputs ).toTensor();

// for(unsigned j=0; j<_n_out; j++) {  --> TODO maybe fix for more dimensions

// compute gradients of CV 
torch::Tensor grad_output = torch::ones({1}).expand({1, 1}).to(device);
torch::Tensor gradient = torch::autograd::grad({output},
                      {input_S},
    /*grad_outputs=*/ {grad_output},
    /*retain_graph=*/true,
    /*create_graph=*/true)[0]; // the [0] is to get a tensor and not a vector<at::tensor>


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////// HERE WE COMPUTE THE BIAS, IN CASE MODIFY HERE /////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// from the gradient we compute the bias
// torch::Tensor log_grad_sq = (barrier/alpha)* torch::log( torch::sum( torch::pow(torch::masked_select(gradient,mask), 2)*torch::masked_select(rank_std,mask) ) + 1 ) / max_bias * torch::prod(torch::exp(-0.5*torch::div(torch::pow((torch::masked_select(input_S, mask) - torch::masked_select(mean,mask)),2), torch::pow(torch::masked_select(std,mask)*std_spread,2))));
torch::Tensor Epsilon = torch::tensor(epsilon);
torch::Tensor log_grad_sq = lambda * ( torch::log( torch::sum( torch::pow(gradient, 2) ) + Epsilon ) - torch::log(Epsilon) );

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// get derivatives of bias --> forces
torch::Tensor grad_output2 = torch::ones({1}).to(device);
torch::Tensor gradient2 = torch::autograd::grad({log_grad_sq},
                        {input_S},
      /*grad_outputs=*/ {grad_output2},
      /*retain_graph=*/true,
      /*create_graph=*/false)[0]; // the [0] is to get a tensor and not a vector<at::tensor>

// we set the derivatives for plumed
vector<float> der = this->tensor_to_vector ( gradient );  
string name_comp = "node-"+std::to_string(0); //TODO fix for multi output
    //set derivatives of component j
    for(unsigned i=0; i<_n_in; i++)
      setDerivative( getPntrToComponent(name_comp),i, der[i] ); 

vector<float> der2 = this->tensor_to_vector ( gradient2 );  
string name_comp_bias = "bias-"+std::to_string(0); //TODO fix for multi output
    //set derivatives of component j
    for(unsigned i=0; i<_n_in; i++)
      setDerivative( getPntrToComponent(name_comp_bias),i, der2[i] ); 

//set CV values
vector<float> cvs = this->tensor_to_vector (output);
for(unsigned j=0; j<_n_out; j++) {
    string name_comp = "node-"+std::to_string(j);
    getPntrToComponent(name_comp)->set(cvs[j]);
  }

// set BIAS value
vector<float> bias = this->tensor_to_vector (log_grad_sq);
for(unsigned j=0; j<_n_out; j++) {
    string name_comp = "bias-"+std::to_string(j);
    getPntrToComponent(name_comp)->set(bias[j]);
  }

}


} //pytorch
} //function
} //PLMD

// #endif //PLUMED_HAS_LIBTORCH
