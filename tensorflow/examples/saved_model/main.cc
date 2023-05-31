#include <vector>
#include <string>
#include <stdio.h>
#include <chrono>
#include <thread>
#include <mutex>

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/profiler/lib/profiler_session.h"
#include "tensorflow/core/profiler/rpc/client/capture_profile.h"

#include "tensorflow/core/platform/threadpool.h"

using Clock = std::chrono::high_resolution_clock;
using MS = std::chrono::duration<double, std::milli>;

int main(int argc, char* argv[]) {
  if (argc < 5) {
    printf("Usage: saved_model_inference PATH [BS] [inter_op] [intra_op] [amp_flag] [profile_flag] [enableopt] [ncore]\n");
    return 0;
  }

  std::string model_path = argv[1];
 
  int BS = atoi(argv[2]);
  printf("BS: %d \n", BS);

  tensorflow::SavedModelBundle bundle;
  tensorflow::SessionOptions session_options;
  tensorflow::RunOptions run_options;

  int inter_op;
  int intra_op;
  if (argc >= 5) {
    inter_op = atoi(argv[3]);
    intra_op = atoi(argv[4]);
    session_options.config.set_intra_op_parallelism_threads(intra_op);
    session_options.config.set_inter_op_parallelism_threads(inter_op);
  }


  printf("intra_op: %d\n", session_options.config.intra_op_parallelism_threads());
  printf("inter_op: %d\n", session_options.config.inter_op_parallelism_threads());


  tensorflow::RewriterConfig* cfg = session_options.config.mutable_graph_options()->mutable_rewrite_options();     

  bool auto_mix_enable = 0;
  if (argc >= 6) {
     auto_mix_enable = atoi(argv[5]);
     printf("auto mix enabled! \n");
     if (auto_mix_enable == 1) {
      cfg->set_auto_mixed_precision_mkl(tensorflow::RewriterConfig::ON);
     }
  }

  int profile = 0;
  if (argc >= 7) {
    profile = atoi(argv[6]);
    printf("profile: %d\n", profile);
  }

  int enableopt = 0;
  if (argc >= 8) {
    enableopt = atoi(argv[7]);
    if (enableopt == 1) {
      putenv("TF_TOWER_FUSION=1");
    }
    printf("enableopt: %d\n", enableopt);
  }

  int ncore = 0;
  if (argc >= 9) {
    ncore = atoi(argv[8]);
    printf("ncore: %d\n", ncore);
  }



  auto status = tensorflow::LoadSavedModel(session_options, run_options, model_path, {"serve"}, &bundle);

  if (!status.ok()) {
    printf("Failed to load model. Error message: %s \n", status.error_message().c_str());
    return -1;
  }

  auto session = bundle.GetSession();

  auto signature = bundle.GetSignatures().at("serving_default");

  std::vector<std::pair<std::string, tensorflow::Tensor>> input_data;
  // input_data.resize(2);
  auto inputs = signature.inputs();
  printf("Inputs:\n");
  for (auto it = inputs.begin(); it != inputs.end(); it++) {
    auto tensor_info = it->second;
    printf("%s %s %s \n", tensor_info.name().c_str(), DataTypeString(tensor_info.dtype()).c_str(), tensor_info.tensor_shape().DebugString().c_str());
    tensorflow::TensorShape shape;
    for (const auto& dim : tensor_info.tensor_shape().dim()) {
      shape.AddDim(dim.size() == -1 ? BS : dim.size());
    }
    if (tensor_info.name() == "input:0"){
      input_data.push_back(make_pair(tensor_info.name(), tensorflow::Tensor(tensor_info.dtype(), shape)));
    }
  }

  for (auto x : input_data) {
    std::cout << "name: " << x.first << ", tensor: " << x.second.DebugString() << "\n";
  }

  std::vector<std::string> output_nodes;
  auto outputs = signature.outputs();
  printf("Output nodes:\n");
  for (auto it = outputs.begin(); it != outputs.end(); it++) {
    output_nodes.push_back(it->second.name());
    //printf("%s\n", it->second.name().c_str());
  }

  //init model
  /*
  {
    std::vector<std::string> init = {"model_init"};
    auto status = session->Run({}, {}, init, nullptr);
    if (!status.ok()) {
      printf("Failed to init model. Error message: %s \n", status.error_message().c_str());
      return 0;
    }
  }
  */

  std::vector<tensorflow::Tensor> predictions;

  int warm = 100;
  for(int i = 0; i < warm; i++) {
    auto status = session->Run(input_data, output_nodes, {}, &predictions);
    if (!status.ok()) {
      printf("Failed to run. Error message: %s \n", status.error_message().c_str());
      return 0;
    }
  }

  printf("start benchmarking.\n");


  int loop = 1000;
  Clock::time_point start = Clock::now();
  for(int i = 0; i < loop; i++) {
    auto status = session->Run(input_data, output_nodes, {}, &predictions);
    if (!status.ok()) {
      printf("Failed to run. Error message: %s \n", status.error_message().c_str());
      return 0;
    }
  }
  Clock::time_point stop = Clock::now();
  MS diff = std::chrono::duration_cast<MS>(stop - start);
  printf("TOWERDENSE OPT: %d, BF16 %d, inter/intra %d/%d, BS %d, core %d, Average Latency: %f ms.\n", enableopt, auto_mix_enable, inter_op, intra_op, BS, ncore, diff.count()/loop);


  if (profile > 0) {
    std::unique_ptr<tensorflow::ProfilerSession> profiler_session;
    printf("start profiler\n");
    auto opt = tensorflow::ProfilerSession::DefaultOptions();
    opt.set_host_tracer_level(3);
    profiler_session = tensorflow::ProfilerSession::Create(opt);

    for(int i = 0; i < 10; i++) {
      auto status = session->Run(input_data, output_nodes, {}, &predictions);
      if (!status.ok()) {
        printf("Failed to run. Error message: %s \n", status.error_message().c_str());
        return 0;
      }
    }


    tensorflow::profiler::XSpace xspace;
    auto status1 = profiler_session->CollectData(&xspace);
    if (!status1.ok()) {
        printf("Failed to profile. Error message: %s \n", status1.error_message().c_str());
        return 0;
    }
    auto status2 = tensorflow::profiler::ExportToTensorBoard(xspace, "cpp-logdir");
    if (!status2.ok()) {
        printf("Failed to profile. Error message: %s \n", status2.error_message().c_str());
        return 0;
    }
    profiler_session.reset();
    printf("stop profiler\n");
  }

  return 0;
}