/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifdef INTEL_MKL

// #include "dnnl.hpp"
// #include "tensorflow/cc/ops/const_op.h"
// #include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
// #include "tensorflow/core/framework/fake_input.h"
// #include "tensorflow/core/framework/node_def_builder.h"
// #include "tensorflow/core/framework/tensor.h"
// #include "tensorflow/core/framework/types.pb.h"
// #include "tensorflow/core/kernels/ops_testutil.h"
// #include "tensorflow/core/kernels/ops_util.h"
// #include "tensorflow/core/platform/test.h"
// #include "tensorflow/core/platform/test_benchmark.h"

#include "absl/strings/match.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/stacktrace_handler.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/mkl_util.h"  


// Compare the performance of default tensorflow Cast kernel with
// _MklCast kernel on CPU.
//
// Then you could use below command to test _MklCast and default Cast
// performance:
// $ numactl -C xx-xx bazel run --action_env=TF_ENABLE_ONEDNN_OPTS=1 -c opt \
//  //tensorflow/core/kernels/mkl:mkl_cast_op_test -- --benchmark_filter=all
//
// Test with config MKL
// $ numactl -C xx-xx bazel run --config=mkl -c opt                         \
//  //tensorflow/core/kernels/mkl:mkl_cast_op_test -- --benchmark_filter=all

namespace tensorflow { 


// --------------------------------------------------------------------------//
//           Test Cast performance VS _MklCast Kernel                        //
// --------------------------------------------------------------------------//

static Graph* MulGraph(const string& kind, const TensorShape& shapeA, const TensorShape& shapeB) {
  auto* graph = new Graph(OpRegistry::Global());
  const bool isDefault = (kind == "Default");
  Tensor input_t_a(DT_FLOAT, shapeA);
  Tensor input_t_b(DT_FLOAT, shapeB);
  input_t_a.flat<float>().setRandom();
  input_t_b.flat<float>().setRandom(); 

  Node* input_a = test::graph::Constant(graph, input_t_a, "input_a");
  Node* input_b = test::graph::Constant(graph, input_t_b, "input_b");

  Node* not_mkl_shape =
      test::graph::Constant(graph, GetMklMetaTensor(), "not_mkl");

  if (isDefault) {
    TF_CHECK_OK(NodeBuilder(graph->NewName("Default_Mul"), "Mul")
                    .Input(input_a)
                    .Input(input_b)
                    .Attr("T", DT_FLOAT)
                    .Finalize(graph, nullptr));

    return graph;
  }
  // Mkl Cast op.
  TF_CHECK_OK(NodeBuilder(graph->NewName("Mkl_OneDNNMul"), "_MklOneDNNMul")
                  .Input(input_a)
                  .Input(input_b)
                  .Input(not_mkl_shape)
                  .Input(not_mkl_shape)
                  .Attr("T", DT_FLOAT)
                  .Attr("_kernel", "MklLayoutDependentOp")
                  .Finalize(graph, nullptr));
  return graph; 
}

 
#define BM_Mul(kind, A, B, C, D, type)                 \
  static void BM_##kind##_##type##_##A##_##B##_##C##_##D(  \
      ::testing::benchmark::State& state) {                       \
    std::cout << "hebi-dbg: enter BM"; \
    int64 num_computed_elements = (A) * (B);          \
    int64 flops_per_iter = num_computed_elements;                 \
                                                                  \
    test::Benchmark(#type, MulGraph(#kind, {A, B}, {C, D}),  \
                    /*old_benchmark_api*/ false)                  \
        .Run(state);                                              \
    state.SetItemsProcessed(state.iterations() * flops_per_iter); \
  }                                                               \
  BENCHMARK(BM_##kind##_##type##_##A##_##B##_##C##_##D)

#define BM(A, B, C, D, type)                \
  BM_Mul(Mkl, A, B, C, D, type); \
  // BM_Mul(Default, A, B, C, D, type); \

BM(512, 256, 512, 256, cpu)
// BM(512, 256, 1, 1, cpu)


}  // namespace tensorflow

#endif  // INTEL_MKL && ENABLE_MKL 