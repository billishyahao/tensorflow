/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#if defined(INTEL_MKL) && defined(ENABLE_MKL)
#include "tensorflow/cc/ops/nn_ops_internal.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/devices.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/remapper.h"
#include "tensorflow/core/grappler/optimizers/constant_folding.h"


#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/mkl_util.h"

namespace tensorflow {
namespace grappler {

class MklRemapperTest : public GrapplerTest {
 public:
  const string kAddNOp = "AddN";
  const string kAddOp = "Add";
  const string kAddV2Op = "AddV2";

//  protected:
//   void FuseConv2DWithBiasAndAddNOrAdd(const string& data_format,
//                                       const string& activation, string add_op,
//                                       bool add_with_bcast) {
//     using ::tensorflow::ops::Placeholder;

//     tensorflow::Scope s = tensorflow::Scope::NewRootScope();

//     auto input_shape = (data_format == "NHWC")
//                            ? ops::Placeholder::Shape({8, 32, 32, 3})
//                            : ops::Placeholder::Shape({8, 3, 32, 32});
//     auto input_shape_addn = ops::Placeholder::Shape({});
//     if (data_format == "NHWC") {
//       if (add_with_bcast)
//         input_shape_addn = ops::Placeholder::Shape({128});
//       else
//         input_shape_addn = ops::Placeholder::Shape({8, 32, 32, 128});
//     } else {
//       if (add_with_bcast)
//         input_shape_addn = ops::Placeholder::Shape({32});
//       else
//         input_shape_addn = ops::Placeholder::Shape({8, 128, 32, 32});
//     }
//     auto filter_shape = ops::Placeholder::Shape({1, 1, 3, 128});
//     auto bias_shape = ops::Placeholder::Shape({128});

//     auto input = Placeholder(s.WithOpName("input"), DT_FLOAT, input_shape);
//     auto input_addn =
//         Placeholder(s.WithOpName("input_addn"), DT_FLOAT, input_shape_addn);
//     auto filter = Placeholder(s.WithOpName("filter"), DT_FLOAT, filter_shape);
//     auto bias = Placeholder(s.WithOpName("bias"), DT_FLOAT, bias_shape);

//     std::vector<int> strides = {1, 1, 1, 1};
//     auto conv =
//         ops::Conv2D(s.WithOpName("conv"), input, filter, strides, "SAME",
//                     ops::Conv2D::Attrs().DataFormat(data_format));
//     auto bias_add = ops::BiasAdd(s.WithOpName("bias_add"), conv, bias,
//                                  ops::BiasAdd::Attrs().DataFormat(data_format));

//     auto addfetch = [&](::tensorflow::Input addop) {
//       auto activate = s.WithOpName("activation");
//       auto fetch = s.WithOpName("fetch");
//       if (activation == "Relu") {
//         ops::Identity(fetch, ops::Relu(activate, addop));
//       } else if (activation == "Relu6") {
//         ops::Identity(fetch, ops::Relu6(activate, addop));
//       } else if (activation == "Elu") {
//         ops::Identity(fetch, ops::Elu(activate, addop));
//       } else if (activation == "LeakyRelu") {
//         ops::Identity(fetch, ops::internal::LeakyRelu(activate, addop));
//       } else {
//         DCHECK(activation == "None");
//         ops::Identity(fetch, addop);
//       }
//     };

//     if (add_op == kAddNOp) {
//       auto addn = ops::AddN(s.WithOpName(add_op),
//                             std::initializer_list<Input>{input_addn, bias_add});
//       addfetch(addn);
//     } else if (add_op == kAddV2Op) {
//       auto add = ops::AddV2(s.WithOpName(add_op), input_addn, bias_add);
//       addfetch(add);
//     } else {
//       auto add = ops::Add(s.WithOpName(add_op), input_addn, bias_add);
//       addfetch(add);
//     }
//     auto input_tensor = GenerateRandomTensor<DT_FLOAT>(
//         TensorShape(input_shape.shape_.dim_sizes()));
//     auto input_addn_tensor = GenerateRandomTensor<DT_FLOAT>(
//         TensorShape(input_shape_addn.shape_.dim_sizes()));
//     auto filter_tensor = GenerateRandomTensor<DT_FLOAT>(
//         TensorShape(filter_shape.shape_.dim_sizes()));
//     auto bias_tensor = GenerateRandomTensor<DT_FLOAT>(
//         TensorShape(bias_shape.shape_.dim_sizes()));

//     GrapplerItem item;
//     item.fetch = {"fetch"};
//     item.feed = {{"input", input_tensor},
//                  {"filter", filter_tensor},
//                  {"bias", bias_tensor},
//                  {"input_addn", input_addn_tensor}};
//     TF_CHECK_OK(s.ToGraphDef(&item.graph));

//     // Place all nodes on CPU.
//     for (int i = 0; i < item.graph.node_size(); ++i) {
//       item.graph.mutable_node(i)->set_device("/device:CPU:0");
//     }

//     // Set Rewriter config to AGGRESSIVE so that we can use Placeholder shape
//     // to test that Add with both inputs having same shape get fused with
//     // Conv2D. Setting this config to AGGRESSIVE is not required for the feature
//     // though.
//     Remapper optimizer(RewriterConfig::AGGRESSIVE);
//     GraphDef output;
//     TF_CHECK_OK(optimizer.Optimize(nullptr, item, &output));

//     bool check_fusion = !add_with_bcast;
//     int found = 0;
//     for (const NodeDef& node : output.node()) {
//       auto fetch_node_name = activation != "None" ? "activation" : add_op;
//       if (node.name() == fetch_node_name) {
//         if (check_fusion) {
//           EXPECT_EQ("_FusedConv2D", node.op());
//           EXPECT_EQ("input", node.input(0));
//           EXPECT_EQ("filter", node.input(1));

//           EXPECT_EQ(2, node.attr().at("num_args").i());
//           EXPECT_EQ("bias", node.input(2));
//           EXPECT_EQ("input_addn", node.input(3));

//           const auto fused_ops = node.attr().at("fused_ops").list().s();
//           if (activation != "None") {
//             EXPECT_EQ(3, fused_ops.size());
//             EXPECT_EQ("BiasAdd", fused_ops[0]);
//             EXPECT_EQ("Add", fused_ops[1]);
//             EXPECT_EQ(activation, fused_ops[2]);
//           } else {
//             EXPECT_EQ(2, fused_ops.size());
//             EXPECT_EQ("BiasAdd", fused_ops[0]);
//             EXPECT_EQ("Add", fused_ops[1]);
//           }
//         } else {
//           if (activation != "None") {
//             EXPECT_EQ(node.op(), activation);
//             ASSERT_EQ(node.input_size(), 1);
//             EXPECT_EQ(node.input(0), add_op);
//           } else {
//             EXPECT_EQ(node.op(), add_op);
//             ASSERT_EQ(node.input_size(), 2);
//           }
//         }
//         found++;
//       }
//     }
//     EXPECT_EQ(1, found);

//     auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
//     auto tensors = EvaluateNodes(output, item.fetch, item.feed);
//     EXPECT_EQ(1, tensors_expected.size());
//     EXPECT_EQ(1, tensors.size());
//     // Using relative tolerance since oneDNN could produce different results
//     // when float32 numbers need to be rounded during accumulation.
//     test::ExpectClose(tensors_expected[0], tensors[0], 0, 1e-6);
//   }
};

// #define CREATE_CONV2DFUSION_TEST(data_format, addop, activation, bcast)                          \
//   TEST_F(                                                                                        \
//       MklRemapperTest,                                                                           \
//       FuseConv2DWithBiasAnd##addop##_##data_format##_activation##activation##_addbcast##bcast) { \
//     FuseConv2DWithBiasAndAddNOrAdd(#data_format, #activation, #addop, bcast);                    \
//   }

// #define CREATE_CONV2DFUSION_ADD_ACTIVATION_TEST(data_format, addop, bcast) \
//   CREATE_CONV2DFUSION_TEST(data_format, addop, Relu, bcast);               \
//   CREATE_CONV2DFUSION_TEST(data_format, addop, Relu6, bcast);              \
//   CREATE_CONV2DFUSION_TEST(data_format, addop, Elu, bcast);                \
//   CREATE_CONV2DFUSION_TEST(data_format, addop, LeakyRelu, bcast);          \
//   CREATE_CONV2DFUSION_TEST(data_format, addop, None, bcast);

// #define CREATE_CONV2DFUSION_ADD_NOBCAST_TEST(addop)            \
//   CREATE_CONV2DFUSION_ADD_ACTIVATION_TEST(NHWC, addop, false); \
//   CREATE_CONV2DFUSION_ADD_ACTIVATION_TEST(NCHW, addop, false);

// CREATE_CONV2DFUSION_ADD_NOBCAST_TEST(AddN);

// #define CREATE_CONV2DFUSION_ADD_BCAST_TEST(addop)              \
//   CREATE_CONV2DFUSION_ADD_ACTIVATION_TEST(NHWC, addop, false); \
//   CREATE_CONV2DFUSION_ADD_ACTIVATION_TEST(NCHW, addop, false); \
//   CREATE_CONV2DFUSION_ADD_ACTIVATION_TEST(NHWC, addop, true);  \
//   CREATE_CONV2DFUSION_ADD_ACTIVATION_TEST(NCHW, addop, true);

// CREATE_CONV2DFUSION_ADD_BCAST_TEST(Add);
// CREATE_CONV2DFUSION_ADD_BCAST_TEST(AddV2);

// #undef CREATE_CONV2DFUSION_ADD_NOBCAST_TEST
// #undef CREATE_CONV2DFUSION_ADD_BCAST_TEST
// #undef CREATE_CONV2DFUSION_ADD_ACTIVATION_TEST
// #undef CREATE_CONV2DFUSION_TEST

// #define REGISTER_TEST(NAME, T, INPUT)                                         \
//   TEST_F(MklRemapperTest, NAME##_##T) {                                       \
//     using ::tensorflow::ops::Placeholder;                                     \
//                                                                               \
//     for (const string& activation : {"Relu", "Relu6", "Elu", "None"}) {       \
//       tensorflow::Scope s = tensorflow::Scope::NewRootScope();                \
//                                                                               \
//       auto input_shape = Placeholder::Shape({8, 32, 32, 3});                  \
//       auto filter_shape = Placeholder::Shape({1, 1, 3, 1});                   \
//       auto bias_shape = Placeholder::Shape({3});                              \
//                                                                               \
//       auto input = Placeholder(s.WithOpName("input"), DT_FLOAT, input_shape); \
//       auto filter =                                                           \
//           Placeholder(s.WithOpName("filter"), DT_FLOAT, filter_shape);        \
//       auto bias = Placeholder(s.WithOpName("bias"), DT_FLOAT, bias_shape);    \
//                                                                               \
//       std::vector<int> strides = {1, 1, 1, 1};                                \
//       auto conv = ops::DepthwiseConv2dNative(s.WithOpName("depthwise_conv"),  \
//                                              input, filter, strides, "SAME"); \
//       auto bias_add = ops::BiasAdd(s.WithOpName("bias_add"), conv, bias);     \
//                                                                               \
//       ops::Identity fetch = [&]() -> ops::Identity {                          \
//         auto activate = s.WithOpName("activation");                           \
//         auto fetch = s.WithOpName("fetch");                                   \
//                                                                               \
//         if (activation == "Relu") {                                           \
//           return ops::Identity(fetch, ops::Relu(activate, bias_add));         \
//         } else if (activation == "Relu6") {                                   \
//           return ops::Identity(fetch, ops::Relu6(activate, bias_add));        \
//         } else if (activation == "Elu") {                                     \
//           return ops::Identity(fetch, ops::Elu(activate, bias_add));          \
//         }                                                                     \
//                                                                               \
//         DCHECK(activation == "None");                                         \
//         return ops::Identity(fetch, bias_add);                                \
//       }();                                                                    \
//                                                                               \
//       auto input_t = GenerateRandomTensor<DT_FLOAT>({8, 32, 32, 3});          \
//       auto filter_t = GenerateRandomTensor<DT_FLOAT>({1, 1, 3, 1});           \
//       auto bias_t = GenerateRandomTensor<DT_FLOAT>({3});                      \
//                                                                               \
//       GrapplerItem item;                                                      \
//       item.fetch = {"fetch"};                                                 \
//       item.feed = {                                                           \
//           {"input", input_t}, {"filter", filter_t}, {"bias", bias_t}};        \
//       TF_CHECK_OK(s.ToGraphDef(&item.graph));                                 \
//                                                                               \
//       for (int i = 0; i < item.graph.node_size(); ++i) {                      \
//         item.graph.mutable_node(i)->set_device("/device:CPU:0");              \
//       }                                                                       \
//                                                                               \
//       Remapper optimizer(RewriterConfig::ON);                                 \
//       GraphDef output;                                                        \
//       TF_CHECK_OK(optimizer.Optimize(nullptr, item, &output));                \
//                                                                               \
//       int found = 0;                                                          \
//       for (const NodeDef& node : output.node()) {                             \
//         if (node.name() != "bias_add" && node.name() != "activation")         \
//           continue;                                                           \
//                                                                               \
//         EXPECT_EQ(node.op(), "_FusedDepthwiseConv2dNative");                  \
//         ASSERT_EQ(node.input_size(), 3);                                      \
//         EXPECT_EQ(node.input(0), "input");                                    \
//         EXPECT_EQ(node.input(1), "filter");                                   \
//                                                                               \
//         EXPECT_EQ(node.attr().at("num_args").i(), 1);                         \
//         EXPECT_EQ(node.input(2), "bias");                                     \
//                                                                               \
//         const auto fused_ops = node.attr().at("fused_ops").list().s();        \
//         if (node.name() == "bias_add") {                                      \
//           ASSERT_EQ(fused_ops.size(), 1);                                     \
//           EXPECT_EQ(fused_ops[0], "BiasAdd");                                 \
//           found++;                                                            \
//         }                                                                     \
//         if (node.name() == "activation") {                                    \
//           ASSERT_EQ(fused_ops.size(), 2);                                     \
//           EXPECT_EQ(fused_ops[0], "BiasAdd");                                 \
//           EXPECT_EQ(fused_ops[1], activation);                                \
//           found++;                                                            \
//         }                                                                     \
//       }                                                                       \
//       EXPECT_EQ(found, 1);                                                    \
//                                                                               \
//       auto tensors_expected =                                                 \
//           EvaluateNodes(item.graph, item.fetch, item.feed);                   \
//       ASSERT_EQ(tensors_expected.size(), 1);                                  \
//       auto tensors = EvaluateNodes(output, item.fetch, item.feed);            \
//       ASSERT_EQ(tensors.size(), 1);                                           \
//       test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);   \
//     }                                                                         \
//   }
// REGISTER_TEST_ALL_TYPES(FuseDepthwiseConv2DWithBiasAndActivation);
// #undef REGISTER_TEST

// TEST_F(MklRemapperTest, FuseBatchNormWithRelu) {
//   using ::tensorflow::ops::Placeholder;

//   for (bool is_training : {true, false}) {
//     for (bool has_side_input : {true, false}) {
//       tensorflow::Scope s = tensorflow::Scope::NewRootScope();

//       const int num_channels = 24;

//       TensorShape channel_shape({num_channels});
//       TensorShape empty_shape({0});

//       auto input =
//           Placeholder(s.WithOpName("input"), DT_FLOAT,
//                       ops::Placeholder::Shape({2, 8, 8, num_channels}));
//       auto input_cast = ops::Cast(s.WithOpName("input_cast"), input, DT_FLOAT);
//       auto scale = Placeholder(s.WithOpName("scale"), DT_FLOAT);
//       auto offset = Placeholder(s.WithOpName("offset"), DT_FLOAT);
//       auto mean = Placeholder(s.WithOpName("mean"), DT_FLOAT);
//       auto var = Placeholder(s.WithOpName("var"), DT_FLOAT);

//       float epsilon = 0.1f;
//       auto fbn =
//           ops::FusedBatchNormV3(s.WithOpName("fused_batch_norm"), input_cast,
//                                 scale, offset, mean, var,
//                                 ops::FusedBatchNormV3::IsTraining(is_training)
//                                     .Epsilon(epsilon)
//                                     .DataFormat("NHWC"));

//       if (has_side_input) {
//         auto side_input =
//             Placeholder(s.WithOpName("side_input"), DT_FLOAT,
//                         ops::Placeholder::Shape({2, 8, 8, num_channels}));
//         auto side_input_cast =
//             ops::Cast(s.WithOpName("side_input_cast"), side_input, DT_FLOAT);
//         auto add = ops::Add(s.WithOpName("add"), fbn.y, side_input_cast);
//         auto relu = ops::Relu(s.WithOpName("relu"), add);
//       } else {
//         auto relu = ops::Relu(s.WithOpName("relu"), fbn.y);
//       }

//       auto input_t = GenerateRandomTensor<DT_FLOAT>({2, 8, 8, num_channels});
//       auto scale_t = GenerateRandomTensor<DT_FLOAT>(channel_shape);
//       auto offset_t = GenerateRandomTensor<DT_FLOAT>(channel_shape);
//       auto mean_t = GenerateRandomTensor<DT_FLOAT>(is_training ? empty_shape
//                                                                : channel_shape);
//       auto var_t = GenerateRandomTensor<DT_FLOAT>(is_training ? empty_shape
//                                                               : channel_shape);
//       auto side_input_t =
//           GenerateRandomTensor<DT_FLOAT>({2, 8, 8, num_channels});

//       GrapplerItem item;
//       item.fetch = {"relu"};
//       if (has_side_input)
//         item.feed = {{"input", input_t},   {"scale", scale_t},
//                      {"offset", offset_t}, {"mean", mean_t},
//                      {"var", var_t},       {"side_input", side_input_t}};
//       else
//         item.feed = {{"input", input_t},
//                      {"scale", scale_t},
//                      {"offset", offset_t},
//                      {"mean", mean_t},
//                      {"var", var_t}};
//       TF_ASSERT_OK(s.ToGraphDef(&item.graph));

//       // Place all nodes on CPU.
//       for (int i = 0; i < item.graph.node_size(); ++i) {
//         item.graph.mutable_node(i)->set_device("/device:CPU:0");
//       }

//       Remapper optimizer(RewriterConfig::AGGRESSIVE);
//       GraphDef output;
//       TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

//       int found = 0;
//       if (has_side_input) {
//         for (const NodeDef& node : output.node()) {
//           if (node.name() == "add") {
//             EXPECT_EQ(node.op(), "Add");
//             ASSERT_EQ(node.input_size(), 2);
//             EXPECT_EQ(node.input(0), "fused_batch_norm");
//             EXPECT_EQ(node.input(1), "side_input_cast");
//             found++;
//           }
//           if (node.name() == "relu") {
//             EXPECT_EQ(node.op(), "Relu");
//             ASSERT_EQ(node.input_size(), 1);
//             EXPECT_EQ(node.input(0), "add");
//             found++;
//           }
//           if (node.name() == "fused_batch_norm") {
//             EXPECT_EQ(node.op(), "FusedBatchNormV3");
//             ASSERT_EQ(node.input_size(), 5);
//             EXPECT_EQ(node.input(0), "input_cast");
//             EXPECT_EQ(node.input(1), "scale");
//             EXPECT_EQ(node.input(2), "offset");
//             EXPECT_EQ(node.input(3), "mean");
//             EXPECT_EQ(node.input(4), "var");
//             found++;
//           }
//         }
//         EXPECT_EQ(found, 3);
//       } else {
//         for (const NodeDef& node : output.node()) {
//           if (node.name() == "relu") {
//             EXPECT_EQ(node.op(), "Identity");
//             ASSERT_EQ(node.input_size(), 1);
//             EXPECT_EQ(node.input(0), "fused_batch_norm");
//             found++;
//           }
//           if (node.name() == "fused_batch_norm") {
//             EXPECT_EQ(node.op(), "_FusedBatchNormEx");
//             ASSERT_EQ(node.input_size(), 5);
//             EXPECT_EQ(node.input(0), "input_cast");
//             EXPECT_EQ(node.input(1), "scale");
//             EXPECT_EQ(node.input(2), "offset");
//             EXPECT_EQ(node.input(3), "mean");
//             EXPECT_EQ(node.input(4), "var");

//             auto attr = node.attr();
//             EXPECT_EQ(attr["num_side_inputs"].i(), 0);
//             EXPECT_EQ(attr["activation_mode"].s(), "Relu");
//             found++;
//           }
//         }
//         EXPECT_EQ(found, 2);
//       }

//       auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
//       ASSERT_EQ(tensors_expected.size(), 1);
//       auto tensors = EvaluateNodes(output, item.fetch, item.feed);
//       ASSERT_EQ(tensors.size(), 1);
//       test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
//     }
//   }
// }

// TEST_F(MklRemapperTest, FuseMatMulWithBiasAddAndAdd) {
//   using ::tensorflow::ops::Placeholder;

//   for (const string& add_op : {"BiasAdd", "AddV2", "Add"}) {
//     tensorflow::Scope s = tensorflow::Scope::NewRootScope();

//     auto input_shape = ops::Placeholder::Shape({4, 32});
//     auto input_shape_add = ops::Placeholder::Shape({4, 8});
//     auto filter_shape = ops::Placeholder::Shape({32, 8});
//     auto bias_shape = ops::Placeholder::Shape({8});

//     auto input = Placeholder(s.WithOpName("input"), DT_FLOAT, input_shape);
//     auto input_add =
//         Placeholder(s.WithOpName("input_add"), DT_FLOAT, input_shape_add);
//     auto filter = Placeholder(s.WithOpName("filter"), DT_FLOAT, filter_shape);
//     auto bias = Placeholder(s.WithOpName("bias"), DT_FLOAT, bias_shape);

//     auto matmul = ops::MatMul(s.WithOpName("matmul"), input, filter);
//     Output bias_add;
//     if (add_op == "BiasAdd")
//       bias_add = ops::BiasAdd(s.WithOpName("bias_add"), matmul, bias);
//     else if (add_op == "AddV2")
//       bias_add = ops::AddV2(s.WithOpName("bias_add"), matmul, bias);
//     else if (add_op == "Add")
//       bias_add = ops::Add(s.WithOpName("bias_add"), bias, matmul);

//     auto fetch = s.WithOpName("fetch");
//     auto add = ops::Add(s.WithOpName("add"), bias_add, input_add);

//     ops::Identity(fetch, add);

//     auto input_tensor = GenerateRandomTensor<DT_FLOAT>(
//         TensorShape(input_shape.shape_.dim_sizes()));
//     auto input_add_tensor = GenerateRandomTensor<DT_FLOAT>(
//         TensorShape(input_shape_add.shape_.dim_sizes()));
//     auto filter_tensor = GenerateRandomTensor<DT_FLOAT>(
//         TensorShape(filter_shape.shape_.dim_sizes()));
//     auto bias_tensor = GenerateRandomTensor<DT_FLOAT>(
//         TensorShape(bias_shape.shape_.dim_sizes()));

//     GrapplerItem item;
//     item.fetch = {"fetch"};
//     item.feed = {{"input", input_tensor},
//                  {"filter", filter_tensor},
//                  {"bias", bias_tensor},
//                  {"input_add", input_add_tensor}};
//     TF_CHECK_OK(s.ToGraphDef(&item.graph));

//     // Place all nodes on CPU.
//     for (int i = 0; i < item.graph.node_size(); ++i) {
//       item.graph.mutable_node(i)->set_device("/device:CPU:0");
//     }

//     Remapper optimizer(RewriterConfig::AGGRESSIVE);
//     GraphDef output;
//     TF_CHECK_OK(optimizer.Optimize(nullptr, item, &output));

//     int found = 0;
//     for (const NodeDef& node : output.node()) {
//       auto fetch_node_name = "add";
//       if (node.name() == fetch_node_name) {
//         EXPECT_EQ("_FusedMatMul", node.op());
//         EXPECT_EQ("input", node.input(0));
//         EXPECT_EQ("filter", node.input(1));
//         EXPECT_EQ(2, node.attr().at("num_args").i());
//         EXPECT_EQ("bias", node.input(2));
//         EXPECT_EQ("input_add", node.input(3));

//         const auto fused_ops = node.attr().at("fused_ops").list().s();
//         EXPECT_EQ(2, fused_ops.size());
//         EXPECT_EQ("BiasAdd", fused_ops[0]);
//         EXPECT_EQ("Add", fused_ops[1]);
//         found++;
//       }
//     }
//     EXPECT_EQ(1, found);

//     auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
//     auto tensors = EvaluateNodes(output, item.fetch, item.feed);
//     EXPECT_EQ(1, tensors_expected.size());
//     EXPECT_EQ(1, tensors.size());
//     test::ExpectClose(tensors_expected[0], tensors[0], 0, 1e-5);
//   }
// }

















// TODO: hebi added
// TEST_F(MklRemapperTest, FuseTowerMatMul) {
//   using ::tensorflow::ops::Placeholder;

//   // TODO: more types of adding op support 
//   // for (const string& add_op : {"BiasAdd", "AddV2", "Add"}) {

//   tensorflow::Scope s = tensorflow::Scope::NewRootScope();

//   auto input_shape = ops::Placeholder::Shape({50, 775});
//   auto filter_shape = ops::Placeholder::Shape({775, 256});
//   auto bias_shape = ops::Placeholder::Shape({256});

//   auto input = Placeholder(s.WithOpName("input"), DT_FLOAT, input_shape);
//   auto filter1 = Placeholder(s.WithOpName("filter1"), DT_FLOAT, filter_shape);
//   auto bias1 = Placeholder(s.WithOpName("bias1"), DT_FLOAT, bias_shape);

//   auto filter2 = Placeholder(s.WithOpName("filter2"), DT_FLOAT, filter_shape);
//   auto bias2 = Placeholder(s.WithOpName("bias2"), DT_FLOAT, bias_shape);

//   auto filter3 = Placeholder(s.WithOpName("filter3"), DT_FLOAT, filter_shape);
//   auto bias3 = Placeholder(s.WithOpName("bias3"), DT_FLOAT, bias_shape);

//   auto matmul1 = ops::MatMul(s.WithOpName("matmul1"), input, filter1);
//   auto bias_add1 = ops::BiasAdd(s.WithOpName("bias_add"), matmul1, bias1);
//   Output relu1 = ops::Relu(s.WithOpName("relu1"), bias_add1);

//   auto matmul2 = ops::MatMul(s.WithOpName("matmul2"), input, filter2);
//   auto bias_add2 = ops::BiasAdd(s.WithOpName("bias_add"), matmul2, bias2);
//   Output relu2 = ops::Relu(s.WithOpName("relu2"), bias_add2);

//   auto matmul3 = ops::MatMul(s.WithOpName("matmul3"), input, filter3);
//   auto bias_add3 = ops::BiasAdd(s.WithOpName("bias_add"), matmul3, bias3);
//   Output relu3 = ops::Relu(s.WithOpName("relu3"), bias_add3);
  
//   auto addn_out = ops::AddN(s.WithOpName("output"), {relu1, relu2, relu3});

//   auto fetch = s.WithOpName("fetch");
//   ops::Identity(fetch, addn_out);

//   auto input_tensor = GenerateRandomTensor<DT_FLOAT>(
//       TensorShape(input_shape.shape_.dim_sizes()));
//   auto filter_tensor1 = GenerateRandomTensor<DT_FLOAT>(
//       TensorShape(filter_shape.shape_.dim_sizes()));
//   auto bias_tensor1 = GenerateRandomTensor<DT_FLOAT>(
//       TensorShape(bias_shape.shape_.dim_sizes()));

//   auto filter_tensor2 = GenerateRandomTensor<DT_FLOAT>(
//       TensorShape(filter_shape.shape_.dim_sizes()));
//   auto bias_tensor2 = GenerateRandomTensor<DT_FLOAT>(
//       TensorShape(bias_shape.shape_.dim_sizes()));

//   auto filter_tensor3 = GenerateRandomTensor<DT_FLOAT>(
//       TensorShape(filter_shape.shape_.dim_sizes()));
//   auto bias_tensor3 = GenerateRandomTensor<DT_FLOAT>(
//       TensorShape(bias_shape.shape_.dim_sizes()));

//   GrapplerItem item;
//   item.fetch = {"fetch"};
//   item.feed = {{"input", input_tensor},
//                 {"filter1", filter_tensor1},
//                 {"bias1", bias_tensor1},
//                 {"filter2", filter_tensor2},
//                 {"bias2", bias_tensor2},
//                 {"filter3", filter_tensor3},
//                 {"bias3", bias_tensor3}
//               };

//   TF_CHECK_OK(s.ToGraphDef(&item.graph));

//   // Place all nodes on CPU.
//   for (int i = 0; i < item.graph.node_size(); ++i) {
//     item.graph.mutable_node(i)->set_device("/device:CPU:0");
//   }

//   std::cout << "hebi-dbg: before graph def: " << item.graph.DebugString() << "\n";

//   Remapper optimizer(RewriterConfig::AGGRESSIVE);
//   GraphDef output;
//   TF_CHECK_OK(optimizer.Optimize(nullptr, item, &output));

//   std::cout << "hebi-dbg: after graph def: " << output.DebugString() << "\n";

//   // int found = 0;
//   // for (const NodeDef& node : output.node()) {
//   //   auto fetch_node_name = "add";
//   //   if (node.name() == fetch_node_name) {
//   //     EXPECT_EQ("_FusedMatMul", node.op());
//   //     EXPECT_EQ("input", node.input(0));
//   //     EXPECT_EQ("filter", node.input(1));
//   //     EXPECT_EQ(2, node.attr().at("num_args").i());
//   //     EXPECT_EQ("bias", node.input(2));
//   //     EXPECT_EQ("input_add", node.input(3));

//   //     const auto fused_ops = node.attr().at("fused_ops").list().s();
//   //     EXPECT_EQ(2, fused_ops.size());
//   //     EXPECT_EQ("BiasAdd", fused_ops[0]);
//   //     EXPECT_EQ("Add", fused_ops[1]);
//   //     found++;
//   //   }
//   // }
//   // EXPECT_EQ(1, found);

//   std::cout << "hebi-dbg: Benchmarking old graph: ";
//   BenchmarkNodes(item.graph, item.fetch, item.feed);
//   std::cout << "hebi-dbg: Benchmarking optimized graph: ";
//   BenchmarkNodes(output, item.fetch, item.feed);
//   std::cout << "Benchmark done\n\n";


//   auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
//   auto tensors = EvaluateNodes(output, item.fetch, item.feed);
//   EXPECT_EQ(1, tensors_expected.size());
//   EXPECT_EQ(1, tensors.size());
//   test::ExpectClose(tensors_expected[0], tensors[0], 0, 1e-5);
// }

























TEST_F(MklRemapperTest, FuseTowerMatMulBytedanceConst) {
  using ::tensorflow::ops::Placeholder;

  // TODO: more types of adding op support 
  // for (const string& add_op : {"BiasAdd", "AddV2", "Add"}) {

  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  // auto input_shape = ops::Placeholder::Shape({50, 775});
  // auto filter_shape = TensorShape({775, 256});
  // auto bias_shape = TensorShape({256});

  // auto input_shape = ops::Placeholder::Shape({5, 225});
  // auto filter1_shape = TensorShape({225, 80});
  // auto bias1_shape = TensorShape({80});

  // auto filter2_shape = TensorShape({80, 32});
  // auto bias2_shape = TensorShape({32});

  // auto filter3_shape = TensorShape({32, 1});
  // auto bias3_shape = TensorShape({1});

  auto input_shape = ops::Placeholder::Shape({2, 3});
  auto filter1_shape = TensorShape({3, 2});
  auto bias1_shape = TensorShape({2});

  auto filter2_shape = TensorShape({2, 3});
  auto bias2_shape = TensorShape({3});

  auto filter3_shape = TensorShape({3, 1});
  auto bias3_shape = TensorShape({1});


  auto input = Placeholder(s.WithOpName("input"), DT_FLOAT, input_shape);
  int num_tower = 8;

  OutputList outputs;
  for (int i=0; i<num_tower; i++) {
    // layer #1
    Tensor filter1_data(DT_FLOAT, filter1_shape);
    test::FillIota<float>(&filter1_data, 1.0f);
    auto filter1 = ops::Const(s.WithOpName(absl::StrCat("real_filter_0_", i)), Input::Initializer(filter1_data));

    Tensor bias1_data(DT_FLOAT, bias1_shape);
    test::FillIota<float>(&bias1_data, 1.0f);
    auto bias1 = ops::Const(s.WithOpName(absl::StrCat("real_bias_0_", i)), Input::Initializer(bias1_data));

    auto matmul1 = ops::MatMul(s.WithOpName(absl::StrCat("real_matmul_0_", i)), input, filter1);
    auto bias_add1 = ops::BiasAdd(s.WithOpName(absl::StrCat("real_bias_add_0_", i)), matmul1, bias1);
    auto relu1 = ops::Relu(s.WithOpName(absl::StrCat("real_relu_0_", i)), bias_add1);

    // layer #2
    Tensor filter2_data(DT_FLOAT, filter2_shape);
    test::FillIota<float>(&filter2_data, 1.0f);
    auto filter2 = ops::Const(s.WithOpName(absl::StrCat("real_filter_1_", i)), Input::Initializer(filter2_data));

    Tensor bias2_data(DT_FLOAT, bias2_shape);
    test::FillIota<float>(&bias2_data, 1.0f);
    auto bias2 = ops::Const(s.WithOpName(absl::StrCat("real_bias_1_", i)), Input::Initializer(bias2_data));

    auto matmul2 = ops::MatMul(s.WithOpName(absl::StrCat("real_matmul_1_", i)), relu1, filter2);
    auto bias_add2 = ops::BiasAdd(s.WithOpName(absl::StrCat("real_bias_add_1_", i)), matmul2, bias2);
    // auto relu2 = ops::Relu(s.WithOpName(absl::StrCat("real_relu_1_", i)), bias_add2);

    // // layer #3
    // Tensor filter3_data(DT_FLOAT, filter3_shape);
    // test::FillIota<float>(&filter3_data, 1.0f);
    // auto filter3 = ops::Const(s.WithOpName(absl::StrCat("real_filter_2_", i)), Input::Initializer(filter3_data));

    // Tensor bias3_data(DT_FLOAT, bias3_shape);
    // test::FillIota<float>(&bias3_data, 1.0f);
    // auto bias3 = ops::Const(s.WithOpName(absl::StrCat("real_bias_2_", i)), Input::Initializer(bias3_data));

    // auto matmul3 = ops::MatMul(s.WithOpName(absl::StrCat("real_matmul_2_", i)), relu2, filter3);
    // auto bias_add3 = ops::BiasAdd(s.WithOpName(absl::StrCat("real_bias_add_2_", i)), matmul3, bias3);

    // layer #4 sum
    Tensor indices_data(DT_INT32, TensorShape({}));
    test::FillIota<int>(&indices_data, 1);
    auto indices = ops::Const(s.WithOpName(absl::StrCat("real_indices_2_", i)), Input::Initializer(indices_data));
    Output sum = ops::Sum(s.WithOpName(absl::StrCat("real_sum_3_", i)), bias_add2, indices);
    // Output sum = ops::Sum(s.WithOpName(absl::StrCat("real_sum_3_", i)), relu2, indices);

    outputs.push_back(sum);
  }
  
  // auto addn_out = ops::AddN(s.WithOpName("output"), {outputs[0], outputs[1]});
  auto addn_out = ops::AddN(s.WithOpName("output"), outputs);

  auto fetch = s.WithOpName("fetch");
  ops::Identity(fetch, addn_out);

  auto input_tensor = GenerateRandomTensor<DT_FLOAT>(
      TensorShape(input_shape.shape_.dim_sizes()));

  GrapplerItem item;
  // item.fetch = {"real_relu_1_0"};
  item.fetch = {"output"};
  item.feed = {{"input", input_tensor},
              };

  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  // Place all nodes on CPU.
  for (int i = 0; i < item.graph.node_size(); ++i) {
    item.graph.mutable_node(i)->set_device("/device:CPU:0");
  }


  // std::cout << "hebi-dbg: Benchmarking old graph: ";

  // GraphDef cf_output;
  // ConstantFolding cf_optimizer(/*cpu_device=*/nullptr);
  // Status status = cf_optimizer.Optimize(/*cluster=*/nullptr, item, &cf_output);
  
  // std::cout << "hebi-dbg: before graph def: " << item.graph.DebugString() << "\n";
  // BenchmarkNodes(cf_output, item.fetch, item.feed);

  Remapper remapper_optimizer(RewriterConfig::AGGRESSIVE);
  GraphDef output;
  TF_CHECK_OK(remapper_optimizer.Optimize(nullptr, item, &output));

  item.graph = output;

  // status = cf_optimizer.Optimize(/*cluster=*/nullptr, item, &output);

  std::cout << "hebi-dbg: after optimizing, graph def: " << output.DebugString() << "\n";

  std::cout << "hebi-dbg: Benchmarking optimized graph: \n";
  BenchmarkNodes(output, item.fetch, item.feed);
  std::cout << "Benchmark done\n\n";


  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
  auto tensors = EvaluateNodes(output, item.fetch, item.feed);
  EXPECT_EQ(1, tensors_expected.size());
  EXPECT_EQ(1, tensors.size());
  test::ExpectClose(tensors_expected[0], tensors[0], 0, 1e-5);
}




// TEST_F(MklRemapperTest, FuseTowerMatMulConstSplitVer) {
//   using ::tensorflow::ops::Placeholder;

//   // TODO: more types of adding op support 
//   // for (const string& add_op : {"BiasAdd", "AddV2", "Add"}) {

//   tensorflow::Scope s = tensorflow::Scope::NewRootScope();

//   // auto input_shape = ops::Placeholder::Shape({50, 775});
//   // auto filter_shape = TensorShape({775, 256});
//   // auto bias_shape = TensorShape({256});

//   // auto input_shape = ops::Placeholder::Shape({5, 225});
//   // auto filter1_shape = TensorShape({225, 80});
//   // auto bias1_shape = TensorShape({80});

//   // auto filter2_shape = TensorShape({80, 32});
//   // auto bias2_shape = TensorShape({32});

//   // auto filter3_shape = TensorShape({32, 1});
//   // auto bias3_shape = TensorShape({1});

//   auto input_shape = ops::Placeholder::Shape({2, 3});
//   auto filter1_shape = TensorShape({3, 2});
//   auto bias1_shape = TensorShape({2});

//   auto filter2_shape = TensorShape({2, 3});
//   auto bias2_shape = TensorShape({3});

//   auto filter3_shape = TensorShape({3, 1});
//   auto bias3_shape = TensorShape({1});


//   auto input = Placeholder(s.WithOpName("input"), DT_FLOAT, input_shape);
//   int num_tower = 2;

//   OutputList outputs;
//   OutputList filter1_list;
//   OutputList bias1_list;
//   for (int i=0; i<num_tower; i++) {
//     // layer #1
//     Tensor filter1_data(DT_FLOAT, filter1_shape);
//     test::FillIota<float>(&filter1_data, 1.0f);
//     auto filter1 = ops::Const(s.WithOpName(absl::StrCat("real_filter_0_", i)), Input::Initializer(filter1_data));
//     filter1_list.push_back(filter1);

//     Tensor bias1_data(DT_FLOAT, bias1_shape);
//     test::FillIota<float>(&bias1_data, 1.0f);
//     auto bias1 = ops::Const(s.WithOpName(absl::StrCat("real_bias_0_", i)), Input::Initializer(bias1_data));
//     bias1_list.push_back(bias1);

//     // auto matmul1 = ops::MatMul(s.WithOpName(absl::StrCat("real_matmul_0_", i)), input, filter1);
//     // auto bias_add1 = ops::BiasAdd(s.WithOpName(absl::StrCat("real_bias_add_0_", i)), matmul1, bias1);
//     // auto relu1 = ops::Relu(s.WithOpName(absl::StrCat("real_relu_0_", i)), bias_add1);

//   }

//   auto filteraxis = ops::Const(s.WithOpName("real_filter_axis"), 1, {});
//   Output filterall = ops::Concat(s.WithOpName("real_filter_all_0_"), filter1_list, filteraxis);

//   auto biasaxis = ops::Const(s.WithOpName("real_bias_axis"), 0, {});
//   Output biasall = ops::Concat(s.WithOpName("real_bias_0_"), bias1_list, biasaxis);

//   Output matmul = ops::MatMul(s.WithOpName("real_matmul_0_"), input, filterall);
//   Output bias_add = ops::BiasAdd(s.WithOpName("real_bias_add_0_"), matmul, biasall);
//   Output relu = ops::Relu(s.WithOpName("real_relu_0_"), bias_add);


//   auto split_axis = ops::Const(s.WithOpName("split_axis"), 1, {});
//   OutputList split = ops::Split(s.WithOpName("split"), split_axis, relu, num_tower).output;



//   for (int i=0; i<num_tower; i++) {
//     // layer #2
//     Tensor filter2_data(DT_FLOAT, filter2_shape);
//     test::FillIota<float>(&filter2_data, 1.0f);
//     auto filter2 = ops::Const(s.WithOpName(absl::StrCat("real_filter_1_", i)), Input::Initializer(filter2_data));

//     Tensor bias2_data(DT_FLOAT, bias2_shape);
//     test::FillIota<float>(&bias2_data, 1.0f);
//     auto bias2 = ops::Const(s.WithOpName(absl::StrCat("real_bias_1_", i)), Input::Initializer(bias2_data));

//     auto matmul2 = ops::MatMul(s.WithOpName(absl::StrCat("real_matmul_1_", i)), split.at(i), filter2);
//     auto bias_add2 = ops::BiasAdd(s.WithOpName(absl::StrCat("real_bias_add_1_", i)), matmul2, bias2);
//     auto relu2 = ops::Relu(s.WithOpName(absl::StrCat("real_relu_1_", i)), bias_add2);

//     // layer #3
//     Tensor filter3_data(DT_FLOAT, filter3_shape);
//     test::FillIota<float>(&filter3_data, 1.0f);
//     auto filter3 = ops::Const(s.WithOpName(absl::StrCat("real_filter_2_", i)), Input::Initializer(filter3_data));

//     Tensor bias3_data(DT_FLOAT, bias3_shape);
//     test::FillIota<float>(&bias3_data, 1.0f);
//     auto bias3 = ops::Const(s.WithOpName(absl::StrCat("real_bias_2_", i)), Input::Initializer(bias3_data));

//     auto matmul3 = ops::MatMul(s.WithOpName(absl::StrCat("real_matmul_2_", i)), relu2, filter3);
//     auto bias_add3 = ops::BiasAdd(s.WithOpName(absl::StrCat("real_bias_add_2_", i)), matmul3, bias3);

//     // layer #4 sum
//     Tensor indices_data(DT_INT32, TensorShape({}));
//     test::FillIota<int>(&indices_data, 1);
//     auto indices = ops::Const(s.WithOpName(absl::StrCat("real_indices_2_", i)), Input::Initializer(indices_data));
//     Output sum = ops::Sum(s.WithOpName(absl::StrCat("real_sum_3_", i)), bias_add3, indices);

//     outputs.push_back(sum);
//   }
  
//   // auto addn_out = ops::AddN(s.WithOpName("output"), {outputs[0], outputs[1]});
//   auto addn_out = ops::AddN(s.WithOpName("output"), outputs);

//   auto fetch = s.WithOpName("fetch");
//   ops::Identity(fetch, addn_out);

//   auto input_tensor = GenerateRandomTensor<DT_FLOAT>(
//       TensorShape(input_shape.shape_.dim_sizes()));

//   GrapplerItem item;
//   item.fetch = {"fetch"};
//   item.feed = {{"input", input_tensor},
//               };

//   TF_CHECK_OK(s.ToGraphDef(&item.graph));

//   // Place all nodes on CPU.
//   for (int i = 0; i < item.graph.node_size(); ++i) {
//     item.graph.mutable_node(i)->set_device("/device:CPU:0");
//   }


//   // std::cout << "hebi-dbg: Benchmarking old graph: ";

//   // GraphDef cf_output;
//   // ConstantFolding cf_optimizer(/*cpu_device=*/nullptr);
//   // Status status = cf_optimizer.Optimize(/*cluster=*/nullptr, item, &cf_output);
  
//   // std::cout << "hebi-dbg: before graph def: " << item.graph.DebugString() << "\n";
//   // BenchmarkNodes(cf_output, item.fetch, item.feed);

//   Remapper remapper_optimizer(RewriterConfig::AGGRESSIVE);
//   GraphDef output;
//   TF_CHECK_OK(remapper_optimizer.Optimize(nullptr, item, &output));

//   item.graph = output;

//   // status = cf_optimizer.Optimize(/*cluster=*/nullptr, item, &output);

//   std::cout << "hebi-dbg: after graph def: " << output.DebugString() << "\n";

//   std::cout << "hebi-dbg: Benchmarking optimized graph: ";
//   BenchmarkNodes(output, item.fetch, item.feed);
//   std::cout << "Benchmark done\n\n";


//   auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
//   auto tensors = EvaluateNodes(output, item.fetch, item.feed);
//   EXPECT_EQ(1, tensors_expected.size());
//   EXPECT_EQ(1, tensors.size());
//   test::ExpectClose(tensors_expected[0], tensors[0], 0, 1e-5);
// }








// TEST_F(MklRemapperTest, FuseTowerMatMulBMMVer) {
//   using ::tensorflow::ops::Placeholder;

//   // TODO: more types of adding op support 
//   // for (const string& add_op : {"BiasAdd", "AddV2", "Add"}) {

//   tensorflow::Scope s = tensorflow::Scope::NewRootScope();

//   // auto input_shape = ops::Placeholder::Shape({50, 775});
//   // auto filter_shape = TensorShape({775, 256});
//   // auto bias_shape = TensorShape({256});

//   // auto input_shape = ops::Placeholder::Shape({5, 225});
//   // auto filter1_shape = TensorShape({225, 80});
//   // auto bias1_shape = TensorShape({80});

//   // auto filter2_shape = TensorShape({80, 32});
//   // auto bias2_shape = TensorShape({32});

//   // auto filter3_shape = TensorShape({32, 1});
//   // auto bias3_shape = TensorShape({1});

//   auto input_shape = ops::Placeholder::Shape({2, 3});
//   auto filter1_shape = TensorShape({3, 2});
//   auto bias1_shape = TensorShape({2});

//   auto filter2_shape = TensorShape({2, 3});
//   auto bias2_shape = TensorShape({3});

//   auto filter3_shape = TensorShape({3, 1});
//   auto bias3_shape = TensorShape({1});


//   auto input = Placeholder(s.WithOpName("input"), DT_FLOAT, input_shape);
//   int num_tower = 2;

//   OutputList outputs;
//   OutputList filter1_list;
//   OutputList bias1_list;
//   for (int i=0; i<num_tower; i++) {
//     // layer #1
//     Tensor filter1_data(DT_FLOAT, filter1_shape);
//     test::FillIota<float>(&filter1_data, 1.0f);
//     auto filter1 = ops::Const(s.WithOpName(absl::StrCat("real_filter_0_", i)), Input::Initializer(filter1_data));
//     filter1_list.push_back(filter1);

//     Tensor bias1_data(DT_FLOAT, bias1_shape);
//     test::FillIota<float>(&bias1_data, 1.0f);
//     auto bias1 = ops::Const(s.WithOpName(absl::StrCat("real_bias_0_", i)), Input::Initializer(bias1_data));
//     bias1_list.push_back(bias1);

//     // auto matmul1 = ops::MatMul(s.WithOpName(absl::StrCat("real_matmul_0_", i)), input, filter1);
//     // auto bias_add1 = ops::BiasAdd(s.WithOpName(absl::StrCat("real_bias_add_0_", i)), matmul1, bias1);
//     // auto relu1 = ops::Relu(s.WithOpName(absl::StrCat("real_relu_0_", i)), bias_add1);

//   }

//   // auto filteraxis = ops::Const(s.WithOpName("real_filter_axis"), 1, {});
//   // Output filterall = ops::Concat(s.WithOpName("real_filter_all_0_"), filter1_list, filteraxis);

//   Output filterstackall = ops::Stack(s.WithOpName("real_pack_filter_all_0_"), filter1_list, ops::Stack::Axis(0));


//   // auto biasaxis = ops::Const(s.WithOpName("real_bias_axis"), 0, {});
//   // Output biasall = ops::Concat(s.WithOpName("real_bias_0_"), bias1_list, biasaxis);
//   Output biasstackall = ops::Stack(s.WithOpName("real_pack_bias_all_0_"), bias1_list, ops::Stack::Axis(0));


//   Output batchmatmul = ops::BatchMatMulV2(s.WithOpName("real_batch_matmul_0_"), input, filterstackall);
//   // Output bias_add = ops::BiasAdd(s.WithOpName("real_bias_add_0_"), batchmatmul, bias1_list[0]);
//   Output bias_add = ops::AddV2(s.WithOpName("real_bias_add_0_"), batchmatmul, biasstackall);
//   Output relu = ops::Relu(s.WithOpName("real_relu_0_"), bias_add);


//   // auto split_axis = ops::Const(s.WithOpName("split_axis"), 1, {});
//   // OutputList split = ops::Split(s.WithOpName("split"), split_axis, relu, num_tower).output;



//   // for (int i=0; i<num_tower; i++) {
//   //   // layer #2
//   //   Tensor filter2_data(DT_FLOAT, filter2_shape);
//   //   test::FillIota<float>(&filter2_data, 1.0f);
//   //   auto filter2 = ops::Const(s.WithOpName(absl::StrCat("real_filter_1_", i)), Input::Initializer(filter2_data));

//   //   Tensor bias2_data(DT_FLOAT, bias2_shape);
//   //   test::FillIota<float>(&bias2_data, 1.0f);
//   //   auto bias2 = ops::Const(s.WithOpName(absl::StrCat("real_bias_1_", i)), Input::Initializer(bias2_data));

//   //   auto matmul2 = ops::MatMul(s.WithOpName(absl::StrCat("real_matmul_1_", i)), split.at(i), filter2);
//   //   auto bias_add2 = ops::BiasAdd(s.WithOpName(absl::StrCat("real_bias_add_1_", i)), matmul2, bias2);
//   //   auto relu2 = ops::Relu(s.WithOpName(absl::StrCat("real_relu_1_", i)), bias_add2);

//   //   // layer #3
//   //   Tensor filter3_data(DT_FLOAT, filter3_shape);
//   //   test::FillIota<float>(&filter3_data, 1.0f);
//   //   auto filter3 = ops::Const(s.WithOpName(absl::StrCat("real_filter_2_", i)), Input::Initializer(filter3_data));

//   //   Tensor bias3_data(DT_FLOAT, bias3_shape);
//   //   test::FillIota<float>(&bias3_data, 1.0f);
//   //   auto bias3 = ops::Const(s.WithOpName(absl::StrCat("real_bias_2_", i)), Input::Initializer(bias3_data));

//   //   auto matmul3 = ops::MatMul(s.WithOpName(absl::StrCat("real_matmul_2_", i)), relu2, filter3);
//   //   auto bias_add3 = ops::BiasAdd(s.WithOpName(absl::StrCat("real_bias_add_2_", i)), matmul3, bias3);

//   //   // layer #4 sum
//   //   Tensor indices_data(DT_INT32, TensorShape({}));
//   //   test::FillIota<int>(&indices_data, 1);
//   //   auto indices = ops::Const(s.WithOpName(absl::StrCat("real_indices_2_", i)), Input::Initializer(indices_data));
//   //   Output sum = ops::Sum(s.WithOpName(absl::StrCat("real_sum_3_", i)), bias_add3, indices);

//   //   outputs.push_back(sum);
//   // }
  
//   // // auto addn_out = ops::AddN(s.WithOpName("output"), {outputs[0], outputs[1]});
//   // auto addn_out = ops::AddN(s.WithOpName("output"), outputs);

//   auto fetch = s.WithOpName("fetch");
//   ops::Identity(fetch, relu);

//   auto input_tensor = GenerateRandomTensor<DT_FLOAT>(
//       TensorShape(input_shape.shape_.dim_sizes()));

//   GrapplerItem item;
//   item.fetch = {"fetch"};
//   item.feed = {{"input", input_tensor},
//               };

//   TF_CHECK_OK(s.ToGraphDef(&item.graph));

//   // Place all nodes on CPU.
//   for (int i = 0; i < item.graph.node_size(); ++i) {
//     item.graph.mutable_node(i)->set_device("/device:CPU:0");
//   }


//   // std::cout << "hebi-dbg: Benchmarking old graph: ";

//   // GraphDef cf_output;
//   // ConstantFolding cf_optimizer(/*cpu_device=*/nullptr);
//   // Status status = cf_optimizer.Optimize(/*cluster=*/nullptr, item, &cf_output);
  
//   // std::cout << "hebi-dbg: before graph def: " << item.graph.DebugString() << "\n";
//   // BenchmarkNodes(cf_output, item.fetch, item.feed);

//   Remapper remapper_optimizer(RewriterConfig::AGGRESSIVE);
//   GraphDef output;
//   TF_CHECK_OK(remapper_optimizer.Optimize(nullptr, item, &output));

//   item.graph = output;

//   // status = cf_optimizer.Optimize(/*cluster=*/nullptr, item, &output);

//   std::cout << "hebi-dbg: after graph def: " << output.DebugString() << "\n";

//   std::cout << "hebi-dbg: Benchmarking optimized graph: ";
//   BenchmarkNodes(output, item.fetch, item.feed);
//   std::cout << "Benchmark done\n\n";


//   auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
//   auto tensors = EvaluateNodes(output, item.fetch, item.feed);
//   EXPECT_EQ(1, tensors_expected.size());
//   EXPECT_EQ(1, tensors.size());
//   test::ExpectClose(tensors_expected[0], tensors[0], 0, 1e-5);
// }



























// TEST_F(MklRemapperTest, FuseTowerMatMulAfter) {
//   using ::tensorflow::ops::Placeholder;

//   // TODO: more types of adding op support 
//   // for (const string& add_op : {"BiasAdd", "AddV2", "Add"}) {

//   tensorflow::Scope s = tensorflow::Scope::NewRootScope();

//   auto input_shape = ops::Placeholder::Shape({4, 32});
//   auto filter_shape = ops::Placeholder::Shape({32, 8});
//   auto bias_shape = ops::Placeholder::Shape({8});

//   auto input = Placeholder(s.WithOpName("input"), DT_FLOAT, input_shape);

//   Output filter1 = Placeholder(s.WithOpName("filter1"), DT_FLOAT, filter_shape);
//   Output bias1 = Placeholder(s.WithOpName("bias1"), DT_FLOAT, bias_shape);

//   Output filter2 = Placeholder(s.WithOpName("filter2"), DT_FLOAT, filter_shape);
//   Output bias2 = Placeholder(s.WithOpName("bias2"), DT_FLOAT, bias_shape);

//   Output filter3 = Placeholder(s.WithOpName("filter3"), DT_FLOAT, filter_shape);
//   Output bias3 = Placeholder(s.WithOpName("bias3"), DT_FLOAT, bias_shape);


//   auto filteraxis = ops::Const(s.WithOpName("filteraxis"), 1, {});
//   Output filterall = ops::Concat(s.WithOpName("filterall"), {filter1, filter2, filter3}, filteraxis);

//   auto biasaxis = ops::Const(s.WithOpName("biasaxis"), 0, {});
//   Output biasall = ops::Concat(s.WithOpName("biasall"), {bias1, bias2, bias3}, biasaxis);

//   Output matmul = ops::MatMul(s.WithOpName("matmul"), input, filterall);
//   Output bias_add = ops::BiasAdd(s.WithOpName("bias_add"), matmul, biasall);
//   Output relu = ops::Relu(s.WithOpName("relu"), bias_add);


//   auto split_axis = ops::Const(s.WithOpName("split_axis"), 1, {});
//   OutputList split = ops::Split(s.WithOpName("split"), split_axis, relu, 3).output;

//   auto addn_out = ops::AddN(s.WithOpName("output"), split);


//   auto fetch = s.WithOpName("fetch");
//   ops::Identity(fetch, addn_out);

//   auto input_tensor = GenerateRandomTensor<DT_FLOAT>(
//       TensorShape(input_shape.shape_.dim_sizes()));
//   auto filter_tensor1 = GenerateRandomTensor<DT_FLOAT>(
//       TensorShape(filter_shape.shape_.dim_sizes()));
//   auto bias_tensor1 = GenerateRandomTensor<DT_FLOAT>(
//       TensorShape(bias_shape.shape_.dim_sizes()));

//   auto filter_tensor2 = GenerateRandomTensor<DT_FLOAT>(
//       TensorShape(filter_shape.shape_.dim_sizes()));
//   auto bias_tensor2 = GenerateRandomTensor<DT_FLOAT>(
//       TensorShape(bias_shape.shape_.dim_sizes()));

//   auto filter_tensor3 = GenerateRandomTensor<DT_FLOAT>(
//       TensorShape(filter_shape.shape_.dim_sizes()));
//   auto bias_tensor3 = GenerateRandomTensor<DT_FLOAT>(
//       TensorShape(bias_shape.shape_.dim_sizes()));

//   GrapplerItem item;
//   item.fetch = {"fetch"};
//   item.feed = {{"input", input_tensor},
//                 {"filter1", filter_tensor1},
//                 {"bias1", bias_tensor1},
//                 {"filter2", filter_tensor2},
//                 {"bias2", bias_tensor2},
//                 {"filter3", filter_tensor3},
//                 {"bias3", bias_tensor3}
//               };

//   TF_CHECK_OK(s.ToGraphDef(&item.graph));

//   // Place all nodes on CPU.
//   for (int i = 0; i < item.graph.node_size(); ++i) {
//     item.graph.mutable_node(i)->set_device("/device:CPU:0");
//   }

//   std::cout << "hebi-dbg: before graph def: " << item.graph.DebugString() << "\n";

//   Remapper optimizer(RewriterConfig::AGGRESSIVE);
//   GraphDef output;
//   TF_CHECK_OK(optimizer.Optimize(nullptr, item, &output));

//   std::cout << "hebi-dbg: after graph def: " << output.DebugString() << "\n";

//   // int found = 0;
//   // for (const NodeDef& node : output.node()) {
//   //   auto fetch_node_name = "add";
//   //   if (node.name() == fetch_node_name) {
//   //     EXPECT_EQ("_FusedMatMul", node.op());
//   //     EXPECT_EQ("input", node.input(0));
//   //     EXPECT_EQ("filter", node.input(1));
//   //     EXPECT_EQ(2, node.attr().at("num_args").i());
//   //     EXPECT_EQ("bias", node.input(2));
//   //     EXPECT_EQ("input_add", node.input(3));

//   //     const auto fused_ops = node.attr().at("fused_ops").list().s();
//   //     EXPECT_EQ(2, fused_ops.size());
//   //     EXPECT_EQ("BiasAdd", fused_ops[0]);
//   //     EXPECT_EQ("Add", fused_ops[1]);
//   //     found++;
//   //   }
//   // }
//   // EXPECT_EQ(1, found);

//   BenchmarkNodes(item.graph, item.fetch, item.feed);
//   BenchmarkNodes(output, item.fetch, item.feed);

//   auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
//   auto tensors = EvaluateNodes(output, item.fetch, item.feed);
//   EXPECT_EQ(1, tensors_expected.size());
//   EXPECT_EQ(1, tensors.size());
//   test::ExpectClose(tensors_expected[0], tensors[0], 0, 1e-5);
// }


























// class RelpaceAddWithBiasAddTest : public GrapplerTest {
//  public:
//   const string kAddOp = "Add";
//   const string kAddV2Op = "AddV2";

//  protected:
//   template <DataType DTYPE>
//   void RelpaceAddWithBiasAddDepthwiseConv2D(const string& add_op) {
//     using ::tensorflow::ops::Placeholder;

//     for (const string& activation : {"None", "Relu", "Relu6", "Elu"}) {
//       tensorflow::Scope s = tensorflow::Scope::NewRootScope();

//       auto input_shape = Placeholder::Shape({8, 32, 32, 3});
//       auto filter_shape = Placeholder::Shape({1, 1, 3, 128});
//       auto bias_shape = Placeholder::Shape({128 * 3});

//       auto input = Placeholder(s.WithOpName("input"), DTYPE, input_shape);
//       auto filter = Placeholder(s.WithOpName("filter"), DTYPE, filter_shape);
//       auto bias = Placeholder(s.WithOpName("bias"), DTYPE, bias_shape);

//       std::vector<int> strides = {1, 1, 1, 1};
//       auto conv = ops::DepthwiseConv2dNative(s.WithOpName("depthwise_conv"),
//                                              input, filter, strides, "SAME");

//       Output bias_add;
//       if (add_op == kAddV2Op) {
//         bias_add = ops::AddV2(s.WithOpName(add_op), conv, bias);
//       } else {
//         bias_add = ops::Add(s.WithOpName(add_op), bias, conv);
//       }

//       ops::Identity fetch = [&]() -> ops::Identity {
//         auto activate = s.WithOpName("activation");
//         auto fetch = s.WithOpName("fetch");

//         if (activation == "Relu") {
//           return ops::Identity(fetch, ops::Relu(activate, bias_add));
//         } else if (activation == "Relu6") {
//           return ops::Identity(fetch, ops::Relu6(activate, bias_add));
//         } else if (activation == "Elu") {
//           return ops::Identity(fetch, ops::Elu(activate, bias_add));
//         }

//         return ops::Identity(fetch, bias_add);
//       }();

//       auto input_t = GenerateRandomTensor<DTYPE>({8, 32, 32, 3});
//       auto filter_t = GenerateRandomTensor<DTYPE>({1, 1, 3, 128});
//       auto bias_t = GenerateRandomTensor<DTYPE>({128 * 3});

//       GrapplerItem item;
//       item.fetch = {"fetch"};
//       item.feed = {{"input", input_t}, {"filter", filter_t}, {"bias", bias_t}};
//       TF_ASSERT_OK(s.ToGraphDef(&item.graph));

//       // Place all nodes on CPU.
//       for (int i = 0; i < item.graph.node_size(); ++i) {
//         item.graph.mutable_node(i)->set_device("/device:CPU:0");
//       }

//       Remapper optimizer(RewriterConfig::AGGRESSIVE);
//       GraphDef output;
//       TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

//       int found = 0;
//       for (const NodeDef& node : output.node()) {
//         if (node.name() == "activation") {
//           EXPECT_EQ(node.op(), "_FusedDepthwiseConv2dNative");
//           ASSERT_GE(node.input_size(), 3);
//           EXPECT_EQ(node.input(0), "input");
//           EXPECT_EQ(node.input(1), "filter");
//           EXPECT_EQ(node.attr().at("num_args").i(), 1);
//           EXPECT_EQ(node.input(2), "bias");

//           const auto fused_ops = node.attr().at("fused_ops").list().s();
//           ASSERT_EQ(fused_ops.size(), 2);
//           EXPECT_EQ(fused_ops[0], "BiasAdd");
//           EXPECT_EQ(fused_ops[1], activation);

//           found++;
//         } else if (node.name() == add_op) {
//           EXPECT_EQ(node.op(), "_FusedDepthwiseConv2dNative");
//           ASSERT_GE(node.input_size(), 3);
//           EXPECT_EQ(node.input(0), "input");
//           EXPECT_EQ(node.input(1), "filter");
//           EXPECT_EQ(node.attr().at("num_args").i(), 1);
//           EXPECT_EQ(node.input(2), "bias");

//           const auto fused_ops = node.attr().at("fused_ops").list().s();
//           ASSERT_EQ(fused_ops.size(), 1);
//           EXPECT_EQ(fused_ops[0], "BiasAdd");
//           found++;
//         }
//       }
//       EXPECT_EQ(found, 1);

//       auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
//       ASSERT_EQ(tensors_expected.size(), 1);
//       auto tensors = EvaluateNodes(output, item.fetch, item.feed);
//       ASSERT_EQ(tensors.size(), 1);

//       if (DTYPE == DT_BFLOAT16)
//         test::ExpectClose(tensors[0], tensors_expected[0], 1e-2, 1e-2);
//       else
//         test::ExpectClose(tensors[0], tensors_expected[0], 1e-6);
//     }
//   }
// };

// #define CREATE_REPLACEADDWITHBIASADD_TEST_1(ops, addop, dtype)              \
//   TEST_F(RelpaceAddWithBiasAddTest, RelpaceAddWithBiasAdd##ops##_##addop) { \
//     RelpaceAddWithBiasAddDepthwiseConv2D<dtype>(#addop);                    \
//   }
// CREATE_REPLACEADDWITHBIASADD_TEST_1(DepthConv2D, AddV2, DT_FLOAT);
// CREATE_REPLACEADDWITHBIASADD_TEST_1(DepthConv2D, Add, DT_FLOAT);

// class FusedMatMulBiasAddAndGeluTest : public GrapplerTest {
//  public:
//   template <DataType DTYPE>
//   void RunTest() {
//     using ::tensorflow::ops::Placeholder;

//     tensorflow::Scope s = tensorflow::Scope::NewRootScope();

//     auto lhs_shape = ops::Placeholder::Shape({8, 32});
//     auto rhs_shape = ops::Placeholder::Shape({32, 64});
//     auto bias_shape = ops::Placeholder::Shape({64});

//     auto lhs = Placeholder(s.WithOpName("lhs"), DTYPE, lhs_shape);
//     auto rhs = Placeholder(s.WithOpName("rhs"), DTYPE, rhs_shape);
//     auto bias = Placeholder(s.WithOpName("bias"), DTYPE, bias_shape);

//     auto matmul = ops::MatMul(s.WithOpName("matmul"), lhs, rhs);
//     auto bias_add = ops::BiasAdd(s.WithOpName("bias_add"), matmul, bias);

//     // Add Gelu approximate with smaller ops
//     auto square_root_one_half =
//         ops::Const(s.WithOpName("square_root_one_half"), {0.707106f}, {});
//     auto bias_add_times_square_root_one_half =
//         ops::Mul(s.WithOpName("bias_add_times_square_root_one_half"), bias_add,
//                  square_root_one_half);
//     auto erf =
//         ops::Erf(s.WithOpName("erf"), bias_add_times_square_root_one_half);
//     auto one = ops::Const(s.WithOpName("one"), {1.0f}, {});
//     auto erf_plus_one = ops::AddV2(s.WithOpName("one_plus_erf"), erf, one);
//     auto one_half = ops::Const(s.WithOpName("one_half"), {0.5f}, {});
//     auto erf_plus_one_times_one_half = ops::Mul(
//         s.WithOpName("erf_plus_one_times_one_half"), erf_plus_one, one_half);
//     auto gelu = ops::Mul(s.WithOpName("fusion_output"),
//                          erf_plus_one_times_one_half, bias_add);
//     auto fetch = ops::Identity(s.WithOpName("fetch"), gelu);

//     auto lhs_t = GenerateTensorWithSetRandom<DTYPE>({8, 32});
//     auto rhs_t = GenerateTensorWithSetRandom<DTYPE>({32, 64});
//     auto bias_t = GenerateTensorWithSetRandom<DTYPE>({64});

//     GrapplerItem item;
//     item.fetch = {"fetch"};
//     item.feed = {{"lhs", lhs_t}, {"rhs", rhs_t}, {"bias", bias_t}};
//     TF_ASSERT_OK(s.ToGraphDef(&item.graph));

//     // Place all nodes on CPU.
//     for (int i = 0; i < item.graph.node_size(); ++i) {
//       item.graph.mutable_node(i)->set_device("/device:CPU:0");
//     }

//     Remapper optimizer(RewriterConfig::ON);
//     GraphDef optimized_graph;
//     TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &optimized_graph));
//     int found = 0;
//     for (const NodeDef& node : optimized_graph.node()) {
//       if (node.name() == "fusion_output") {
//         EXPECT_EQ(node.op(), "_FusedMatMul");
//         ASSERT_GE(node.input_size(), 3);
//         EXPECT_EQ(node.input(0), "lhs");
//         EXPECT_EQ(node.input(1), "rhs");
//         EXPECT_EQ(node.input(2), "bias");
//         EXPECT_EQ(node.attr().at("num_args").i(), 1);
//         const auto fused_ops = node.attr().at("fused_ops").list().s();
//         ASSERT_EQ(fused_ops.size(), 2);
//         EXPECT_EQ(fused_ops[0], "BiasAdd");
//         EXPECT_EQ(fused_ops[1], "GeluExact");
//         found++;
//       }
//     }
//     EXPECT_EQ(1, found);

//     // Evaluate result without remapper fusion
//     auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
//     ASSERT_EQ(tensors_expected.size(), 1);

//     auto tensors_evaluated =
//         EvaluateNodes(optimized_graph, item.fetch, item.feed);
//     ASSERT_EQ(tensors_evaluated.size(), 1);
//     test::ExpectClose(tensors_evaluated[0], tensors_expected[0], 1e-6);
//   }
// };

// // Gelu has two implementations (1) exact and (2) approximate. Exact cannot be
// // used with bfloat16 numeric since the Erf is not supported in bfloat16 yet.
// // Here gelu-exact is tested for float32 numeric only. Gelu-approximate test
// // is added in tensorflow/python/grappler/remapper_test.py, since the pattern is
// // changed by other optimizers before the remapper optimizer.
// TEST_F(FusedMatMulBiasAddAndGeluTest, Float32GeluExact) { RunTest<DT_FLOAT>(); }

// class MklFusedBatchMatMul : public MklRemapperTest {
//  public:
//   template <typename T>
//   void VerifyFused(bool adjx, bool adjy) {
//     using ::tensorflow::ops::Placeholder;
//     using normal_generator = Eigen::internal::NormalRandomGenerator<T>;

//     int b0 = 2;
//     int b1 = 2;
//     int m = 32;
//     int k = 16;
//     int n = 64;

//     tensorflow::Scope s = tensorflow::Scope::NewRootScope();

//     auto input_shape =
//         adjx ? TensorShape({b0, b1, k, m}) : TensorShape({b0, b1, m, k});
//     auto weight_shape =
//         adjy ? TensorShape({b0, b1, n, k}) : TensorShape({b0, b1, k, n});
//     auto add_shape = TensorShape({b0, 1, m, n});

//     auto input_placeholder_shape = ops::Placeholder::Shape(input_shape);
//     auto weight_placeholder_shape = ops::Placeholder::Shape(weight_shape);
//     auto add_placeholder_shape = ops::Placeholder::Shape(add_shape);

//     auto input = Placeholder(s.WithOpName("input"), DataTypeToEnum<T>::v(),
//                              input_placeholder_shape);
//     auto weight = Placeholder(s.WithOpName("weight"), DataTypeToEnum<T>::v(),
//                               weight_placeholder_shape);
//     auto addend = Placeholder(s.WithOpName("addend"), DataTypeToEnum<T>::v(),
//                               add_placeholder_shape);

//     auto batchmatmul =
//         ops::BatchMatMulV2(s.WithOpName("batchmatmul"), input, weight,
//                            ops::BatchMatMulV2::Attrs().AdjX(adjx).AdjY(adjy));
//     auto scale_const = ops::Const(s.WithOpName("scale_const"), {0.1f});
//     auto scale =
//         ops::Cast(s.WithOpName("scale"), scale_const, DataTypeToEnum<T>::v());
//     auto mul = ops::Multiply(s.WithOpName("mul"), batchmatmul, scale);
//     auto add = ops::AddV2(s.WithOpName("add"), mul, addend);
//     auto fetch = ops::Identity(s.WithOpName("fetch"), add);

//     Tensor input_t = Tensor(DataTypeToEnum<T>::v(), input_shape);
//     Tensor weight_t = Tensor(DataTypeToEnum<T>::v(), weight_shape);
//     Tensor add_t = Tensor(DataTypeToEnum<T>::v(), add_shape);
//     input_t.flat<T>() =
//         input_t.flat<T>().template setRandom<normal_generator>();
//     weight_t.flat<T>() =
//         weight_t.flat<T>().template setRandom<normal_generator>();
//     add_t.flat<T>() = add_t.flat<T>().template setRandom<normal_generator>();

//     GrapplerItem item;
//     item.fetch = {"fetch"};
//     item.feed = {{"input", input_t}, {"weight", weight_t}, {"addend", add_t}};
//     TF_CHECK_OK(s.ToGraphDef(&item.graph));

//     // Place all nodes on CPU.
//     for (int i = 0; i < item.graph.node_size(); ++i) {
//       item.graph.mutable_node(i)->set_device("/device:CPU:0");
//     }

//     Remapper optimizer(RewriterConfig::ON);
//     GraphDef output;
//     TF_CHECK_OK(optimizer.Optimize(nullptr, item, &output));

//     int found = 0;
//     for (const NodeDef& node : output.node()) {
//       if (node.name() == "add") {
//         EXPECT_EQ("_MklFusedBatchMatMulV2", node.op());
//         EXPECT_EQ("input", node.input(0));
//         EXPECT_EQ("weight", node.input(1));
//         EXPECT_EQ("scale", node.input(2));
//         EXPECT_EQ("addend", node.input(3));
//         const auto fused_ops = node.attr().at("fused_ops").list().s();
//         EXPECT_EQ(2, fused_ops.size());
//         EXPECT_EQ("Mul", fused_ops[0]);
//         found++;
//         EXPECT_EQ("Add", fused_ops[1]);
//         found++;
//       }
//     }
//     EXPECT_EQ(2, found);

//     auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
//     auto tensors = EvaluateNodes(output, item.fetch, item.feed);
//     std::is_same<T, float>::value
//         ? test::ExpectClose(tensors_expected[0], tensors[0], 1e-6, 1e-6)
//         : test::ExpectClose(tensors_expected[0], tensors[0], 1e-2, 1e-2);
//   }
// };

// TEST_F(MklFusedBatchMatMul, MulAndAdd) {
//   for (const auto adjx : {false, true})
//     for (const auto adjy : {false, true}) {
//       this->VerifyFused<float>(adjx, adjy);
//       this->VerifyFused<bfloat16>(adjx, adjy);
//     }
// }

// class MklRemapperSwishTest : public GrapplerTest {
//  protected:
//   template <DataType DTYPE>
//   void RunTest() {
//     using ::tensorflow::ops::Placeholder;

//     tensorflow::Scope s = tensorflow::Scope::NewRootScope();
//     auto mul_shape = ops::Placeholder::Shape({64, 64});

//     // We will test four sitations:
//     //  1. y = x * sigmoid(x)
//     //  2. y = sigmoid(x) * x
//     //  3. y = sigmoid(x) * sigmoid(sigmoid(x))
//     //  4. y = sigmoid(sigmoid(x)) * sigmoid(x)
//     auto input = Placeholder(s.WithOpName("input"), DTYPE, mul_shape);
//     auto sigmoid1 = ops::Sigmoid(s.WithOpName("sigmoid1"), input);
//     auto sigmoid2 = ops::Sigmoid(s.WithOpName("sigmoid2"), input);
//     auto sigmoid3_1 = ops::Sigmoid(s.WithOpName("sigmoid3_1"), input);
//     auto sigmoid3_2 = ops::Sigmoid(s.WithOpName("sigmoid3_2"), sigmoid3_1);
//     auto sigmoid4_1 = ops::Sigmoid(s.WithOpName("sigmoid4_1"), input);
//     auto sigmoid4_2 = ops::Sigmoid(s.WithOpName("sigmoid4_2"), sigmoid4_1);
//     auto mul1 = ops::Mul(s.WithOpName("mul1"), input, sigmoid1);
//     auto mul2 = ops::Mul(s.WithOpName("mul2"), sigmoid2, input);
//     auto mul3 = ops::Mul(s.WithOpName("mul3"), sigmoid3_1, sigmoid3_2);
//     auto mul4 = ops::Mul(s.WithOpName("mul4"), sigmoid4_2, sigmoid4_1);
//     auto fetch1 = ops::Identity(s.WithOpName("fetch1"), mul1);
//     auto fetch2 = ops::Identity(s.WithOpName("fetch2"), mul2);
//     auto fetch3 = ops::Identity(s.WithOpName("fetch3"), mul3);
//     auto fetch4 = ops::Identity(s.WithOpName("fetch4"), mul4);
//     auto mul_t = GenerateTensorWithSetRandom<DTYPE>({64, 64});

//     GrapplerItem item;
//     item.fetch = {"fetch1", "fetch2", "fetch3", "fetch4"};
//     item.feed = {{"input", mul_t}};
//     TF_ASSERT_OK(s.ToGraphDef(&item.graph));

//     // Place all nodes on CPU.
//     for (int i = 0; i < item.graph.node_size(); ++i) {
//       item.graph.mutable_node(i)->set_device("/device:CPU:0");
//     }

//     Remapper optimizer(RewriterConfig::ON);
//     GraphDef output;
//     TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

//     int found = 0;
//     for (const NodeDef& node : output.node()) {
//       if (node.name() == "mul1") {
//         EXPECT_EQ(node.op(), "_MklSwish");
//         ASSERT_EQ(node.input_size(), 1);
//         EXPECT_EQ(node.input(0), "input");
//         ++found;
//       }
//       if (node.name() == "mul2") {
//         EXPECT_EQ(node.op(), "_MklSwish");
//         ASSERT_EQ(node.input_size(), 1);
//         EXPECT_EQ(node.input(0), "input");
//         ++found;
//       }
//       // mul3 won't be replaced by swish
//       // Coz of the limitation of patternMatcher with commutative op
//       if (node.name() == "mul4") {
//         EXPECT_EQ(node.op(), "_MklSwish");
//         ASSERT_EQ(node.input_size(), 1);
//         EXPECT_EQ(node.input(0), "sigmoid4_1");
//         ++found;
//       }
//     }
//     EXPECT_EQ(found, 3);

//     auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
//     ASSERT_EQ(tensors_expected.size(), 4);
//     auto tensors = EvaluateNodes(output, item.fetch, item.feed);
//     ASSERT_EQ(tensors.size(), 4);
//     float atol = 1e-6, rtol = 1e-6;
//     if (DTYPE == DT_BFLOAT16) {
//       atol = 1e-2;
//       rtol = 1e-2;
//     }
//     test::ExpectClose(tensors[0], tensors_expected[0], atol, rtol);
//     test::ExpectClose(tensors[1], tensors_expected[1], atol, rtol);
//     test::ExpectClose(tensors[2], tensors_expected[2], atol, rtol);
//     test::ExpectClose(tensors[3], tensors_expected[3], atol, rtol);
//   }
// };

// TEST_F(MklRemapperSwishTest, F32) { RunTest<DT_FLOAT>(); }
// TEST_F(MklRemapperSwishTest, BF16) { RunTest<DT_BFLOAT16>(); }

}  // namespace grappler
}  // namespace tensorflow
#endif  // INTEL_MKL && ENABLE_MKL
