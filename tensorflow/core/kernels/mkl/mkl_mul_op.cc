/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/nn_ops.cc.
// #if defined(INTEL_MKL) && defined(ENABLE_MKL)

#include <unordered_map>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "dnnl.hpp"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/mkl_util.h"
#ifdef DNNL_AARCH64_USE_ACL
#include "tensorflow/core/platform/mutex.h"
#endif

using dnnl::algorithm;
using dnnl::binary;
using dnnl::memory;
using dnnl::prop_kind;
using dnnl::stream;

using BinaryPd = dnnl::binary::primitive_desc;

namespace tensorflow {

template <typename T>
class MklBinaryParams {
 public:
  memory::dims src_0_dims;
  memory::desc src_0_md;
  memory::dims src_1_dims;
  memory::desc src_1_md;

  algorithm alg_kind;

  MklBinaryParams(memory::dims src_0_dims, memory::desc src_0_md,
                  memory::dims src_1_dims, memory::desc src_1_md,
                  algorithm alg_kind)
      : src_0_dims(src_0_dims),
        src_0_md(src_0_md),
        src_1_dims(src_1_dims),
        src_1_md(src_1_md),
        alg_kind(alg_kind) {}
};


template <typename T>
class MklBinaryPrimitive : public MklPrimitive {
 public:
  explicit MklBinaryPrimitive(const MklBinaryParams<T>& fwdParams)
      : MklPrimitive(engine(engine::kind::cpu, 0)) {
    // create binary primitive
    if (context_.binary_prim == nullptr) {
      Setup(fwdParams);
    }
  }

  ~MklBinaryPrimitive() {}

  // Eltwise forward execute
  //   src_0_data:  input data buffer 0 of src
  //   src_1_data:  input data buffer 1 of src
  //   dst_data:  output data buffer of dst
  void Execute(const T* src_0_data, const T* src_1_data, T* dst_data,
               std::shared_ptr<stream> fwd_stream) {
#ifdef DNNL_AARCH64_USE_ACL
    mutex_lock lock(primitive_execution_mu_);
#endif
#ifndef ENABLE_ONEDNN_OPENMP
    context_.src_0_mem->set_data_handle(
        static_cast<void*>(const_cast<T*>(src_0_data)), *fwd_stream);
    context_.src_1_mem->set_data_handle(
        static_cast<void*>(const_cast<T*>(src_1_data)), *fwd_stream);
    context_.dst_mem->set_data_handle(static_cast<void*>(dst_data),
                                      *fwd_stream);
#else
    context_.src_0_mem->set_data_handle(
        static_cast<void*>(const_cast<T*>(src_0_data)));
    context_.src_1_mem->set_data_handle(
        static_cast<void*>(const_cast<T*>(src_1_data)));
    context_.dst_mem->set_data_handle(static_cast<void*>(dst_data));
#endif  // !ENABLE_ONEDNN_OPENMP
    DCHECK_EQ(context_.fwd_primitives.size(),
              context_.fwd_primitives_args.size());
    execute_primitives(context_.fwd_primitives, fwd_stream,
                       context_.fwd_primitives_args);

    // After execution, set data handle back.
    context_.src_0_mem->set_data_handle(DummyData);
    context_.src_1_mem->set_data_handle(DummyData);
    context_.dst_mem->set_data_handle(DummyData);
  }

  std::shared_ptr<BinaryPd> GetBinaryPd() { return context_.fwd_pd; }

 private:
  // Primitive reuse context for eltwise Fwd ops: Relu, Elu, Tanh
  struct BinaryContext {
    // oneDNN memory
    std::shared_ptr<memory> src_0_mem;
    std::shared_ptr<memory> src_1_mem;
    std::shared_ptr<memory> dst_mem;

    // desc & primitive desc
    std::shared_ptr<dnnl::binary::desc> fwd_desc;
    std::shared_ptr<BinaryPd> fwd_pd;

    // memory desc
    std::shared_ptr<memory::desc> src_0_md;
    std::shared_ptr<memory::desc> src_1_md;
    std::shared_ptr<memory::desc> dst_md;

    // binary primitive
    std::shared_ptr<dnnl::primitive> binary_prim;

    std::vector<dnnl::primitive> fwd_primitives;

    std::vector<std::unordered_map<int, memory>> fwd_primitives_args;

    BinaryContext()
        : src_0_mem(nullptr),
          src_1_mem(nullptr),
          dst_mem(nullptr),
          fwd_desc(nullptr),
          fwd_pd(nullptr),
          src_0_md(nullptr),
          src_1_md(nullptr),
          dst_md(nullptr),
          binary_prim(nullptr) {}
  };

  // Eltwise forward primitive setup
  void Setup(const MklBinaryParams<T>& fwdParams) {
    // create memory descriptors for eltwise data with specified format
    context_.src_0_md.reset(new memory::desc(fwdParams.src_0_md.data));
    context_.src_1_md.reset(new memory::desc(fwdParams.src_1_md.data));

    // Create an eltwise forward descriptor and primitive descriptor
    context_.fwd_desc.reset(new binary::desc(fwdParams.alg_kind, *context_.src_0_md, *context_.src_1_md, *context_.dst_md));
    context_.fwd_pd.reset(new BinaryPd(*context_.fwd_desc, cpu_engine_));
    auto fwd_pd = context_.fwd_pd.get();

    // Create memory primitive based on dummy data
    context_.src_0_mem.reset(
        new memory(fwd_pd->src0_desc(), cpu_engine_, DummyData));
    context_.src_1_mem.reset(
        new memory(fwd_pd->src1_desc(), cpu_engine_, DummyData));
    context_.dst_mem.reset(
        new memory(fwd_pd->dst_desc(), cpu_engine_, DummyData));
    // Create eltwise primitive and add it to net
    context_.binary_prim.reset(new binary(*context_.fwd_pd));
    context_.fwd_primitives_args.push_back(
        {{DNNL_ARG_SRC_0, *context_.src_0_mem},
         {DNNL_ARG_SRC_1, *context_.src_1_mem},
         {DNNL_ARG_DST, *context_.dst_mem}});
    context_.fwd_primitives.push_back(*context_.binary_prim);
  }

  struct BinaryContext context_;

#ifdef DNNL_AARCH64_USE_ACL
  mutex primitive_execution_mu_;
#endif
};


template <typename T>
class MklBinaryPrimitiveFactory : public MklPrimitiveFactory<T> {
 public:
  static MklBinaryPrimitive<T>* Get(
      const MklBinaryParams<T>& fwdParams) {
    MklBinaryPrimitive<T>* binary = nullptr;

    // Get a eltwise fwd primitive from the cached pool
    binary = static_cast<MklBinaryPrimitive<T>*>(
        MklBinaryPrimitiveFactory<T>::GetInstance().GetBinary(
            fwdParams));
    if (binary == nullptr) {
      binary = new MklBinaryPrimitive<T>(fwdParams);
      MklBinaryPrimitiveFactory<T>::GetInstance().SetBinary(
          fwdParams, binary);
    }

    return binary;
  }

  static MklBinaryPrimitiveFactory& GetInstance() {
    static MklBinaryPrimitiveFactory instance_;
    return instance_;
  }

 private:
  MklBinaryPrimitiveFactory() {}
  ~MklBinaryPrimitiveFactory() {}

  static string CreateKey(const MklBinaryParams<T>& fwdParams) {
    string prefix = "binary_fwd_bwd";
    FactoryKeyCreator key_creator;
    key_creator.AddAsKey(prefix);
    key_creator.AddAsKey(fwdParams.src_0_dims);
    key_creator.AddAsKey(fwdParams.src_1_dims);
    key_creator.AddAsKey<int>(static_cast<int>(fwdParams.alg_kind));
    return key_creator.GetKey();
  }

  MklPrimitive* GetBinary(const MklBinaryParams<T>& fwdParams) {
    string key = CreateKey(fwdParams);
    return this->GetOp(key);
  }

  void SetBinary(const MklBinaryParams<T>& fwdParams,
                     MklPrimitive* op) {
    string key = CreateKey(fwdParams);
    this->SetOp(key, op);
  }
};


template <typename Device, typename T, algorithm alg_kind>
class MklBinaryOp : public OpKernel {
 public:
  ~MklBinaryOp() {}

  explicit MklBinaryOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    std::cout << "hebi-dbg: enter compute\n";
    try {
      const size_t src_0_index = 0;  // index of src input tensor
      const size_t src_1_index = 1;  // index of src input tensor
      const size_t dst_index = 0;  // index of dst output tensor
      const Tensor& src_0_tensor = MklGetInput(context, src_0_index);
      const Tensor& src_1_tensor = MklGetInput(context, src_1_index);
      MklDnnShape dnn_shape_src_0;
      MklDnnShape dnn_shape_src_1;
      GetMklShape(context, src_0_index, &dnn_shape_src_0);
      GetMklShape(context, src_1_index, &dnn_shape_src_1);
      if (src_0_tensor.dims() == 0 ||
          src_1_tensor.dims() == 0 ) {
        // Compute_Scalar(context);
        return;
      }
      MklDnnShape dnn_shape_dst;
      TensorShape tf_shape_dst;
      Tensor* dst_tensor = nullptr;
      // Nothing to compute, return.
      if (src_0_tensor.shape().num_elements() == 0) {
        dnn_shape_dst.SetMklTensor(false);
        tf_shape_dst = MklGetInput(context, src_0_index).shape();
        AllocateOutputSetMklShape(context, dst_index, &dst_tensor, tf_shape_dst,
                                  dnn_shape_dst);
        return;
      }
      // Set DNN primitive - src
      MklDnnData<T> src(&cpu_engine);
      memory::dims src_0_dims;
      memory::dims src_1_dims;
      memory::desc src_0_md({}, memory::data_type::undef,
                          memory::format_tag::undef);
      memory::desc src_1_md({}, memory::data_type::undef,
                          memory::format_tag::undef);
      if (dnn_shape_src_0.IsMklTensor()) {
        src_0_md = dnn_shape_src_0.GetMklLayout();
        src_0_dims = dnn_shape_src_0.GetSizesAsMklDnnDims();
      } else {
        src_0_dims = TFShapeToMklDnnDims(src_0_tensor.shape());
        auto src_0_strides = CalculateTFStrides(src_0_dims);
        // Create blocked memory descriptor
        src_0_md = MklDnnData<T>::CreateBlockedMemDesc(src_1_dims, src_0_strides);
      }

      if (dnn_shape_src_1.IsMklTensor()) {
        src_1_md = dnn_shape_src_1.GetMklLayout();
        src_1_dims = dnn_shape_src_1.GetSizesAsMklDnnDims();
      } else {
        src_1_dims = TFShapeToMklDnnDims(src_1_tensor.shape());
        auto src_1_strides = CalculateTFStrides(src_1_dims);
        // Create blocked memory descriptor
        src_1_md = MklDnnData<T>::CreateBlockedMemDesc(src_1_dims, src_1_strides);
      }

      // Try to get an eltwise forward primitive from caching pool
      MklBinaryParams<T> fwdParams(src_0_dims, src_0_md, src_1_dims, src_1_md, alg_kind);
      MklBinaryPrimitive<T>* binary_prim =
          MklBinaryPrimitiveFactory<T>::Get(fwdParams);
      auto bin_pd = binary_prim->GetBinaryPd();
      std::shared_ptr<stream> fwd_cpu_stream;
      MklDnnThreadPool eigen_tp(context);
      fwd_cpu_stream.reset(CreateStream(&eigen_tp, binary_prim->GetEngine()));
      
      // Check if src_0 needs to be reordered
      bool is_src_0_reordered = false;
      const T* src_0_data = src_0_tensor.flat<T>().data();
      if (src_0_md != bin_pd->src0_desc()) {
        src.SetUsrMem(src_0_md, &src_0_tensor);
        src.CheckReorderToOpMem(bin_pd->src0_desc(), cpu_engine,
                                context);
        src_0_data = const_cast<T*>(
            reinterpret_cast<T*>(src.GetOpMem().get_data_handle()));
        is_src_0_reordered = true;
      }

      // TODO:
      // // Check if src_1 needs to be reordered
      // bool is_src_1_reordered = false;
      const T* src_1_data = src_1_tensor.flat<T>().data();
      // if (src_1_md != bin_pd->src1_desc()) {
      //   src_1.SetUsrMem(src_1_md, &src_1_tensor);
      //   src_1.CheckReorderToOpMem(bin_pd->src1_desc(), cpu_engine,
      //                           context);
      //   src_1_data = const_cast<T*>(
      //       reinterpret_cast<T*>(src_1.GetOpMem().get_data_handle()));
      //   is_src_1_reordered = true;
      // }

      // If src 0 is reordered, then dst tensor would be in blocked layout.
      // So we propagate this blocked layout on the output. We follow same
      // logic when src is in blocked (MKL) layout to start of with also.
      if (is_src_0_reordered || dnn_shape_src_0.IsMklTensor()) {
        dnn_shape_dst.SetMklTensor(true);
        auto dst_pd = bin_pd->dst_desc();
        dnn_shape_dst.SetMklLayout(&dst_pd);
        dnn_shape_dst.SetElemType(MklDnnType<T>());
        if (dnn_shape_src_0.IsMklTensor()) {
          dnn_shape_dst.SetTfLayout(dnn_shape_src_0.GetDimension(),
                                    dnn_shape_src_0.GetSizesAsMklDnnDims(),
                                    dnn_shape_src_0.GetTfDataFormat());
        } else {
          dnn_shape_dst.SetTfLayout(src_0_tensor.dims(),
                                    TFShapeToMklDnnDims(src_0_tensor.shape()),
                                    MklTensorFormat::FORMAT_BLOCKED);
        }
        tf_shape_dst.AddDim(dst_pd.get_size() / sizeof(T));
      } else {
        // If src is not in blocked layout or it is not reordered, then dst is
        // in native layout.
        dnn_shape_dst.SetMklTensor(false);
        tf_shape_dst = src_0_tensor.shape();
      }


      if (is_src_0_reordered) {
        // If src is reordered, then src and dst would be in different layouts.
        AllocateOutputSetMklShape(context, dst_index, &dst_tensor, tf_shape_dst,
                                  dnn_shape_dst);
      } else {
        // forwarding input to output works only when layouts of src and
        // dst tensor remains same -- either both of them are in native layout
        // or in blocked (MKL) layout.
        OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                    {static_cast<const int>(src_0_index)},
                                    static_cast<const int>(dst_index),
                                    tf_shape_dst, &dst_tensor));
        AllocateOutputSetMklShape(context, dst_index, dnn_shape_dst);
      }
      T* dst_data = dst_tensor->flat<T>().data();

      // execute eltwise
      binary_prim->Execute(src_0_data, src_1_data, dst_data, fwd_cpu_stream);
    } catch (dnnl::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          context,
          errors::Aborted("Operation received an exception:", error_msg));
    }
  }

 private:
  engine cpu_engine = engine(engine::kind::cpu, 0);
  // TODO: to be removed.
  // std::shared_ptr<BinaryPd> relu_fwd_pd;

};


template <typename Device, typename T>
class MklMulOp
    : public MklBinaryOp<Device, T, dnnl::algorithm::binary_mul> {
 public:
  ~MklMulOp() {}

  explicit MklMulOp(OpKernelConstruction* context)
      : MklBinaryOp<Device, T, dnnl::algorithm::binary_mul>(context) {}
};


// register dnn kernels for supported operations and supported types
#define REGISTER_MUL_MKL_SUPPORTED_KERNELS_TYPES(type)        \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklOneDNNMul")                                         \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                          \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel),  \
      MklMulOp<CPUDevice, type>);                             \
 

// TF_CALL_ALL_TYPES(REGISTER_MUL_MKL_SUPPORTED_KERNELS_TYPES);

TF_CALL_float(REGISTER_MUL_MKL_SUPPORTED_KERNELS_TYPES);
TF_CALL_bfloat16(REGISTER_MUL_MKL_SUPPORTED_KERNELS_TYPES);

}  // namespace tensorflow

// #endif  // INTEL_MKL
