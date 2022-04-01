// Copyright 2021 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "openvino/openvino.hpp"
#include "openvino/openvino.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/add.hpp"
#include "openvino/runtime/compiled_model.hpp"

namespace mediapipe {
namespace api2 {

// A Calculator that returns 0 if INPUT is 0, and 1 otherwise.
class OVAddCalculator : public Node {
 public:
  static constexpr Input<ov::Tensor>::Optional kInA{"INPUTA"};
  static constexpr Input<ov::Tensor>::Optional kInB{"INPUTB"};
  static constexpr Output<ov::Tensor>::Optional kOut{"OUTPUT"};

  MEDIAPIPE_NODE_CONTRACT(kInA, kInB, kOut);

  absl::Status UpdateContract(CalculatorContract* cc) {
    RET_CHECK(kOut(cc).IsConnected())
        << "At least  output stream is expected.";
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) final {
    ov::Core core;
    auto A = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape({2, 2}));
    auto B = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape({2, 2}));
    auto arg = std::make_shared<ov::op::v1::Add>(A, B);
    auto f = std::make_shared<ov::Model>(arg, ov::ParameterVector{A, B});
    compiledModel = core.compile_model(f, "CPU");
    inferRequest = compiledModel.create_infer_request();
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) final {
    inferRequest.set_input_tensor(0, *kInA(cc));
    inferRequest.set_input_tensor(1, *kInB(cc));

    inferRequest.infer();
    kOut(cc).Send(std::make_unique<ov::Tensor>(inferRequest.get_output_tensor()));
    return absl::OkStatus();
  }

private:
  ov::CompiledModel compiledModel;
  ov::InferRequest inferRequest;
};

MEDIAPIPE_REGISTER_NODE(OVAddCalculator);

}  // namespace api2
}  // namespace mediapipe
