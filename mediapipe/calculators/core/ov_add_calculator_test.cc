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

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/framework/tool/validate_type.h"
#include "openvino/openvino.hpp"

namespace mediapipe {

class OVAddCalculatorTest : public ::testing::Test {
 protected:
  OVAddCalculatorTest()
      : runner_(
            R"pb(
              calculator: "OVAddCalculator"
              input_stream: "INPUTA:input_a"
              input_stream: "INPUTB:input_b"
              output_stream: "OUTPUT:output"
            )pb") {}

  void SetInputs(const std::vector<ov::Tensor>& inputs) {
    int timestamp = 0;
    for (const auto input : inputs) {
      runner_.MutableInputs()
          ->Get("INPUTA", 0)
          .packets.push_back(MakePacket<ov::Tensor>(input).At(Timestamp(timestamp)));
      runner_.MutableInputs()
          ->Get("INPUTB", 0)
          .packets.push_back(MakePacket<ov::Tensor>(input).At(Timestamp(timestamp++)));
    }
  }

  std::vector<ov::Tensor> GetOutput() {
    std::vector<ov::Tensor> result;
    for (const auto output : runner_.Outputs().Get("OUTPUT", 0).packets) {
      result.push_back(output.Get<ov::Tensor>());
    }
    return result;
  }
  CalculatorRunner runner_;
};

TEST_F(OVAddCalculatorTest, ResultIsOkay) {
    ov::Shape shape{2, 2};
    size_t eleNum = 1;
    for (auto value : shape)
        eleNum *= value;
    uint8_t *intputData = new uint8_t [eleNum];
    for (auto i = 0; i < eleNum; i++) {
        *(intputData+i) = i;
    }
  ov::Tensor inputTensor{ov::element::u8, shape, intputData};
  std::vector<ov::Tensor> tensors(10, inputTensor);
  SetInputs(tensors);

  MP_ASSERT_OK(runner_.Run());
  auto outputTensorVec =  GetOutput();
  EXPECT_THAT(outputTensorVec.size(), 10);


    for (auto outputTensor : outputTensorVec){
        uint8_t *outputData = static_cast<uint8_t*>(outputTensor.data());
        for (auto i = 0; i < eleNum; i++) {
              EXPECT_THAT(*(outputData+i), i+i);
        }
    }
  delete[] intputData;

}

}  // namespace mediapipe
