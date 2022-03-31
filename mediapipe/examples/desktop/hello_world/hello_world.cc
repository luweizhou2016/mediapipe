// Copyright 2019 The MediaPipe Authors.
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
//
// A simple example to print out "Hello World!" from a MediaPipe graph.

#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"

#include "openvino/openvino.hpp"
#include "openvino/openvino.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/add.hpp"

namespace mediapipe {

absl::Status PrintHelloWorld() {
  // Configures a simple graph, which concatenates 2 PassThroughCalculators.
  CalculatorGraphConfig config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "in"
        output_stream: "out"
        node {
          calculator: "PassThroughCalculator"
          input_stream: "in"
          output_stream: "out1"
        }
        node {
          calculator: "PassThroughCalculator"
          input_stream: "out1"
          output_stream: "out"
        }
      )pb");

  CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config));
  ASSIGN_OR_RETURN(OutputStreamPoller poller,
                   graph.AddOutputStreamPoller("out"));
  MP_RETURN_IF_ERROR(graph.StartRun({}));
  // Give 10 input packets that contains the same string "Hello World!".
  for (int i = 0; i < 10; ++i) {
    MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
        "in", MakePacket<std::string>("Hello World!").At(Timestamp(i))));
  }
  // Close the input stream "in".
  MP_RETURN_IF_ERROR(graph.CloseInputStream("in"));
  mediapipe::Packet packet;
  // Get the output packets string.
  while (poller.Next(&packet)) {
    LOG(INFO) << packet.Get<std::string>();
  }
  return graph.WaitUntilDone();
}
}  // namespace mediapipe

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  CHECK(mediapipe::PrintHelloWorld().ok());

    ov::Core core;
    ov::Shape shape{2, 2};
    size_t eleNum = 1;
    for (auto value : shape)
        eleNum *= value;
    uint8_t *intputData = new uint8_t [eleNum];
    for (auto i = 0; i < eleNum; i++) {
        *(intputData+i) = i;
        std::cout << static_cast<int32_t>(*(intputData + i)) << ",";
    }
    std::cout << std::endl;

    auto A = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, shape);
    auto B = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, shape);
    auto arg = std::make_shared<ov::op::v1::Add>(A, B);
    auto f = std::make_shared<ov::Model>(arg, ov::ParameterVector{A, B});
    auto compiledModel = core.compile_model(f, "CPU");
    ov::InferRequest infer_request = compiledModel.create_infer_request();
    const ov::Tensor input_tensor{ov::element::u8, shape, intputData};
    infer_request.set_input_tensor(0, input_tensor);
    infer_request.set_input_tensor(1, input_tensor);

    infer_request.infer();
    const ov::Tensor& output_tensor = infer_request.get_output_tensor();
    uint8_t *outputData = static_cast<uint8_t*>(output_tensor.data());


    for (auto i = 0; i < eleNum; i++)
        std::cout << static_cast<int32_t>(*(outputData + i)) << ",";
    std::cout << std::endl;

    delete intputData;

  return 0;
}
