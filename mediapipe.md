# Overview

MediaPipe is a Framework for building machine learning pipelines for processing time-series data like video, audio, etc. This cross-platform Framework works in Desktop/Server, Android, iOS, and embedded devices like Raspberry Pi and Jetson Nano. 

<figure>
  <img src="https://learnopencv.com/wp-content/uploads/2022/02/mediapipe-series-mediapipe-toolkit.png" width="500" />
  <figcaption>mediapipe toolkit</figcaption>
</figure>


The Framework is written in C++, Java, and Obj-C, which consists of the following APIs.

Calculator API (C++).
Graph construction API (Protobuf).
Graph Execution API (C++, Java, Obj-C).


## Graph
The MediaPipe perception pipeline is called a Graph. Let us take the example of the first solution, Hands. We feed a stream of images as input which comes out with hand landmarks rendered on the images. 

<figure>
  <img src="https://learnopencv.com/wp-content/uploads/2022/02/mediapipe-series-hand-solutions-graph-2.jpg" width="800" />
  <figcaption>mediapipe graph</figcaption>
</figure>

[graph visualizer tool](https://viz.mediapipe.dev/)

## Calcualtor
Calculator Types in MediaPipe
All the calculators shown above are built-in into MediaPipe. We can group them into four categories.

**Pre-processing:** calculators are family of image and media processing calculators. The ImageTransform and ImageToTensors in the graph above fall in this category.
**Inference calculators:** allow native integration with Tensorflow and Tensorflow Lite for ML inference.
**Post-processing calculators:** perform ML post-processing tasks such as detection, segmentation, and classification. TensorToLandmark is a post-processing calculator.
**Utility calculators:** are a family of calculators performing final tasks such as image annotation.
The calculator APIs allow you to write your custom calculator. We will cover it in a future post.

## Stream
Every stream carries a sequence of Packets that have ascending time stamps. Packets can be any type. 

# Set up
 
update gcc verion
using c++17, c++11 buiding error.
Using bazel to build.
bazel build/run --define MEDIAPIPE_DISABLE_GPU=1  mediapipe/examples/desktop/hello_world:hello_world


# Debugging with bazel 

**-c dbg** or   **--copt -g**
gdb:/home/luwei/.cache/bazel/_bazel_luwei/6df5ad01d33ec7aa01c6afc1b46c433b/execroot/mediapipe/bazel-out/k8-fastbuild/testlogs/mediapipe/calculators/core/ov_add_calculator_test

# Calcualtor API2
Several calculators in
[`calculators/core`](https://github.com/google/mediapipe/tree/master/mediapipe/calculators/core) and
[`calculators/tensor`](https://github.com/google/mediapipe/tree/master/mediapipe/calculators/tensor)
have been updated to use this API. Reference them for more examples.



# Simple OV add calculator
## Openvino binary as third party dependency.
## Use ov::tensor as packet type
## Open() API to set up and compile network.
## Process() only added 2 inputs and send result as output
## No graph(.pbtxt) is build in test and no extension option(.proto) is supported in calculator



# OV supported in mediapipe
## OV as thirdparty dependency(Maybe source code c++17 building) into mediapipe. 
## Not just a caululator. some common calculators and selector only support tensor flow.
## All the supported models , preprocess, postprocess. If can't reused, need to develop;
## 
https://learnopencv.com/introduction-to-mediapipe/