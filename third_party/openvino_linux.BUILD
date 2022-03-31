# Description:
#   OpenCV libraries for video/image processing on Linux

licenses(["notice"])  # BSD license

exports_files(["LICENSE"])

cc_library(
    name = "openvino",
    hdrs = glob(["include/openvino/*.h*",
                "include/openvino/**/*.h*",
                "include/ie/*.h*",
                "include/ie/**/*.h*",
                "include/ngraph/*.h*",
                "include/ngraph/**/*.h*",
                ]),
    includes = [
        # For OpenCV 4.x
        "include/",
        "include/ie/",
        "include/ngraph/",
        #"include/x86_64-linux-gnu/opencv4/",
        #"include/opencv4/",
    ],
    copts = [
        # For OpenCV 4.x
        "-I:/home/luwei/gittree/debug_install/runtime/include/",
        "-I:/home/luwei/gittree/debug_install/runtime/include/ie/",
        #"include/x86_64-linux-gnu/opencv4/",
        #"include/opencv4/",
    ],
    linkopts = [
        "-l:libopenvino.so",
        "-Wl,-rpath-link=/home/luwei/gittree/debug_install/runtime/3rdparty/tbb/lib/",
        "-L /home/luwei/gittree/debug_install/runtime/lib/intel64"
    ],
    visibility = ["//visibility:public"],

)