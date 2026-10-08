# tests/integration/CMakeLists.txt
#
# 需要 GPU + TensorRT + 已生成的 .engine 与模型文件，因此**不注册到 ctest**
# （普通开发者跑 ctest 时这些文件往往不存在，注册进去只会变成常年失败）。
# 构建与运行方式：
#
#   cmake --build build --target test_relation_cpp
#   ./build/tests/integration/test_relation_cpp \
#       models/relation/relateanything.engine \
#       models/relation/predicate_bank.json \
#       /path/to/test5.jpg \
#       'beside,in front of,behind,on'
#
# 真正跑之前先建 engine：
#   trtexec --onnx=models/relation/relateanything.onnx \
#           --saveEngine=models/relation/relateanything.engine --fp16 \
#           --minShapes=image:1x3x448x448,boxes:1x2x4,box_counts:1,W:243x512,alpha:243 \
#           --optShapes=image:1x3x448x448,boxes:1x32x4,box_counts:1,W:243x512,alpha:243 \
#           --maxShapes=image:1x3x448x448,boxes:1x32x4,box_counts:1,W:243x512,alpha:243

if(WITH_CUDA AND TensorRT_FOUND)
    add_executable(test_relation_cpp
        test_relation_cpp.cpp
    )

    target_include_directories(test_relation_cpp PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/src/hal/infer/common
    )

    target_link_libraries(test_relation_cpp PRIVATE
        ai_stream_hal
        ai_stream_core
        ${OpenCV_LIBS}
    )
else()
    message(STATUS
        "test_relation_cpp skipped: needs WITH_CUDA=ON and TensorRT")
endif()
