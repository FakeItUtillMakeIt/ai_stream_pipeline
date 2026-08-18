# tests/unit/CMakeLists.txt
# 核心框架单元测试（不依赖 GPU/CUDA/TensorRT，仅链接 ai_stream_core）

add_executable(test_core
    core/test_bounded_queue.cpp
    core/test_queued_node.cpp
    core/test_pipeline.cpp
    core/test_packet.cpp
)

target_include_directories(test_core PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(test_core PRIVATE
    ai_stream_core
    GTest::gtest
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(test_core)
