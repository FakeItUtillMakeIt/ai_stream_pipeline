# src/nodes/gate/CMakeLists.txt

set(GATE_SOURCES
    vlm_gate_node.cpp
    report_sink_node.cpp
)

add_library(gate_node OBJECT ${GATE_SOURCES})
set_target_properties(gate_node PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(gate_node
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OpenCV_INCLUDE_DIRS}
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/nodes
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include/ai_stream/nodes
)

target_link_libraries(gate_node
    PUBLIC
        nlohmann_json::nlohmann_json
        ai_stream_core
    PRIVATE
        opencv_core
        opencv_imgproc
        opencv_imgcodecs
        CURL::libcurl
)

target_compile_features(gate_node PRIVATE cxx_std_17)
