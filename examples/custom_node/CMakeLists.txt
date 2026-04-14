
find_package(OpenCV REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(spdlog REQUIRED)

# 将自定义节点编译为静态库（可选），或直接加入可执行文件
add_library(my_custom_node STATIC my_custom_node.cpp)
target_include_directories(my_custom_node PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src/nodes    
    ${PROJECT_SOURCE_DIR}
)

add_executable(custom_demo main.cpp)

if(TARGET ai_stream_core)
    target_link_libraries(my_custom_node
        PUBLIC
            ai_stream_core
            opencv_core opencv_imgproc
        PRIVATE
            spdlog::spdlog
    )
    
    target_link_libraries(custom_demo
        PRIVATE
            ai_stream_core
            ai_stream_nodes   # 为了使用 mock_source / mock_sink
            my_custom_node
            nlohmann_json::nlohmann_json
            spdlog::spdlog
    )
else()
    target_link_libraries(my_custom_node
        PUBLIC
            ai_stream_pipeline::core
            opencv_core opencv_imgproc
        PRIVATE
            spdlog::spdlog
    )
    
    target_link_libraries(custom_demo
        PRIVATE
            ai_stream_pipeline::core
            ai_stream_pipeline::nodes   # 为了使用 mock_source / mock_sink
            my_custom_node
            nlohmann_json::nlohmann_json
            spdlog::spdlog
    )
endif()


