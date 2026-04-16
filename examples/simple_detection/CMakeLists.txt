# examples/simple_detection/CMakeLists.txt

find_package(nlohmann_json REQUIRED)
find_package(spdlog REQUIRED)

if(WITH_CUDA)
    find_package(CUDA REQUIRED)
    find_package(CUDAToolkit REQUIRED)
endif()

add_executable(simple_detection 
    ${PROJECT_SOURCE_DIR}/3rd_party/log_mgr/log_mgr.cpp
    main.cpp)

# 判断是作为子项目构建还是独立构建
if(TARGET ai_stream_core)
    message("target ai_stream_core exist")
    if(WITH_CUDA)
        target_link_libraries(simple_detection
            PRIVATE
                
                nlohmann_json::nlohmann_json
                spdlog::spdlog
                -Wl,--no-as-needed
                ai_stream_nodes
                ai_stream_core
                -Wl,--as-needed
                CUDA::cudart
                ${CUDA_TOOLKIT_TARGET_DIR}/targets/x86_64-linux/lib/libcudadevrt.a
                
        )
    else()
        target_link_libraries(simple_detection
            PRIVATE
                
                nlohmann_json::nlohmann_json
                spdlog::spdlog
                -Wl,--no-as-needed
                ai_stream_nodes
                ai_stream_core
                -Wl,--as-needed
        )
    endif()    
    target_include_directories(simple_detection
        PRIVATE
            ${CMAKE_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}
            ${CUDA_TOOLKIT_TARGET_DIR}/targets/x86_64-linux/include
    )
else()
    # 作为独立项目构建，需要 find_package
    find_package(ai_stream_pipeline REQUIRED)
    target_link_libraries(simple_detection
        PRIVATE
            nlohmann_json::nlohmann_json
            spdlog::spdlog
            ai_stream_pipeline::core
            ai_stream_pipeline::nodes
    )

        target_include_directories(simple_detection
        PRIVATE
            ${CMAKE_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}
            ${CUDA_TOOLKIT_TARGET_DIR}/targets/x86_64-linux/include
    )
endif()


