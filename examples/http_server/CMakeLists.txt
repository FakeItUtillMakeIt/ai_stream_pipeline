find_package(nlohmann_json REQUIRED)
find_package(spdlog REQUIRED)

add_executable(ai_stream_server 
    ${PROJECT_SOURCE_DIR}/3rd_party/log_mgr/log_mgr.cpp
    ai_stream_server.cpp
    ${PROJECT_SOURCE_DIR}/src/http/api_server.cpp
)

target_link_libraries(ai_stream_server
    PRIVATE
    httplib::httplib
)

# 子项目构建
if(TARGET ai_stream_core)
    if(WITH_CUDA)
        target_link_libraries(ai_stream_server
            PRIVATE
                nlohmann_json::nlohmann_json
                spdlog::spdlog
                -Wl,--no-as-needed
                ai_stream_nodes
                ai_stream_core
                $<TARGET_OBJECTS:alert_rules>
                -Wl,--as-needed
        )
    else()
        target_link_libraries(ai_stream_server
            PRIVATE
                nlohmann_json::nlohmann_json
                spdlog::spdlog
                -Wl,--no-as-needed
                ai_stream_nodes
                ai_stream_core
                $<TARGET_OBJECTS:alert_rules>
                -Wl,--as-needed
        )
    endif()
else()
    # 独立项目
    find_package(ai_stream_pipeline REQUIRED)
    target_link_libraries(ai_stream_server
        PRIVATE
            nlohmann_json::nlohmann_json
            spdlog::spdlog
            ai_stream_pipeline::nodes
            ai_stream_pipeline::core
    )
endif()

target_include_directories(ai_stream_server
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}
)