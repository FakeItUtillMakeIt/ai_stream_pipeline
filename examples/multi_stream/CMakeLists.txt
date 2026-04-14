

find_package(nlohmann_json REQUIRED)
find_package(spdlog REQUIRED)

add_executable(multi_stream main.cpp)

if(TARGET ai_stream_core)
    target_link_libraries(multi_stream
    PRIVATE
        ai_stream_core
        ai_stream_nodes
        nlohmann_json::nlohmann_json
        spdlog::spdlog
    )
    
else()
    target_link_libraries(multi_stream
    PRIVATE
        ai_stream_pipeline::core
        ai_stream_pipeline::nodes
        nlohmann_json::nlohmann_json
        spdlog::spdlog
    )
endif()

target_include_directories(multi_stream
    PUBLIC
        ${PROJECT_SOURCE_DIR})
