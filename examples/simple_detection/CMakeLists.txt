find_package(nlohmann_json REQUIRED)
find_package(spdlog REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(simple_detection
    ${PROJECT_SOURCE_DIR}/3rd_party/log_mgr/log_mgr.cpp
    main.cpp
)

# 子项目构建
if(TARGET ai_stream_core)
    target_link_libraries(simple_detection
        PRIVATE
            nlohmann_json::nlohmann_json
            spdlog::spdlog
            ${OpenCV_LIBS}
            -Wl,--no-as-needed
            ai_stream_nodes
            ai_stream_core
            -Wl,--as-needed
    )
    # alert_rules 为可选对象库（WITH_ALERT=OFF 时不存在）
    if(TARGET alert_rules)
        target_link_libraries(simple_detection PRIVATE $<TARGET_OBJECTS:alert_rules>)
    endif()
else()
    # 独立项目
    find_package(ai_stream_pipeline REQUIRED)
    target_link_libraries(simple_detection
        PRIVATE
            nlohmann_json::nlohmann_json
            spdlog::spdlog
            ai_stream_pipeline::nodes
            ai_stream_pipeline::core
    )
endif()

target_include_directories(simple_detection
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}
)