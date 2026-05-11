file(GLOB ALERT_RULE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB ALERT_RULE_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

message(STATUS "=== Alert Rule Sources ===")
foreach(src ${ALERT_RULE_SOURCES})
    message(STATUS "Source: ${src}")
endforeach()

add_library(alert_rules OBJECT ${ALERT_RULE_SOURCES} ${ALERT_RULE_HEADERS})

target_compile_options(alert_rules PRIVATE -fPIC)
target_include_directories(alert_rules 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        
    PRIVATE 
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(alert_rules
    PUBLIC
        nlohmann_json::nlohmann_json
    PRIVATE
        opencv_core
        opencv_imgproc
)

target_compile_features(alert_rules PRIVATE cxx_std_17)