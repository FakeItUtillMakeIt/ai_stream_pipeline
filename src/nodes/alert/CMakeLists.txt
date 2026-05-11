
file(GLOB ALERT_NODE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB ALERT_NODE_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

add_library(alert_node OBJECT ${ALERT_NODE_SOURCES} ${ALERT_NODE_HEADERS})

target_compile_options(alert_node PRIVATE -fPIC)
target_include_directories(alert_node 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OpenCV_INCLUDE_DIRS}
    PRIVATE 
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/nodes
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(alert_node
    PUBLIC
        nlohmann_json::nlohmann_json
)   

target_compile_features(alert_node PRIVATE cxx_std_17)