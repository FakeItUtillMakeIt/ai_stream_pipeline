# src/nodes/evidence/CMakeLists.txt

file(GLOB EVIDENCE_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

set(EVIDENCE_SOURCES
    frame_buffer.cpp
    video_recorder.cpp
    video_rollover.cpp
    evidence_node.cpp
)

if(WITH_FTP)
    list(APPEND EVIDENCE_SOURCES ftp_uploader.cpp)
endif()

add_library(evidence_node OBJECT 
    ${EVIDENCE_SOURCES}
    ${EVIDENCE_HEADERS}
)

target_compile_options(evidence_node PRIVATE -fPIC)

target_include_directories(evidence_node
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OpenCV_INCLUDE_DIRS}
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/nodes
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/nodes/sink
        ${CMAKE_SOURCE_DIR}/include/ai_stream/nodes
)

target_link_libraries(evidence_node
    PUBLIC
        nlohmann_json::nlohmann_json
        ai_stream_core
    PRIVATE
        ${FFMPEG_LIBRARIES}
        opencv_core
        opencv_imgproc
        opencv_videoio
)

if(WITH_FTP)
    target_compile_definitions(evidence_node PRIVATE WITH_FTP)
    target_link_libraries(evidence_node PRIVATE CURL::libcurl)
endif()

target_compile_features(evidence_node PRIVATE cxx_std_17)
