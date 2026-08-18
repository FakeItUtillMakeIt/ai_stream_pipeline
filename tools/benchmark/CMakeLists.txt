# tools/benchmark/CMakeLists.txt
add_executable(bench bench.cpp)

target_include_directories(bench PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
)

# 链接节点库以注册全部节点类型。
# bench 不直接引用节点符号，必须 --no-as-needed 强制产生 DT_NEEDED，
# 否则动态加载器不会加载节点库，静态工厂注册不会执行
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_link_libraries(bench PRIVATE
        ai_stream_core
        -Wl,--no-as-needed
        ai_stream_nodes
        -Wl,--as-needed
    )
else()
    target_link_libraries(bench PRIVATE ai_stream_core ai_stream_nodes)
endif()

install(TARGETS bench
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
