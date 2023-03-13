cmake_minimum_required(VERSION 3.15)

project(svg_renderer)
set(CMAKE_CXX_STANDARD 17)

add_executable(svg_renderer main.cpp)
set_property(TARGET svg_renderer PROPERTY
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# # bx
add_library(
    bx
    "${CMAKE_CURRENT_SOURCE_DIR}/bx/src/bx.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/bx/src/string.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/bx/src/math.cpp"
)
target_include_directories(bx PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/bx/include")
target_compile_definitions(bx PUBLIC BX_CONFIG_DEBUG=1)

if(MSVC)
    target_compile_options(bx PUBLIC "/Zc:__cplusplus")
endif()

# # #

# # # simple-svg
add_library(
    simple_svg
    "${CMAKE_CURRENT_SOURCE_DIR}/simple-svg/src/ssvg_builder.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/simple-svg/src/ssvg_parser.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/simple-svg/src/ssvg_writer.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/simple-svg/src/ssvg.cpp"
)
target_include_directories(simple_svg PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/simple-svg/include")
target_link_libraries(simple_svg PUBLIC bx)

# #

# ## simpleBMP
add_library(
    SimpleBMP
    "${CMAKE_CURRENT_SOURCE_DIR}/SimpleBMP/simplebmp/simplebmp.cpp"
)
target_include_directories(SimpleBMP PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/SimpleBMP")
set_property(TARGET SimpleBMP PROPERTY
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# ###

# ##
set(MSDFGEN_CORE_ONLY ON)
set(MSDFGEN_BUILD_STANDALONE OFF)
set(MSDFGEN_USE_VCPKG OFF)
set(MSDFGEN_USE_SKIA OFF)
add_subdirectory(msdfgen)

target_link_libraries(svg_renderer PUBLIC msdfgen-core)
target_link_libraries(svg_renderer PUBLIC SimpleBMP)

target_link_libraries(svg_renderer PUBLIC simple_svg)