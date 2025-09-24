cmake_minimum_required(VERSION 3.26)
project(yolov11_new)

set(CMAKE_CXX_STANDARD 17)

# 可以使用中文正常显示
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/source-charset:utf-8>")

# opencv
set(OpenCV_DIR D:\\opencv\\build)
include_directories(${OpenCV_DIR}/include)
link_directories(${OpenCV_DIR}/x64/vc16/lib)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    link_libraries(opencv_world480d)
else()
    link_libraries(opencv_world480)
endif()

# ncnn
set(ncnn_DIR D:\\company_Tenda\\25.zc\\install)
include_directories(${ncnn_DIR}/include)
link_directories(${ncnn_DIR}/lib)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    link_libraries(ncnnd)
else()
    link_libraries(ncnn)
endif()

# eigen3
include_directories(D:\\company_Tenda\\env\\eigen-3.3.8\\eigen-3.3.8)

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/test)

add_definitions(-DNOMINMAX)

# 根据src目录中每一个cpp文件生成一个库
file(GLOB SRC_FILES
        ${PROJECT_SOURCE_DIR}/src/*.cpp
        ${PROJECT_SOURCE_DIR}/src/*.h
        ${PROJECT_SOURCE_DIR}/src/byte_track/*.cpp
        ${PROJECT_SOURCE_DIR}/src/byte_track/*.h
)

# 根据test目录中每一个cpp文件生成一个可执行文件
file(GLOB TEST_FILES ${PROJECT_SOURCE_DIR}/test/*.cpp)
foreach (TEST_FILE ${TEST_FILES})
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    message(STATUS "  Add test: ${TEST_NAME}")
    add_executable(${TEST_NAME} ${TEST_FILE} ${SRC_FILES})
endforeach ()
