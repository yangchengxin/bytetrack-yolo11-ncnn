add_executable(yolo11 main.cpp)
target_link_libraries(yolo11 PRIVATE
        opencv_java4
        mediandk
        z
        jnigraphics
        log
        c++_shared)
target_include_directories(yolo11 PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/mylib
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/include/ncnn/${ANDROID_ABI}/include
) # 不在第一个cmakelists中使用全局include_directories() 在独立的模块中显示的使用target_include_directories