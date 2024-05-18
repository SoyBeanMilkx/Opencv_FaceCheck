cmake_minimum_required(VERSION 3.28)
project(OpenCV_Face)

set(CMAKE_CXX_STANDARD 17)
set(OpenCV_DIR "D:\\OpenCv\\opencv\\mingw-build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(OpenCV_Face main.cpp)

link_directories(${OpenCV_LIBRARY_DIRS})
target_link_libraries(OpenCV_Face ${OpenCV_LIBS})

