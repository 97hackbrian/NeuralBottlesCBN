cmake_minimum_required(VERSION 3.16)

project(NeuralBottlesCBN LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)

add_executable(cbn_camera_test
	test/camera_test.cpp
)

target_link_libraries(cbn_camera_test PRIVATE ${OpenCV_LIBS})
target_include_directories(cbn_camera_test PRIVATE ${OpenCV_INCLUDE_DIRS})

