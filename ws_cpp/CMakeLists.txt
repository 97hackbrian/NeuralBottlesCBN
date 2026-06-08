cmake_minimum_required(VERSION 3.16)

project(NeuralBottlesCBN LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)

# Find GLFW and OpenGL
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)

# Fetch Dear ImGui
include(FetchContent)
FetchContent_Declare(
    imgui
    URL https://github.com/ocornut/imgui/archive/refs/tags/v1.90.4.tar.gz
)
FetchContent_MakeAvailable(imgui)

# Gather ImGui source files
set(IMGUI_SOURCES
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)

add_executable(cbn_camera_test
	test/camera_test.cpp
)
target_link_libraries(cbn_camera_test PRIVATE ${OpenCV_LIBS})
target_include_directories(cbn_camera_test PRIVATE ${OpenCV_INCLUDE_DIRS})

# Test de Inferencia (OpenVINO + OpenCV)
add_executable(cbn_test_inference
    test/test_inference.cpp
)
target_include_directories(cbn_test_inference PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(cbn_test_inference PRIVATE ${OpenCV_LIBS} openvino::runtime)

# Main executable (Inference Node / Laboratorio)
add_executable(laboratorio_cbn 
    src/laboratorio_cbn.cpp
    src/pipeline.cpp
    ${IMGUI_SOURCES}
)

target_include_directories(laboratorio_cbn PRIVATE 
    include
    ${OpenCV_INCLUDE_DIRS}
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)

target_link_libraries(laboratorio_cbn PRIVATE 
    ${OpenCV_LIBS}
    openvino::runtime
    glfw
    OpenGL::GL
)


# Nuevo ejecutable principal (Inferencia final con lógica)
add_executable(main 
    src/main.cpp
    src/cbn_detector_inference.cpp
    src/pipeline.cpp
)
target_include_directories(main PRIVATE 
    include
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(main PRIVATE 
    ${OpenCV_LIBS}
    openvino::runtime
)
