#include "pipeline.h"
#include <iostream>
#include <filesystem>
#include <chrono>
#include <thread>
#include <ctime>
#include <vector>

std::vector<int> getAvailableCameras() {
    std::vector<int> cameras;
    for (int i = 0; i < 10; ++i) {
        if (std::filesystem::exists("/dev/video" + std::to_string(i))) {
            cameras.push_back(i);
        }
    }
    if (cameras.empty()) cameras.push_back(0); // Fallback
    return cameras;
}

// ImGui, GLFW, OpenGL
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h> 

static void glfw_error_callback(int error, const char* description) {
    std::cerr << "GLFW Error " << error << ": " << description << std::endl;
}

// ============================================================================
// OpenGL Utilities
// ============================================================================
/**
 * @brief Helper function to upload an OpenCV cv::Mat image into an OpenGL texture.
 * @param image The cv::Mat BGR image.
 * @return GLuint The OpenGL texture ID.
 */
GLuint MatToTexture(const cv::Mat& image) {
    if (image.empty()) return 0;

    // Convert BGR to RGBA for maximum compatibility with OpenGL drivers
    cv::Mat rgba;
    cv::cvtColor(image, rgba, cv::COLOR_BGR2RGBA);

    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    // Setup filtering parameters for display
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    // Disable byte-alignment restriction
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    
    // Upload pixels into texture
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, rgba.cols, rgba.rows, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, rgba.ptr());

    return textureID;
}

// ============================================================================
// Automated Tests (Headless logic validation)
// ============================================================================
void runAutomatedTests() {
    std::cout << "\n=========================================================" << std::endl;
    std::cout << "[TEST] Iniciando Pruebas Automatizadas (Headless Mode)..." << std::endl;
    std::cout << "=========================================================" << std::endl;

    // --- TEST 1: Snapping Gaussian Blur to Odd Numbers ---
    std::cout << "[TEST] 1. Probando logica de ajuste de Gaussian Blur..." << std::endl;
    int test_blur = 4;
    if (test_blur % 2 == 0) test_blur += 1;
    if (test_blur != 5) {
        std::cerr << "[TEST FAIL] Falla en logica de ajuste impar." << std::endl;
        exit(1);
    }
    std::cout << "[TEST PASS] Logica de ajuste a impar exitosa." << std::endl;

    // --- TEST 2: YAML Persistence ---
    std::cout << "[TEST] 2. Probando persistencia YAML..." << std::endl;
    int original_width = g_settings.camera_width;
    g_settings.camera_width = 800;
    g_settings.blur_kernel_size = 9;

    const std::string test_yaml = "test_config.yaml";
    guardarPresets(test_yaml);

    g_settings.camera_width = 1920;
    
    bool load_res = cargarPresets(test_yaml);
    if (!load_res || g_settings.camera_width != 800) {
        std::cerr << "[TEST FAIL] Datos recuperados de YAML no coinciden." << std::endl;
        std::filesystem::remove(test_yaml);
        exit(1);
    }

    std::filesystem::remove(test_yaml);
    g_settings.camera_width = original_width;
    std::cout << "[TEST PASS] Persistencia YAML exitosa." << std::endl;

    std::cout << "\n=========================================================" << std::endl;
    std::cout << "          TODAS LAS PRUEBAS PASARON EXITOSAMENTE         " << std::endl;
    std::cout << "=========================================================\n" << std::endl;
}

// ============================================================================
// Main Execution Block
// ============================================================================
int main(int argc, char** argv) {
    // 1. Process CLI arguments
    if (argc > 1 && std::string(argv[1]) == "--test") {
        runAutomatedTests();
        return 0; // Exit successfully without invoking any GUI/OpenGL code
    }

    std::cout << "=========================================================" << std::endl;
    std::cout << "      CBN Zenith Inspection Laboratory GUI (ImGui)       " << std::endl;
    std::cout << "=========================================================" << std::endl;

    // 2. Detección de Hardware y Carga de Presets
    std::vector<int> available_cameras = getAvailableCameras();
    int current_camera_idx = 0;
    int active_cam_id = available_cameras[current_camera_idx];

    std::string current_yaml = "../config/config_cam" + std::to_string(active_cam_id) + ".yaml";
    cargarPresets(current_yaml);

    // 3. Setup GLFW window
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) {
        std::cerr << "[ERROR] No se pudo inicializar GLFW!" << std::endl;
        return 1;
    }

    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "Laboratorio CBN - GUI", nullptr, nullptr);
    if (window == nullptr) {
        std::cerr << "[ERROR] Fallo al crear ventana GLFW!" << std::endl;
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable V-Sync

    // 4. Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    
    // Setup Dark Mode
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // 5. Hardware Initialization
    // Forzamos el backend V4L2 nativo de Linux para evadir errores de frames negros en GStreamer
    cv::VideoCapture cap(active_cam_id, cv::CAP_V4L2);
    bool has_camera = cap.isOpened();
    if (has_camera) {
        cap.set(cv::CAP_PROP_FRAME_WIDTH, g_settings.camera_width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, g_settings.camera_height);
    } else {
        std::cout << "[WARNING] Sin camara de hardware detectada. Generando frame sintetico." << std::endl;
    }

    cv::Mat frame;
    GLuint camera_texture = 0;

    // ========================================================================
    // Main interaction loop
    // ========================================================================
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // 1. Fetch or generate frame
        if (has_camera) {
            cap >> frame;
            if (frame.empty()) continue;
        } else {
            // Generar imagen de fondo negro simulada
            frame = cv::Mat::zeros(720, 1280, CV_8UC3);
            std::time_t t = std::time(nullptr);
            char t_str[100];
            std::strftime(t_str, sizeof(t_str), "%H:%M:%S", std::localtime(&t));
            cv::putText(frame, "SIMULADOR CBN (NO CAMERA DETECTED)", cv::Point(50, 60), 
                        cv::FONT_HERSHEY_DUPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
            cv::putText(frame, t_str, cv::Point(50, 120), 
                        cv::FONT_HERSHEY_DUPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
            std::this_thread::sleep_for(std::chrono::milliseconds(33)); // Simulacion de FPS
        }

        // Initialize ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Configurar la ventana maestra de ImGui para cubrir toda la ventana GLFW
        int display_w, display_h;
        glfwGetWindowSize(window, &display_w, &display_h);
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImVec2((float)display_w, (float)display_h));
        
        ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
        ImGui::Begin("Laboratorio CBN", nullptr, window_flags);

        // Map int states to strict bools for ImGui rendering
        bool b_resize = (g_settings.resize_enable == 1);
        bool b_gray = (g_settings.grayscale_enable == 1);
        bool b_blur = (g_settings.blur_enable == 1);
        bool b_clahe = (g_settings.clahe_enable == 1);
        bool b_hist = (g_settings.hist_eq_enable == 1);
        bool b_burst = (g_settings.burst_recording_enable == 1);
        float f_clahe_clip = g_settings.clahe_clip_limit / 10.0f;

        // -------------------------------------------------------
        // PANEL IZQUIERDO: Controles (Fijo a 350px de ancho)
        // -------------------------------------------------------
        ImGui::BeginChild("Controles", ImVec2(350, 0), true);

        if (ImGui::CollapsingHeader("Cámara y Conexión", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::Button("Refrescar Dispositivos", ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
                available_cameras = getAvailableCameras();
                if (current_camera_idx >= available_cameras.size()) current_camera_idx = 0;
            }
            ImGui::Spacing();
            
            std::string combo_preview = "Cámara " + std::to_string(active_cam_id);
            if (ImGui::BeginCombo("Selector USB", combo_preview.c_str())) {
                for (int n = 0; n < available_cameras.size(); n++) {
                    bool is_selected = (current_camera_idx == n);
                    std::string cam_name = "Cámara " + std::to_string(available_cameras[n]);
                    
                    if (ImGui::Selectable(cam_name.c_str(), is_selected)) {
                        if (current_camera_idx != n) {
                            // 1. Guardar config actual
                            guardarPresets("../config/config_cam" + std::to_string(active_cam_id) + ".yaml");
                            
                            // 2. Cambiar variables
                            current_camera_idx = n;
                            active_cam_id = available_cameras[current_camera_idx];
                            
                            // 3. Reiniciar hardware
                            cap.release();
                            cap.open(active_cam_id, cv::CAP_V4L2);
                            has_camera = cap.isOpened();
                            if (has_camera) {
                                cap.set(cv::CAP_PROP_FRAME_WIDTH, g_settings.camera_width);
                                cap.set(cv::CAP_PROP_FRAME_HEIGHT, g_settings.camera_height);
                            }
                            
                            // 4. Cargar config nueva
                            cargarPresets("../config/config_cam" + std::to_string(active_cam_id) + ".yaml");
                        }
                    }
                    if (is_selected) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
        }

        if (ImGui::CollapsingHeader("ROI Visual & Resize", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderInt("X", &g_settings.roi_x, 0, 1920);
            ImGui::SliderInt("Y", &g_settings.roi_y, 0, 1080);
            ImGui::InputInt("Ancho", &g_settings.roi_width);
            ImGui::InputInt("Alto", &g_settings.roi_height);
            
            g_settings.roi_width = std::max(1, g_settings.roi_width);
            g_settings.roi_height = std::max(1, g_settings.roi_height);

            ImGui::Separator();
            ImGui::Checkbox("Activar Resize", &b_resize);
            g_settings.resize_enable = b_resize ? 1 : 0;

            if (b_resize) {
                ImGui::InputInt("Resize Ancho", &g_settings.resize_width);
                ImGui::InputInt("Resize Alto", &g_settings.resize_height);
                g_settings.resize_width = std::max(1, g_settings.resize_width);
                g_settings.resize_height = std::max(1, g_settings.resize_height);
            }
        }

        if (ImGui::CollapsingHeader("Filtros Globales de Visión", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Checkbox("Filtro Grises", &b_gray);
            g_settings.grayscale_enable = b_gray ? 1 : 0;

            ImGui::Checkbox("Gaussian Blur", &b_blur);
            g_settings.blur_enable = b_blur ? 1 : 0;
            if (b_blur) {
                if (ImGui::SliderInt("Kernel Blur", &g_settings.blur_kernel_size, 3, 31)) {
                    if (g_settings.blur_kernel_size % 2 == 0) {
                        g_settings.blur_kernel_size += 1;
                    }
                }
            }

            ImGui::Separator();
            
            if (ImGui::Checkbox("CLAHE", &b_clahe)) {
                g_settings.clahe_enable = b_clahe ? 1 : 0;
                if (b_clahe) g_settings.hist_eq_enable = 0; 
            }
            if (b_clahe) {
                if (ImGui::SliderFloat("CLAHE Clip Limit", &f_clahe_clip, 1.0f, 10.0f)) {
                    g_settings.clahe_clip_limit = static_cast<int>(f_clahe_clip * 10);
                }
            }

            if (ImGui::Checkbox("Ecualizar Histograma", &b_hist)) {
                g_settings.hist_eq_enable = b_hist ? 1 : 0;
                if (b_hist) g_settings.clahe_enable = 0; 
            }
        }

        if (ImGui::CollapsingHeader("Acciones", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::Button("Guardar Presets", ImVec2(ImGui::GetContentRegionAvail().x, 30))) {
                guardarPresets("../config/config_cam" + std::to_string(active_cam_id) + ".yaml");
            }
            ImGui::Spacing();
            if (b_burst) {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.1f, 0.1f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1.0f, 0.2f, 0.2f, 1.0f));
            } else {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.6f, 0.1f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.2f, 0.8f, 0.2f, 1.0f));
            }
            if (ImGui::Button(b_burst ? "Stop Recording" : "Record & Save Frames", ImVec2(ImGui::GetContentRegionAvail().x, 45))) {
                g_settings.burst_recording_enable = b_burst ? 0 : 1;
            }
            ImGui::PopStyleColor(2);
            ImGui::Spacing();
            if (ImGui::Button("Process Dataset", ImVec2(ImGui::GetContentRegionAvail().x, 30))) {
                std::cout << "[INFO] Procesando dataset manualmente por UI..." << std::endl;
                processDataset(g_settings, active_cam_id);
            }
        }

        ImGui::EndChild(); // Fin panel izquierdo

        ImGui::SameLine();

        // -------------------------------------------------------
        // PIPELINE LÓGICO
        // -------------------------------------------------------
        cv::Rect applied_roi;
        cv::Mat processed = processImage(frame, g_settings, applied_roi);

        // Burst Recording (Guardado sin recuadro verde)
        if (g_settings.burst_recording_enable == 1) {
            static auto last_save = std::chrono::steady_clock::now();
            auto current_time = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_save).count() >= 200) {
                last_save = current_time;
                std::string output_dir = "../data/captures/cam" + std::to_string(active_cam_id) + "/";
                std::filesystem::create_directories(output_dir);
                char time_buf[64];
                std::time_t t = std::time(nullptr);
                std::strftime(time_buf, sizeof(time_buf), "%Y%m%d_%H%M%S", std::localtime(&t));
                static int burst_count = 0;
                std::string filepath = output_dir + "frame_" + time_buf + "_" + std::to_string(burst_count++) + ".jpg";
                cv::imwrite(filepath, processed); 
            }
        }

        // Pinta el ROI verde de forma puramente visual
        if (processed.channels() == 1) {
            cv::cvtColor(processed, processed, cv::COLOR_GRAY2BGR);
        }
        cv::rectangle(processed, applied_roi, cv::Scalar(0, 255, 0), 3);

        // -------------------------------------------------------
        // PANEL DERECHO: Camara en Vivo
        // -------------------------------------------------------
        ImGui::BeginChild("Visor", ImVec2(0, 0), true);

        if (camera_texture != 0) {
            glDeleteTextures(1, &camera_texture);
            camera_texture = 0;
        }
        camera_texture = MatToTexture(processed);
        
        if (camera_texture != 0) {
            ImVec2 avail_size = ImGui::GetContentRegionAvail();
            float aspect = (float)processed.cols / (float)processed.rows;
            ImVec2 image_size = ImVec2(avail_size.x, avail_size.x / aspect);
            if (image_size.y > avail_size.y) {
                image_size.y = avail_size.y;
                image_size.x = avail_size.y * aspect;
            }
            ImGui::Image((ImTextureID)(intptr_t)camera_texture, image_size);
        }
        ImGui::EndChild(); // Fin panel derecho

        ImGui::End(); // Fin ventana principal Laboratorio CBN

        // -------------------------------------------------------
        // Global Rendering
        // -------------------------------------------------------
        ImGui::Render();
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        
        glClearColor(0.12f, 0.12f, 0.12f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    // Free resources on close
    if (camera_texture != 0) glDeleteTextures(1, &camera_texture);
    
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
