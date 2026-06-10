#include <iostream>
#include <thread>
#include <chrono>
#include "cbn_io.h"

int main() {
    std::cout << "Inicio del test fisico de GPIO (Super I/O IT8786E)\n";
    
    // Ejecutar fail_safe_reset para inicializar
    cbn::fail_safe_reset();
    
    // Pequeña pausa para asegurar la inicialización
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // Prueba modo estático
    std::cout << "Prueba modo estatico: Encender GPO3 (Alarma) por 2 segundos...\n";
    cbn::write_gpo(cbn::GPO_3, cbn::HIGH);
    std::this_thread::sleep_for(std::chrono::seconds(2));
    cbn::write_gpo(cbn::GPO_3, cbn::LOW);
    
    std::this_thread::sleep_for(std::chrono::seconds(1));
    
    // Prueba modo asíncrono
    std::cout << "Prueba modo asincrono: 3 pulsos en GPO2 (Piston) con duracion de 150ms...\n";
    for (int i = 0; i < 3; ++i) {
        std::cout << "Lanzando pulso " << i + 1 << " en GPO2\n";
        cbn::pulse_gpo(cbn::GPO_2, 150);
        
        // Pausa de 1 segundo entre cada llamado en el hilo principal
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    
    // Finalizar con fail_safe_reset
    std::cout << "Finalizando test fisico...\n";
    cbn::fail_safe_reset();
    
    // Esperar a que el reset se ejecute antes de terminar
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    std::cout << "Test completado.\n";
    return 0;
}
