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
    std::cout << "Prueba modo estatico: Encender GPO3 (Alarma) por 5 segundos...\n";
    cbn::write_gpo(cbn::GPO_3, cbn::HIGH);
    std::this_thread::sleep_for(std::chrono::seconds(5));
    cbn::write_gpo(cbn::GPO_3, cbn::LOW);
    
    std::this_thread::sleep_for(std::chrono::seconds(1));
    
    // Prueba modo asíncrono
    std::cout << "Prueba modo asincrono: 3 pulsos en GPO2 (Piston) con duracion de 3000ms (3s) y pausa de 3s...\n";
    for (int i = 0; i < 3; ++i) {
        std::cout << "Lanzando pulso " << i + 1 << " en GPO2 (ON por 3s, OFF por 3s)\n";
        cbn::pulse_gpo(cbn::GPO_2, 3000);
        
        // Pausa de 6 segundos en total en el hilo principal (3s que dura el pulso encendido + 3s de apagado)
        std::this_thread::sleep_for(std::chrono::seconds(6));
    }
    
    // Finalizar con fail_safe_reset
    std::cout << "Finalizando test fisico...\n";
    cbn::fail_safe_reset();
    
    // Esperar a que el reset se ejecute antes de terminar
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    std::cout << "Test completado.\n";
    return 0;
}
