# Documentación: Librería GPIO para Super I/O (IT8786E)

La librería `cbn_io.h` proporciona una interfaz simple, segura y asíncrona en C++ para controlar los pines de salida de propósito general (GPO) del chip Super I/O ITE IT8786E en la computadora industrial. 

Se basa en la utilidad `gpioset` de Linux para modificar directamente las líneas lógicas del dispositivo de hardware.

## Requisitos Previos

- El dispositivo de hardware `/dev/gpiochip0` debe existir y tener permisos de lectura/escritura (en Docker, debe estar mapeado con la directiva `devices`).
- El paquete de sistema `gpiod` debe estar instalado en el entorno (o contenedor) donde se ejecuta, ya que provee el comando base `gpioset`.

---

## Enumeraciones Disponibles

### `cbn::Pin`
Define qué pin físico deseas controlar. Los valores corresponden a las líneas lógicas descubiertas en `gpiochip0`.
- `cbn::GPO_2 = 19`
- `cbn::GPO_3 = 32`

### `cbn::State`
Define los estados eléctricos de salida.
- `cbn::LOW = 0`: Apagado / 0 Voltios.
- `cbn::HIGH = 1`: Encendido / Voltaje de salida (3.3V)

---

## Funciones de la API

### 1. `cbn::fail_safe_reset()`
> [!IMPORTANT]
> Esta función apaga inmediatamente todas las líneas críticas de salida.

- **Uso recomendado:** Se debe llamar de manera obligatoria **al principio del programa** (para limpiar estados anómalos que quedaron de una ejecución fallida) y **al finalizar el programa** de forma controlada.

**Ejemplo de uso:**
```cpp
int main() {
    cbn::fail_safe_reset(); 
    // Tu lógica de programa...
```

### 2. `cbn::write_gpo(Pin pin, State estado)`
Cambia el estado de un pin de manera estática y lo mantiene en ese estado indefinidamente.
La operación es "no bloqueante" ya que envía el comando del sistema en segundo plano.

- **Uso recomendado:** Encender alerta prolongada, activar un paro de emergencia de la cinta, o encender luces led de estado.

**Ejemplo de uso:**
```cpp
// Encender indefinidamente
cbn::write_gpo(cbn::GPO_3, cbn::HIGH); 

// Para apagarla cuando corresponda:
cbn::write_gpo(cbn::GPO_3, cbn::LOW);  
```

### 3. `cbn::pulse_gpo(Pin pin, int duration_ms)`
Lanza un hilo asíncrono e independiente (`std::thread::detach()`) que enciende el pin, espera exactamente el tiempo indicado en milisegundos (`duration_ms`), y luego lo apaga.

> [!TIP]
> Dado que es asíncrono, llamar a esta función **no congela** el hilo principal de tu programa. Es decir, la cámara y la red neuronal seguirán corriendo a los mismos FPS mientras se puede energizar y apagar un contactor.

- **Uso recomendado:** Disparar eyectores rápidos, sopladores de aire o pistones neumáticos para retirar una botella defectuosa.

**Ejemplo de uso:**
```cpp
if (botella_es_defectuosa) {
    // Activa un soplido rápido de 150 milisegundos de forma asíncrona.
    // El programa pasará de largo esta instrucción instantáneamente.
    cbn::pulse_gpo(cbn::GPO_2, 150); 
}
```

---

## Ejemplo: Integración Final en `main.cpp`

A continuación, un esquema base recomendado de cómo ensamblar esto en tu código final de inferencia:

```cpp
#include <iostream>
#include "cbn_io.h"

int main() {
    std::cout << "Inicializando sistema de visión y hardware...\n";
    
    // 1. Reseteo Inicial Obligatorio
    cbn::fail_safe_reset();
    
    // Configuración de OpenVINO / OpenCV iría aquí...
    
    bool ejecutando = true;
    while (ejecutando) {
        
        // ... (1) Lectura de frame (OpenCV) ...
        // ... (2) Inferencia (OpenVINO) ...
        
        bool defecto_detectado = false; // El resultado de tu red
        
        if (defecto_detectado) {
            std::cout << "Defecto encontrado! Descartando...\n";
            // Disparar pulso ultra-rápido al pistón (ej. 100ms) sin perder FPS
            cbn::pulse_gpo(cbn::GPO_2, 100); 
        }
        
        // ... lógica de fin de ciclo o tecla Esc ...
    }
    
    // 2. Apagado Seguro de Hardware
    std::cout << "Apagando componentes...\n";
    cbn::fail_safe_reset();
    
    return 0;
}
```
