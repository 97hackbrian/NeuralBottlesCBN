#pragma once

#include <vector>
#include <optional>
#include "cbn_detector_inference.hpp"

// Clase o funciones para validar la lógica de negocio (Casilleros CBN)
namespace CBNDetector {

    /**
     * @brief Valida si el casillero actual contiene 12 botellas buenas (cap)
     * 
     * @param detections Lista de detecciones entregadas por YOLO26
     * @param target_total El total de espacios que deben sumar entre cap y empty_slot
     * @param target_cap El total de botellas buenas que deben haber para dar "PASA"
     * @return std::optional<bool> 
     *         - std::nullopt si la caja está incompleta (la suma no da target_total)
     *         - true si la caja está completa y tiene exactamente target_cap botellas buenas
     *         - false si la caja está completa pero tiene menos de target_cap botellas buenas
     */
    inline std::optional<bool> validarCasillero(const std::vector<Detection>& detections, 
                                                int target_total = 12, 
                                                int target_cap = 12) {
        int count_cap = 0;
        int count_empty = 0;

        for (const auto& d : detections) {
            // Asumimos que 0 es cap y 1 es empty_slot (según los requerimientos)
            if (d.class_id == 0) {
                count_cap++;
            } else if (d.class_id == 1) {
                count_empty++;
            }
        }

        int total_detected = count_cap + count_empty;

        // Si no se ven exactamente las 12 posiciones, la caja no está completa en la cámara
        if (total_detected != target_total) {
            return std::nullopt; // PENDIENTE
        }

        // Si se ven las 12 posiciones
        if (count_cap == target_cap) {
            return true; // PASA
        } else {
            return false; // RECHAZADO
        }
    }

} // namespace CBNDetector
