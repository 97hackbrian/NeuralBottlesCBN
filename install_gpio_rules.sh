#!/bin/bash

# ==============================================================================
# Verificación de Privilegios de Ejecución
# ==============================================================================
if [ "$EUID" -ne 0 ]; then
    echo "[-] Error: Este script requiere privilegios de superusuario (root)."
    echo "    Por favor, ejecútelo utilizando: sudo ./instalar_persistencia.sh"
    exit 1
fi

# ==============================================================================
# Definición Estricta de Rutas Absolutas
# ==============================================================================
KERNEL_VERSION=$(uname -r)
DIR_DESTINO="/lib/modules/$KERNEL_VERSION/kernel/drivers/gpio"

# Ruta absoluta al archivo compilado previamente
RUTA_ORIGEN_KO="/home/tlg/moxa-it87-gpio-driver/gpio-it87.ko"
REGLAS_UDEV="/etc/udev/rules.d/99-custom-permissions.rules"

echo "========================================================================"
echo " Iniciando Configuración de Persistencia de Hardware y Permisos Udev"
echo "========================================================================"

# ==============================================================================
# 1. Instalación y Persistencia del Módulo del Kernel
# ==============================================================================
if [ ! -f "$RUTA_ORIGEN_KO" ]; then
    echo "[-] Error Crítico: No se encontró el módulo binario en la ruta absoluta:"
    echo "    $RUTA_ORIGEN_KO"
    echo "    Asegúrese de que el directorio y el archivo existan antes de continuar."
    exit 1
fi

echo "[+] Creando directorio de destino en el árbol de módulos del kernel..."
mkdir -p "$DIR_DESTINO"

echo "[+] Copiando el controlador binario desde su ubicación de origen..."
cp "$RUTA_ORIGEN_KO" "$DIR_DESTINO/"

echo "[+] Actualizando el mapa de dependencias del kernel (depmod)..."
depmod -a

# Registrar el módulo para que cargue automáticamente en cada inicio del sistema
if ! grep -q "gpio-it87" /etc/modules; then
    echo "[+] Agregando 'gpio-it87' a /etc/modules para carga en el arranque..."
    echo "gpio-it87" >> /etc/modules
else
    echo "[*] El módulo 'gpio-it87' ya se encuentra registrado para el arranque."
fi

# ==============================================================================
# 2. Creación Estructurada de Reglas Udev (Permisos Automatizados)
# ==============================================================================
echo "[+] Generando reglas udev en $REGLAS_UDEV..."

cat << 'EOF' > "$REGLAS_UDEV"
# Regla de acceso para el controlador de pines industriales GPIO
KERNEL=="gpiochip*", MODE="0666"

# Regla de acceso para los dispositivos de captura de video y cámaras
KERNEL=="video*", MODE="0666"
EOF

# ==============================================================================
# 3. Aplicación Inmediata de Cambios sin Reiniciar
# ==============================================================================
echo "[+] Recargando el demonio udev y aplicando nuevas reglas de permisos..."
udevadm control --reload-rules
udevadm trigger

echo "[+] Forzando la carga inicial del módulo mediante modprobe..."
if modprobe gpio-it87; then
    echo "[+] Módulo 'gpio-it87' cargado con éxito en memoria."
else
    echo "[-] Advertencia: No se pudo cargar el módulo inmediatamente con modprobe."
fi

echo "========================================================================"
echo " INSTALACIÓN COMPLETADA EXITOSAMENTE"
echo "========================================================================"
