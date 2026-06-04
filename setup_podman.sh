#!/bin/bash
# Nombre: setup_podman.sh
# Descripción: Aprovisionamiento determinista de Podman, ajustado para Desarrollo y Producción.
# Uso Estándar (Desarrollo): sudo ./setup_podman.sh
# Uso en Producción (J1900): sudo ./setup_podman.sh --production

set -e

# --- 1. Verificación de privilegios de superusuario ---
if [ "$EUID" -ne 0 ]; then
  echo "[ERROR] El aprovisionamiento de dependencias del sistema requiere elevación de privilegios (root)."
  exit 1
fi

# --- 2. Análisis de Argumentos (CLI Parsing) ---
IS_PRODUCTION=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -p|--production) 
            IS_PRODUCTION=true
            shift 
            ;;
        *) 
            echo "[ERROR] Parámetro no reconocido: $1"
            echo "Uso: $0 [--production|-p]"
            exit 1 
            ;;
    esac
done

echo "[INFO] Iniciando secuencia de instalación de infraestructura OCI..."

# --- 3. Detección de Microarquitectura de Sistema Operativo ---
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
    OS_LIKE=$ID_LIKE
else
    echo "[ERROR] Imposibilidad de resolver la topología del sistema operativo base."
    exit 1
fi

echo "[INFO] Sistema base detectado: $PRETTY_NAME"

# --- 4. Resolución de Paquetes Base por Sistema Operativo ---
if [[ "$OS" == "debian" || "$OS" == "ubuntu" || "$OS_LIKE" == *"debian"* || "$OS_LIKE" == *"ubuntu"* ]]; then
    # Debian/Ubuntu requieren uidmap explícitamente y `passt` para el backend rootless `pasta`
    PACKAGES="podman uidmap slirp4netns passt"
    PKG_MANAGER="apt"
elif [[ "$OS" == "manjaro" || "$OS" == "arch" || "$OS_LIKE" == *"arch"* ]]; then
    # Arch/Manjaro incluyen uidmap en el paquete base 'shadow'; instalar `passt` para `pasta`
    PACKAGES="podman slirp4netns passt"
    PKG_MANAGER="pacman"
else
    echo "[ERROR] Entorno de distribución no soportado por esta rutina de automatización."
    exit 1
fi

# --- 5. Bifurcación Lógica (Desarrollo vs Producción) ---
if [ "$IS_PRODUCTION" = true ]; then
    echo "[INFO] Perfil: PRODUCCIÓN. Añadiendo herramientas V4L2 y omitiendo Compose."
    PACKAGES="$PACKAGES v4l-utils"
else
    echo "[INFO] Perfil: DESARROLLO. Añadiendo podman-compose."
    if [ "$PKG_MANAGER" == "apt" ]; then
        PACKAGES="$PACKAGES python3-pip"
    elif [ "$PKG_MANAGER" == "pacman" ]; then
        PACKAGES="$PACKAGES podman-compose"
    fi
fi

# --- 6. Ejecución del Gestor de Paquetes ---
if [ "$PKG_MANAGER" == "apt" ]; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -y
    apt-get install -y --no-install-recommends $PACKAGES
    
    if [ "$IS_PRODUCTION" = false ]; then
        echo "[INFO] Instalando podman-compose vía pip3..."
        pip3 install --break-system-packages podman-compose || pip3 install podman-compose
    fi
elif [ "$PKG_MANAGER" == "pacman" ]; then
    pacman -Syu --noconfirm $PACKAGES
fi

# --- 7. Validación de binarios rootless necesarios para Podman ---
if ! command -v pasta >/dev/null 2>&1; then
    if command -v passt >/dev/null 2>&1; then
        echo "[INFO] Detectado backend 'passt'; el binario 'pasta' será resuelto por ese paquete."
    else
        echo "[ERROR] No se encontró 'pasta' ni 'passt' después de la instalación."
        echo "        Instale 'passt' manualmente y vuelva a ejecutar este script."
        exit 1
    fi
fi

# --- 8. Configuración de Registros de Ingesta (OCI) ---
REGISTRIES_DIR="/etc/containers/registries.conf.d"
REGISTRIES_CONF="$REGISTRIES_DIR/00-shortnames.conf"

mkdir -p "$REGISTRIES_DIR"
cat <<EOF > "$REGISTRIES_CONF"
unqualified-search-registries = ["docker.io", "quay.io"]
EOF

# --- 9. Mapeo de Identificadores (Rootless SubUID/SubGID) ---
if [ -n "$SUDO_USER" ]; then
    if ! grep -q "^$SUDO_USER:" /etc/subuid; then
        usermod --add-subuids 100000-165535 --add-subgids 100000-165535 "$SUDO_USER"
        echo "[INFO] Espacio de nombres rootless mapeado para el usuario: $SUDO_USER"
    else
        echo "[INFO] Mapeo de espacio de nombres preexistente detectado. Omitiendo."
    fi
fi

echo "=========================================================="
echo "✅ APROVISIONAMIENTO FINALIZADO CON ÉXITO"
echo "=========================================================="
if [ "$IS_PRODUCTION" = true ]; then
    echo "⚠️ MODO PRODUCCIÓN ACTIVO:"
    echo "Recuerde inicializar el contenedor manualmente y registrarlo en systemd:"
    echo "1. podman run -d --name cbn_inference ... neuralbottles_edge:latest"
    echo "2. podman generate systemd --name cbn_inference --files --new"
    echo "3. systemctl --user enable --now container-cbn_inference.service"
else
    echo "🛠️ MODO DESARROLLO ACTIVO:"
    echo "Utilice 'podman-compose run cbn_builder' para iniciar la compilación cruzada."
fi
echo "=========================================================="
