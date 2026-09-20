#!/usr/bin/env bash
# Instala el servidor local de Pokemon Showdown que usa el proyecto.
#
# - Descarga el servidor en pokemonshowdown-server/ (ignorado por git), en el
#   mismo commit con el que se desarrollo y probo el proyecto.
# - Activa dos opciones de config/config.js que el entrenamiento necesita:
#     noguestsecurity = true   nombres de usuario cualquiera, sin login real
#     nothrottle      = true   sin limite de 10 s entre desafios seguidos
#
# Requiere git y Node.js (>= 18).
#
# Uso:
#     bash scripts/setup_server.sh                # instala en pokemonshowdown-server/
#     bash scripts/setup_server.sh /otra/carpeta  # instala en otra carpeta
#
# Si la carpeta ya existe no se descarga de nuevo: solo se revisa la config.

set -euo pipefail

SHOWDOWN_REPO="https://github.com/smogon/pokemon-showdown.git"
SHOWDOWN_COMMIT="9e317a666d9fd250f36f494778e843141f09bdba"
SERVER_DIR="${1:-$(cd "$(dirname "$0")/.." && pwd)/pokemonshowdown-server}"

if [ -d "$SERVER_DIR/.git" ]; then
    echo "Ya existe $SERVER_DIR: no se descarga de nuevo."
else
    echo "Descargando Pokemon Showdown (commit ${SHOWDOWN_COMMIT:0:9}) en $SERVER_DIR ..."
    mkdir -p "$SERVER_DIR"
    git -C "$SERVER_DIR" init -q
    git -C "$SERVER_DIR" remote add origin "$SHOWDOWN_REPO"
    git -C "$SERVER_DIR" fetch -q --depth 1 origin "$SHOWDOWN_COMMIT"
    git -C "$SERVER_DIR" checkout -q FETCH_HEAD
    echo "Instalando dependencias (npm install) ..."
    (cd "$SERVER_DIR" && npm install --silent)
fi

CONFIG="$SERVER_DIR/config/config.js"
[ -f "$CONFIG" ] || cp "$SERVER_DIR/config/config-example.js" "$CONFIG"

sed -i.bak \
    -e 's/^exports\.noguestsecurity = false;/exports.noguestsecurity = true;/' \
    -e 's/^exports\.nothrottle = false;/exports.nothrottle = true;/' \
    "$CONFIG"
rm -f "$CONFIG.bak"

grep -q '^exports\.noguestsecurity = true;' "$CONFIG" || { echo "ERROR: no se pudo activar noguestsecurity en $CONFIG"; exit 1; }
grep -q '^exports\.nothrottle = true;' "$CONFIG" || { echo "ERROR: no se pudo activar nothrottle en $CONFIG"; exit 1; }

echo
echo "Listo. Para levantar el servidor (queda escuchando en localhost:8000):"
echo "    cd \"$SERVER_DIR\" && ./pokemon-showdown start"
