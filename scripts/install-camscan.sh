#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
APP_DIR=/opt/viar-scanner
APP_SRC="$APP_DIR/src"
VENV="$APP_DIR/venv"

if [ "$(id -u)" -ne 0 ]; then
    echo "Запустите установщик через sudo." >&2
    exit 1
fi

if [ ! -f "$PROJECT_DIR/third_party/camscan/camscan/app.py" ]; then
    echo "Не найдены исходники Camscan в $PROJECT_DIR/third_party/camscan" >&2
    exit 1
fi

install -d -m 0755 "$APP_SRC" /usr/local/bin /usr/share/applications
cp -a "$PROJECT_DIR/third_party/camscan/camscan" "$APP_SRC/"
install -m 0644 "$PROJECT_DIR/third_party/camscan/utils.py" "$APP_SRC/utils.py"

if [ ! -x "$VENV/bin/python" ]; then
    python3 -m venv "$VENV"
fi

"$VENV/bin/pip" uninstall -y \
    opencv-python \
    opencv-python-headless \
    opencv-contrib-python \
    opencv-contrib-python-headless >/dev/null 2>&1 || true

"$VENV/bin/pip" install --disable-pip-version-check \
    customtkinter==5.2.2 \
    numpy==2.2.6 \
    opencv-contrib-python-headless==4.12.0.88 \
    Pillow==12.0.0

install -m 0755 "$PROJECT_DIR/packaging/viar-scanner" \
    /usr/local/bin/viar-scanner
install -m 0644 "$PROJECT_DIR/packaging/viar-scanner.desktop" \
    /usr/share/applications/viar-scanner.desktop

install -d -m 0755 /usr/share/doc/viar-scanner
install -m 0644 "$PROJECT_DIR/third_party/camscan/LICENSE.md" \
    /usr/share/doc/viar-scanner/LICENSE.camscan.md

update-desktop-database /usr/share/applications 2>/dev/null || true

echo "Установлено: /usr/local/bin/viar-scanner"
echo "Ярлык: «Сканер VIAR» в меню приложений"
