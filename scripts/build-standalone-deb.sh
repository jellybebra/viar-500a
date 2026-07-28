#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
VERSION=${1:-0.4.0}
ARCH=$(dpkg --print-architecture)
OUTPUT="$PROJECT_DIR/viar-scanner_${VERSION}_${ARCH}.deb"
BUILD_DIR=$(mktemp -d "${TMPDIR:-/tmp}/viar-scanner-build.XXXXXX")

cleanup() {
    rm -rf "$BUILD_DIR"
}
trap cleanup EXIT HUP INT TERM

if [ "$ARCH" != "amd64" ]; then
    echo "Эта сборка проверена только для amd64, текущая архитектура: $ARCH" >&2
    exit 1
fi

for command in python3 dpkg-deb sed du awk; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "Не найдена команда сборки: $command" >&2
        exit 1
    fi
done

VENV="$BUILD_DIR/venv"
python3 -m venv "$VENV"
"$VENV/bin/pip" install --disable-pip-version-check \
    customtkinter==5.2.2 \
    numpy==2.2.6 \
    opencv-contrib-python-headless==4.12.0.88 \
    Pillow==12.0.0 \
    pyinstaller==6.16.0 \
    pyinstaller-hooks-contrib==2025.9 \
    pytest==8.4.2

(
    cd "$PROJECT_DIR/third_party/camscan"
    PYTHONPATH="$PROJECT_DIR/third_party/camscan" "$VENV/bin/pytest" tests
)

"$VENV/bin/pyinstaller" \
    --noconfirm \
    --clean \
    --onedir \
    --windowed \
    --name viar-scanner \
    --distpath "$BUILD_DIR/dist" \
    --workpath "$BUILD_DIR/pyinstaller-work" \
    --specpath "$BUILD_DIR" \
    --paths "$PROJECT_DIR/third_party/camscan" \
    --collect-all customtkinter \
    --hidden-import PIL._tkinter_finder \
    "$PROJECT_DIR/packaging/viar-scanner-entry.py"

PACKAGE_ROOT="$BUILD_DIR/package"
install -d -m 0755 \
    "$PACKAGE_ROOT/DEBIAN" \
    "$PACKAGE_ROOT/opt/viar-scanner" \
    "$PACKAGE_ROOT/usr/share/applications" \
    "$PACKAGE_ROOT/usr/share/doc/viar-scanner"

cp -a "$BUILD_DIR/dist/viar-scanner" "$PACKAGE_ROOT/opt/viar-scanner/app"
install -m 0644 "$PROJECT_DIR/packaging/viar-scanner-standalone.desktop" \
    "$PACKAGE_ROOT/usr/share/applications/viar-scanner.desktop"
install -m 0644 "$PROJECT_DIR/third_party/camscan/LICENSE.md" \
    "$PACKAGE_ROOT/usr/share/doc/viar-scanner/LICENSE.camscan.md"
install -m 0755 "$PROJECT_DIR/packaging/standalone-postinst" \
    "$PACKAGE_ROOT/DEBIAN/postinst"

INSTALLED_SIZE=$(du -sk "$PACKAGE_ROOT" | awk '{print $1}')
sed \
    -e "s/@VERSION@/$VERSION/g" \
    -e "s/@ARCH@/$ARCH/g" \
    -e "s/@INSTALLED_SIZE@/$INSTALLED_SIZE/g" \
    "$PROJECT_DIR/packaging/standalone-control.in" \
    > "$PACKAGE_ROOT/DEBIAN/control"

rm -f "$OUTPUT"
# Most bundled wheels already contain compressed native libraries.  Fast gzip
# avoids spending minutes recompressing them and makes remote builds reliable.
dpkg-deb --build --root-owner-group -Zgzip -z1 "$PACKAGE_ROOT" "$OUTPUT"

echo "Готов автономный установщик: $OUTPUT"
