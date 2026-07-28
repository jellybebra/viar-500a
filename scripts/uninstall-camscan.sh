#!/bin/sh
set -eu

if [ "$(id -u)" -ne 0 ]; then
    echo "Запустите удаление через sudo." >&2
    exit 1
fi

rm -f /usr/local/bin/viar-scanner
rm -f /usr/share/applications/viar-scanner.desktop
rm -rf /opt/viar-scanner
rm -rf /usr/share/doc/viar-scanner
update-desktop-database /usr/share/applications 2>/dev/null || true

echo "Приложение «Сканер VIAR» удалено."
